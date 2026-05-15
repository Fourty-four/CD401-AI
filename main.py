"""
시력측정용 음성 도우미 API (FastAPI)

흐름:
  1) 키오스크가 검사 종류(방향/숫자 등)를 알려주면 → 사용자에게 무엇이 보이는지 자연스럽게 묻는 문장을 돌려준다.
  2) 마이크(STT)로 들어온 대답을 해석해 방향/숫자로 정리한다. (Gemma 4 또는 규칙 폴백)
  3) 화면/검사에서 정답으로 쓰는 값과 비교해 맞는지와, 다음에 들릴 만한 한 마디를 함께 돌려준다.

실행: pip install fastapi uvicorn pydantic
      uvicorn main:app --reload --host 0.0.0.0 --port 8000

Gemma / LLM 백엔드 (택일 또는 자동):

  A) 라즈베리파이 등 자체 서버 — Ollama
     - Pi에서 `ollama serve` 후 예: `ollama pull gemma3:4b`
     - 이 앱 쪽: `OLLAMA_BASE_URL=http://라즈베리IP:11434`
     - `OLLAMA_MODEL=gemma3:4b` (Pi에 깔린 태그와 동일하게)
     - 선택: `OLLAMA_TIMEOUT_SEC` (기본 120, 느린 장비용)

  B) Google 호스팅 — Gemini API의 Gemma 4
     - `GEMINI_API_KEY` 또는 `GOOGLE_API_KEY`, 선택 `GEMMA_MODEL` (기본 gemma-4-31b-it)

  `VISION_LLM_MODE`:
    auto   — Ollama URL이 있으면 Ollama 우선 → 실패 시 Gemini(키 있을 때) → 규칙 폴백
    ollama — Ollama만 (URL 필수)
    gemma  — Gemini API만 (키 필수)
    regex  — 규칙만
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import urllib.error
import urllib.request
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="시력측정 음성 도우미", version="0.4.0")

logger = logging.getLogger(__name__)

# Google AI Gemini API — Gemma 4 hosted models
# https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api
GEMINI_GENERATE_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
DEFAULT_GEMMA_MODEL = "gemma-4-31b-it"
LLM_HTTP_TIMEOUT_SEC = 45

# Ollama (자체 호스트, 예: Raspberry Pi)
# https://github.com/ollama/ollama/blob/main/docs/api.md
DEFAULT_OLLAMA_MODEL = "gemma3:4b"
OLLAMA_CHAT_PATH = "/api/chat"
DEFAULT_OLLAMA_TIMEOUT_SEC = 120

# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------

TaskKind = Literal["direction", "number"]


class QuestionRequest(BaseModel):
    """어떤 유형의 검사인지. (정답은 아직 묻지 않음 — 질문 문구만 생성)"""

    kind: TaskKind = Field(..., description="direction 또는 number")
    variant_seed: Optional[int] = Field(
        None,
        description="같은 kind라도 문장 톤을 고정하고 싶을 때 (예: 세션 id 해시). 없으면 무작위 톤.",
    )


class QuestionResponse(BaseModel):
    question_text: str = Field(..., description="TTS/자막에 쓸 질문 한 줄")
    tone: str = Field(..., description="선택된 말투 태그 (예: calm, cheerful)")


DirectionCode = Literal["UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"]


class InterpretedAnswer(BaseModel):
    """STT → 정규화된 추측."""

    kind: TaskKind
    value: str = Field(
        ...,
        description="direction이면 UP|DOWN|LEFT|RIGHT|UNKNOWN, number면 0-9 또는 UNKNOWN",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="해석 신뢰도(모의/LLM 공통 스키마)")


class CheckAnswerRequest(BaseModel):
    """정답(검사에서 기대하는 입력) + 사용자 발화(STT)."""

    expected_kind: TaskKind
    expected_value: str = Field(
        ...,
        description="direction: UP|DOWN|LEFT|RIGHT, number: 한 자리 0-9",
    )
    user_speech: str = Field(..., min_length=1, description="STT 결과 전체 문장")


class CheckAnswerResponse(BaseModel):
    interpreted: InterpretedAnswer
    matches_expected: bool
    assistant_reply: str = Field(
        ...,
        description="사용자에게 들려줄 한두 문장(맞음/틀림/재요청 등)",
    )


# ---------------------------------------------------------------------------
# 질문 템플릿 (정답 노출 없이 ‘대화하는 느낌’)
# ---------------------------------------------------------------------------

QUESTION_POOL: dict[TaskKind, dict[str, list[str]]] = {
    "direction": {
        "calm": [
            "화면에 나온 표시가, 어느 방향으로 뚫려 보이시나요? 천천히 말씀해 주세요.",
            "기호가 가리키는 쪽이 위·아래·왼쪽·오른쪽 중 어디로 보이세요?",
        ],
        "cheerful": [
            "자, 이번엔 방향이에요. 보이는 쪽으로 편하게 말씀해 주세요—위, 아래, 왼쪽, 오른쪽!",
            "눈에 보이는 방향 그대로, 말로만 알려 주시면 돼요.",
        ],
    },
    "number": {
        "calm": [
            "지금 보이는 숫자를 그대로 말씀해 주시겠어요?",
            "숫자 하나만 크게 말씀해 주세요.",
        ],
        "cheerful": [
            "숫자가 보이시죠? 그 숫자를 한 번만 말해 주세요.",
            "보이는 숫자, 편한 말투로 알려 주세요. 영~구까지요.",
        ],
    },
}


def pick_question(kind: TaskKind, variant_seed: Optional[int]) -> tuple[str, str]:
    tones = list(QUESTION_POOL[kind].keys())
    if variant_seed is not None:
        tone = tones[variant_seed % len(tones)]
        pool = QUESTION_POOL[kind][tone]
        text = pool[variant_seed % len(pool)]
        return text, tone
    tone = random.choice(tones)
    text = random.choice(QUESTION_POOL[kind][tone])
    return text, tone


# ---------------------------------------------------------------------------
# 방향/숫자 한국어 표기 (피드백용)
# ---------------------------------------------------------------------------

DIR_KO = {
    "UP": "위쪽",
    "DOWN": "아래쪽",
    "LEFT": "왼쪽",
    "RIGHT": "오른쪽",
    "UNKNOWN": "잘 모르겠다고 하신 부분",
}

NUM_KO_WORDS = {
    "0": "영",
    "1": "일",
    "2": "이",
    "3": "삼",
    "4": "사",
    "5": "오",
    "6": "육",
    "7": "칠",
    "8": "팔",
    "9": "구",
}


# 사투리·구어 포함 키워드 → 정규 코드 (LLM 붙이기 전까지 규칙으로 넓게 잡음)
DIR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 표준어
    (re.compile(r"위쪽?|위로|위에|상단|UP", re.I), "UP"),
    (re.compile(r"아래쪽?|아래로|밑|하|다운|DOWN", re.I), "DOWN"),
    (re.compile(r"왼쪽?|좌측|왼|레프트|LEFT", re.I), "LEFT"),
    (re.compile(r"오른쪽?|우측|우\s*쪽|라이트|RIGHT", re.I), "RIGHT"),
    # 구어/줄임
    (re.compile(r"우래|우리\s*쪽|우짝"), "RIGHT"),  # 일부 사투·구어
    (re.compile(r"좌짝|왼\s*짝"), "LEFT"),
    (re.compile(r"아래\s*끝|밑\s*동네"), "DOWN"),
]

NUM_WORD_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"영|공|제로"), "0"),
    (re.compile(r"하나"), "1"),
    (re.compile(r"둘"), "2"),
    (re.compile(r"셋|석"), "3"),
    (re.compile(r"넷"), "4"),
    (re.compile(r"다섯"), "5"),
    (re.compile(r"여섯"), "6"),
    (re.compile(r"일곱"), "7"),
    (re.compile(r"여덟"), "8"),
    (re.compile(r"아홉"), "9"),
]


def interpret_direction_regex(text: str) -> Optional[str]:
    for pat, code in DIR_PATTERNS:
        if pat.search(text):
            return code
    return None


def interpret_number_regex(text: str) -> Optional[str]:
    for pat, digit in NUM_WORD_PATTERNS:
        if pat.search(text):
            return digit
    m = re.search(r"[0-9]", text)
    if m:
        return m.group(0)
    return None


LLM_INTERPRET_SYSTEM = (
    "너는 시력검사 안내원이야. 사용자 발화는 STT 결과라 오인식·사투리·말버릇이 섞일 수 있다. "
    "이번 검사 유형(expected_kind)에 맞춰 해석한다. "
    "expected_kind가 direction이면 kind는 direction, value는 UP,DOWN,LEFT,RIGHT,UNKNOWN 중 하나. "
    "expected_kind가 number이면 kind는 number, value는 0-9 한 자리 또는 UNKNOWN. "
    "추측이 애매하면 value는 UNKNOWN, confidence는 낮게. "
    "설명 문장 없이 JSON 한 줄만 출력한다. 스키마: "
    '{"kind":"direction|number","value":"...","confidence":0.0}'
)


def interpret_speech_with_regex(user_speech: str, expected_kind: TaskKind) -> str:
    """규칙·키워드 기반 백업 해석."""
    text = user_speech.strip()
    if expected_kind == "direction":
        v = interpret_direction_regex(text) or "UNKNOWN"
        conf = 0.85 if v != "UNKNOWN" else 0.2
        return json.dumps({"kind": "direction", "value": v, "confidence": conf}, ensure_ascii=False)
    v = interpret_number_regex(text) or "UNKNOWN"
    conf = 0.9 if v != "UNKNOWN" else 0.25
    return json.dumps({"kind": "number", "value": v, "confidence": conf}, ensure_ascii=False)


def _gemini_api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _vision_llm_mode() -> str:
    return (os.environ.get("VISION_LLM_MODE") or "auto").strip().lower()


def _gemma_model_id() -> str:
    return (os.environ.get("GEMMA_MODEL") or DEFAULT_GEMMA_MODEL).strip()


def _ollama_base_url() -> Optional[str]:
    """예: http://192.168.0.50:11434 — 끝 슬래시 없이도 됨."""
    u = (os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST") or "").strip()
    return u or None


def _ollama_model_id() -> str:
    return (os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()


def _ollama_timeout_sec() -> int:
    try:
        return max(5, int(os.environ.get("OLLAMA_TIMEOUT_SEC", str(DEFAULT_OLLAMA_TIMEOUT_SEC))))
    except ValueError:
        return DEFAULT_OLLAMA_TIMEOUT_SEC


def call_ollama_chat(system_instruction: str, user_text: str) -> str:
    """
    Ollama HTTP API `/api/chat` (stream=false).
    라즈베리파이에서 ollama run으로 올린 Gemma 계열 모델 이름을 OLLAMA_MODEL에 맞춘다.
    """
    base = _ollama_base_url()
    if not base:
        raise RuntimeError("OLLAMA_BASE_URL 또는 OLLAMA_HOST 가 설정되어 있지 않습니다.")

    model = _ollama_model_id()
    url = base.rstrip("/") + OLLAMA_CHAT_PATH
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_text},
        ],
        "options": {"temperature": 0.2},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_ollama_timeout_sec()) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {err_body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama 연결 실패 ({url}): {e}") from e

    msg = body.get("message")
    if not isinstance(msg, dict):
        raise RuntimeError(f"Ollama 응답 형식 오류: {json.dumps(body, ensure_ascii=False)[:500]}")
    content = (msg.get("content") or "").strip()
    if not content:
        raise RuntimeError("Ollama 응답 message.content 가 비어 있습니다.")
    return content


def call_gemma4_generate_content(system_instruction: str, user_text: str) -> str:
    """
    Gemma 4 (Gemini API `generateContent`) 호출 후, 모델이 생성한 텍스트 본문만 반환.
    """
    api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 또는 GOOGLE_API_KEY 가 설정되어 있지 않습니다.")

    model = _gemma_model_id()
    url = f"{GEMINI_GENERATE_TEMPLATE.format(model=model)}?key={api_key}"
    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 256,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_HTTP_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemma API HTTP {e.code}: {err_body[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Gemma API 연결 실패: {e}") from e

    candidates = body.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemma 응답에 candidates 없음: {json.dumps(body, ensure_ascii=False)[:500]}")

    parts = (candidates[0].get("content") or {}).get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    out = "".join(texts).strip()
    if not out:
        raise RuntimeError("Gemma 응답 텍스트가 비어 있습니다.")
    return out


def _user_prompt_for_interpret(user_speech: str, expected_kind: TaskKind) -> str:
    return (
        f"expected_kind: {expected_kind}\n"
        f"user_speech (STT):\n{user_speech.strip()}\n\n"
        "위 발화만 보고 JSON 한 줄로 답해."
    )


def interpret_speech_with_llm(user_speech: str, expected_kind: TaskKind) -> str:
    """
    발화 → JSON 한 줄 (kind, value, confidence).

    - regex: 항상 규칙
    - ollama: Ollama만 (OLLAMA_BASE_URL 필수)
    - gemma: Gemini API만 (키 필수)
    - auto: Ollama(URL 있으면) → 실패 시 Gemini(키 있으면) → 규칙
    """
    mode = _vision_llm_mode()
    if mode == "regex":
        return interpret_speech_with_regex(user_speech, expected_kind)

    user_prompt = _user_prompt_for_interpret(user_speech, expected_kind)

    if mode == "ollama":
        return call_ollama_chat(LLM_INTERPRET_SYSTEM, user_prompt)

    if mode == "gemma":
        return call_gemma4_generate_content(LLM_INTERPRET_SYSTEM, user_prompt)

    # ---- auto ----
    if _ollama_base_url():
        try:
            raw = call_ollama_chat(LLM_INTERPRET_SYSTEM, user_prompt)
            parse_llm_json(raw)
            return raw
        except Exception as e:
            logger.warning("Ollama 해석 실패: %s", e)
            if _gemini_api_key():
                try:
                    raw = call_gemma4_generate_content(LLM_INTERPRET_SYSTEM, user_prompt)
                    parse_llm_json(raw)
                    return raw
                except Exception as e2:
                    logger.warning("Gemini 폴백도 실패: %s", e2)
            return interpret_speech_with_regex(user_speech, expected_kind)

    if _gemini_api_key():
        try:
            raw = call_gemma4_generate_content(LLM_INTERPRET_SYSTEM, user_prompt)
            parse_llm_json(raw)
            return raw
        except Exception as e:
            logger.warning("Gemma(Gemini API) 해석 실패, 규칙 기반으로 폴백: %s", e)
            return interpret_speech_with_regex(user_speech, expected_kind)

    return interpret_speech_with_regex(user_speech, expected_kind)


def parse_llm_json(raw: str) -> dict:
    s = raw.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", s, re.I)
    if fence:
        s = fence.group(1).strip()
    return json.loads(s)


def normalize_expected(expected_kind: TaskKind, expected_value: str) -> str:
    v = expected_value.strip().upper()
    if expected_kind == "direction":
        if v not in ("UP", "DOWN", "LEFT", "RIGHT"):
            raise ValueError("expected_value는 UP, DOWN, LEFT, RIGHT 중 하나여야 합니다.")
        return v
    if not re.fullmatch(r"[0-9]", v):
        raise ValueError("expected_value는 0-9 한 자리여야 합니다.")
    return v


def to_interpreted(data: dict) -> InterpretedAnswer:
    kind = data.get("kind")
    if kind not in ("direction", "number"):
        raise ValueError("kind는 direction 또는 number 여야 합니다.")
    val = str(data.get("value", "UNKNOWN")).upper()
    if kind == "direction":
        if val not in ("UP", "DOWN", "LEFT", "RIGHT", "UNKNOWN"):
            val = "UNKNOWN"
    else:
        if not re.fullmatch(r"[0-9]", val):
            val = "UNKNOWN"
    conf = float(data.get("confidence", 0.0))
    conf = max(0.0, min(1.0, conf))
    return InterpretedAnswer(kind=kind, value=val, confidence=conf)  # type: ignore[arg-type]


def humanize_guess(interp: InterpretedAnswer) -> str:
    if interp.kind == "direction":
        return DIR_KO.get(interp.value, interp.value)
    if interp.value == "UNKNOWN":
        return "숫자를 알아듣지 못했어요"
    return interp.value


def describe_expected(expected_kind: TaskKind, expected_canon: str) -> str:
    """피드백 문장에 넣을 ‘정답 쪽’ 설명."""
    if expected_kind == "direction":
        return DIR_KO.get(expected_canon, expected_canon)
    nw = NUM_KO_WORDS.get(expected_canon, "")
    return f"숫자 {expected_canon}" + (f" ({nw})" if nw else "")


def build_assistant_reply(
    matches: bool,
    interp: InterpretedAnswer,
    expected_kind: TaskKind,
    expected_canon: str,
) -> str:
    if matches:
        return random.choice(
            [
                "네, 말씀하신 대로 잘 들렸어요. 그대로 진행할게요.",
                "좋아요, 확인했어요. 다음 단계로 넘어갈게요.",
                "알겠습니다. 편하게 이어가시면 돼요.",
            ]
        )
    guess_ko = humanize_guess(interp)
    exp_desc = describe_expected(expected_kind, expected_canon)
    if expected_kind == "direction":
        exp_clause = f"{exp_desc} 방향으로 보이신다면"
    else:
        exp_clause = f"{exp_desc}이 보이신다면"

    if interp.value == "UNKNOWN" or interp.confidence < 0.35:
        return (
            "제가 잘 못 들었을 수도 있어요. "
            f"{exp_clause}, 그걸 한 번만 또박또박 말씀해 주시겠어요?"
        )
    if interp.kind == "direction":
        heard = f"{guess_ko} 방향으로 들렸어요"
    else:
        heard = f"숫자 {interp.value}라고 들렸어요"
    return (
        f"지금은 {heard}. "
        f"{exp_clause}, 그렇게만 짧게 한 번 더 말씀해 주세요."
    )


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------


@app.post("/question", response_model=QuestionResponse)
def post_question(req: QuestionRequest) -> QuestionResponse:
    """검사 유형만 보고 사용자에게 묻는 한 마디(말투 포함)."""
    text, tone = pick_question(req.kind, req.variant_seed)
    return QuestionResponse(question_text=text, tone=tone)


@app.post("/check_answer", response_model=CheckAnswerResponse)
def post_check_answer(req: CheckAnswerRequest) -> CheckAnswerResponse:
    """
    STT 문장을 해석하고, 키오스크가 알고 있는 정답과 같은지 판별한다.
    """
    try:
        expected_canon = normalize_expected(req.expected_kind, req.expected_value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        raw = interpret_speech_with_llm(req.user_speech, req.expected_kind)
        data = parse_llm_json(raw)
        interp = to_interpreted(data)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=f"LLM 오류: {e}") from e
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"발화 해석 실패: {e}") from e

    if interp.kind != req.expected_kind:
        # LLM이 잘못된 kind를 주면 expected 기준으로 재해석 시도
        raw2 = interpret_speech_with_llm(req.user_speech, req.expected_kind)
        try:
            interp = to_interpreted(parse_llm_json(raw2))
        except (json.JSONDecodeError, ValueError):
            pass

    matches = interp.value == expected_canon and interp.value != "UNKNOWN"
    reply = build_assistant_reply(matches, interp, req.expected_kind, expected_canon)
    return CheckAnswerResponse(
        interpreted=interp,
        matches_expected=matches,
        assistant_reply=reply,
    )


@app.get("/health")
def health() -> dict:
    """LLM 연동 여부(키·URL 값 자체는 노출하지 않음)."""
    ob = _ollama_base_url()
    return {
        "status": "ok",
        "vision_llm_mode": _vision_llm_mode(),
        "ollama_base_url_set": bool(ob),
        "ollama_model": _ollama_model_id(),
        "gemini_api_key_set": bool(_gemini_api_key()),
        "gemma_model": _gemma_model_id(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
