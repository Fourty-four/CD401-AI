"""FastAPI + OpenAI GPT · STT · TTS.

API 키: openAI_API_key.env 또는 export OPENAI_API_KEY

실행:
  source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

엔드포인트:
  GET  /health
  POST /transcribe  — 음성 파일 → 텍스트 (STT)
  POST /gpt         — 텍스트 → GPT 답변
  POST /voice       — 음성 파일 → STT → GPT → TTS → 한 번에 처리

문서 UI: http://127.0.0.1:8000/docs
"""

import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import APIError, APIConnectionError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from prompts import DEFAULT_SYSTEM, SYSTEM_JUDGE, TRANSCRIBE_PROMPT

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "openAI_API_key.env")

app = FastAPI(title="CD401 · GPT API", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
TRANSCRIBE_MODEL = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
TTS_VOICE = os.environ.get("OPENAI_TTS_VOICE", "nova")

# OpenAI 전사 API 지원 확장자 (문서 기준)
ALLOWED_AUDIO_SUFFIXES = {
    ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm",
}
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25MB


class GptRequest(BaseModel):
    message: str = Field(..., min_length=1, description="사용자 말(프롬프트)")
    system: str | None = Field(
        None,
        description="선택. 없으면 prompts.py 의 DEFAULT_SYSTEM 사용",
    )
    model: str | None = Field(
        None, description="비우면 환경변수 OPENAI_MODEL 또는 gpt-5.5"
    )


class GptResponse(BaseModel):
    reply: str
    model: str


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="전사된 텍스트 → /gpt 의 message 로 사용")
    model: str = Field(..., description="사용한 전사 모델")


class VoiceResponse(BaseModel):
    transcript: str = Field(..., description="STT 결과 (사용자가 말한 텍스트)")
    interpreted: str = Field(..., description="정규화된 답 (예: 3, UP)")
    expected: str | None = Field(None, description="정답 (입력했을 때만)")
    correct: bool | None = Field(None, description="맞으면 true, 틀리면 false (정답 있을 때만)")
    reply: str = Field(..., description="GPT 답변 텍스트")
    audio_base64: str = Field(..., description="TTS mp3 음성 (base64 인코딩)")
    audio_format: str = Field(default="mp3", description="오디오 포맷")


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY 가 없습니다. export, .env, openAI_API_key.env 중 하나로 설정하세요.",
        )
    return OpenAI(api_key=key)


def _raise_openai_http(exc: Exception) -> None:
    if isinstance(exc, AuthenticationError):
        raise HTTPException(status_code=401, detail=f"API 키 인증 실패: {exc}") from exc
    if isinstance(exc, RateLimitError):
        raise HTTPException(status_code=429, detail=f"요청 한도: {exc}") from exc
    if isinstance(exc, APIConnectionError):
        raise HTTPException(status_code=502, detail=f"OpenAI 연결 실패: {exc}") from exc
    if isinstance(exc, APIError):
        raise HTTPException(status_code=502, detail=f"OpenAI API 오류: {exc}") from exc
    raise exc


_VALID_DIRS = frozenset({"UP", "DOWN", "LEFT", "RIGHT"})
_VALID_DIGITS = frozenset("0123456789")

# STT가 정답 숫자의 한글 음절을 비슷하게 잘못 쓴 경우 (정답별로만 적용)
_DIGIT_STT_CONFUSIONS: dict[str, dict[str, str]] = {
    "3": {"참": "3"},
    "4": {"차": "4"},
    "8": {"발": "8", "파": "8"},
}

# 발화에서 숫자 후보로 인식할 한글 (긴 것부터 매칭)
_KOR_DIGIT_WORDS: tuple[tuple[str, str], ...] = (
    ("영", "0"), ("공", "0"), ("일", "1"), ("이", "2"), ("삼", "3"), ("사", "4"),
    ("오", "5"), ("육", "6"), ("륙", "6"), ("칠", "7"), ("팔", "8"), ("구", "9"),
    ("참", "3"), ("차", "4"), ("발", "8"), ("파", "8"),
)

_DIR_WORDS: tuple[tuple[str, str], ...] = (
    ("왼쪽", "LEFT"), ("오른쪽", "RIGHT"), ("위쪽", "UP"), ("아래쪽", "DOWN"),
    ("왼", "LEFT"), ("오른", "RIGHT"), ("위", "UP"), ("아래", "DOWN"),
    ("좌", "LEFT"), ("우", "RIGHT"), ("업", "UP"), ("다운", "DOWN"),
)

_AMBIGUITY_PATTERN = re.compile(
    r"아니면|또는|이거나|아님|아니고|말고|할까|뭐지|뭐였|인가요|인가\?|그건|아냐",
    re.IGNORECASE,
)


def _normalize_interpreted(value: str) -> str:
    v = value.strip().upper()
    if v in _VALID_DIRS:
        return v
    if len(v) == 1 and v in _VALID_DIGITS:
        return v
    return v


def _is_known_answer(value: str) -> bool:
    v = _normalize_interpreted(value)
    return v in _VALID_DIRS or v in _VALID_DIGITS


def _answers_match(interpreted: str, expected: str) -> bool:
    return _normalize_interpreted(interpreted) == _normalize_interpreted(expected)


def _parse_judge_json(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)


def _clean_stt_fragment(text: str) -> str:
    return re.sub(r"[이요요~\s.,!?]", "", text.strip())


def _mentioned_digit_values(text: str) -> set[str]:
    found: set[str] = set()
    remaining = _clean_stt_fragment(text)
    for word, digit in _KOR_DIGIT_WORDS:
        while word in remaining:
            found.add(digit)
            remaining = remaining.replace(word, "", 1)
    for ch in remaining:
        if ch in _VALID_DIGITS:
            found.add(ch)
    return found


def _mentioned_direction_values(text: str) -> set[str]:
    found: set[str] = set()
    remaining = text
    for word, direction in _DIR_WORDS:
        while word in remaining:
            found.add(direction)
            remaining = remaining.replace(word, "", 1)
    return found


def _is_ambiguous_response(transcript: str) -> bool:
    """여러 후보를 말했거나, 숫자/방향이 둘 이상이면 시력검사 규칙상 오답."""
    if _AMBIGUITY_PATTERN.search(transcript):
        return True

    digits = _mentioned_digit_values(transcript)
    if len(digits) >= 2:
        return True

    directions = _mentioned_direction_values(transcript)
    if len(directions) >= 2:
        return True

    return False


def _try_stt_confusion_match(transcript: str, expected_norm: str) -> str | None:
    """정답 숫자에 대해 STT가 흔히 헷갈리는 음절이면 해당 숫자로 보정."""
    if expected_norm not in _VALID_DIGITS:
        return None
    fragment = _clean_stt_fragment(transcript)
    if not fragment:
        return None
    mapping = _DIGIT_STT_CONFUSIONS.get(expected_norm, {})
    digit = mapping.get(fragment)
    if digit and digit == expected_norm:
        return digit
    return None


def _stt_kwargs_base(
    upload_name: str,
    audio_bytes: bytes,
    content_type: str,
    language: str | None,
) -> dict:
    kwargs: dict = {
        "model": TRANSCRIBE_MODEL,
        "file": (upload_name, audio_bytes, content_type),
        "prompt": TRANSCRIBE_PROMPT,
    }
    if language and language.strip():
        kwargs["language"] = language.strip()
    return kwargs


def _feedback_for_judgment(*, correct: bool, interpreted: str) -> str:
    if correct:
        return "잘 보셨네요! 다음으로 넘어가 주세요."
    if _normalize_interpreted(interpreted) in ("UNKNOWN", "AMBIGUOUS"):
        if interpreted == "AMBIGUOUS":
            return "답을 하나만 또박또박 말씀해 주세요."
        return "잘 못 들었어요. 다시 한번 또박또박 말씀해 주세요."
    return "조금 어려우셨나 봐요. 다시 한번 천천히 말씀해 주세요."


def _validate_audio_upload(filename: str | None, size: int) -> str:
    if size <= 0:
        raise HTTPException(status_code=400, detail="빈 오디오 파일입니다.")
    if size > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="파일이 25MB를 초과합니다.")
    suffix = Path(filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 확장자입니다. 허용: {', '.join(sorted(ALLOWED_AUDIO_SUFFIXES))}",
        )
    return suffix or ".webm"


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "default_model": DEFAULT_MODEL,
        "transcribe_model": TRANSCRIBE_MODEL,
        "tts_model": TTS_MODEL,
        "tts_voice": TTS_VOICE,
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: UploadFile = File(..., description="wav, webm, mp3 등"),
    language: str | None = Form(
        "ko",
        description="입력 언어 힌트 (ISO-639-1). 비우려면 빈 문자열 전송",
    ),
) -> TranscribeResponse:
    """음성 파일을 OpenAI Transcriptions API로 텍스트(프롬프트)로 변환합니다."""
    audio_bytes = await file.read()
    suffix = _validate_audio_upload(file.filename, len(audio_bytes))
    content_type = file.content_type or "application/octet-stream"
    upload_name = file.filename or f"audio{suffix}"

    client = _client()
    kwargs = _stt_kwargs_base(upload_name, audio_bytes, content_type, language)

    try:
        result = client.audio.transcriptions.create(**kwargs)
    except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
        _raise_openai_http(e)

    text = (result.text or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="전사 결과가 비어 있습니다.")
    used_model = getattr(result, "model", None) or TRANSCRIBE_MODEL
    return TranscribeResponse(text=text, model=used_model)


@app.post("/gpt", response_model=GptResponse)
def gpt(req: GptRequest) -> GptResponse:
    """OpenAI Chat Completions로 한 번 물어보고 답만 돌려줍니다."""
    model = req.model or DEFAULT_MODEL
    system = req.system or DEFAULT_SYSTEM

    client = _client()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": req.message},
            ],
        )
    except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
        _raise_openai_http(e)

    choice = completion.choices[0].message
    text = (choice.content or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="모델이 빈 답을 반환했습니다.")
    used = completion.model or model
    return GptResponse(reply=text, model=used)


@app.post("/voice", response_model=VoiceResponse)
async def voice(
    file: UploadFile = File(..., description="wav, webm, mp3 등"),
    language: str | None = Form("ko", description="언어 힌트 (ISO-639-1)"),
    expected: str | None = Form(None, description="정답 (예: 3, UP). 비우면 판정 안 함"),
) -> VoiceResponse:
    """음성 파일 하나로 STT → 정답 비교 → GPT 피드백 → TTS 한 번에 처리합니다."""
    # --- 1) STT ---
    audio_bytes = await file.read()
    suffix = _validate_audio_upload(file.filename, len(audio_bytes))
    content_type = file.content_type or "application/octet-stream"
    upload_name = file.filename or f"audio{suffix}"

    client = _client()

    stt_kwargs = _stt_kwargs_base(upload_name, audio_bytes, content_type, language)

    try:
        stt_result = client.audio.transcriptions.create(**stt_kwargs)
    except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
        _raise_openai_http(e)

    transcript = (stt_result.text or "").strip()
    if not transcript:
        raise HTTPException(status_code=502, detail="전사 결과가 비어 있습니다.")

    # --- 2) GPT 해석 + 판정 + 피드백 (또는 일반 대화) ---
    interpreted: str = transcript
    correct: bool | None = None
    expected_norm: str | None = None

    has_expected = bool(expected and expected.strip())

    if has_expected:
        expected_norm = expected.strip().upper()

        if _is_ambiguous_response(transcript):
            interpreted = "AMBIGUOUS"
            correct = False
            reply = _feedback_for_judgment(correct=False, interpreted=interpreted)
        else:
            user_msg = f"정답: {expected_norm}\n사용자 발화: {transcript}"
            system_prompt = SYSTEM_JUDGE

            try:
                completion = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )
            except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
                _raise_openai_http(e)

            raw = (completion.choices[0].message.content or "").strip()

            try:
                judge = _parse_judge_json(raw)
                interpreted = _normalize_interpreted(str(judge.get("interpreted", transcript)))
                gpt_correct = judge.get("correct")
                reply = str(judge.get("feedback", "")).strip()
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                interpreted = _normalize_interpreted(transcript)
                gpt_correct = None
                reply = ""

            if interpreted == "AMBIGUOUS" or _is_ambiguous_response(transcript):
                interpreted = "AMBIGUOUS"
                correct = False
                reply = _feedback_for_judgment(correct=False, interpreted=interpreted)
            elif _is_known_answer(interpreted):
                correct = _answers_match(interpreted, expected_norm)
            elif interpreted == "UNKNOWN":
                correct = False
            elif gpt_correct is not None:
                correct = bool(gpt_correct)
            else:
                correct = False

            if interpreted != "AMBIGUOUS":
                stt_fix = _try_stt_confusion_match(transcript, expected_norm)
                if stt_fix is not None:
                    interpreted = stt_fix
                    correct = True
                    reply = _feedback_for_judgment(correct=True, interpreted=interpreted)
                elif correct:
                    if not reply:
                        reply = _feedback_for_judgment(correct=True, interpreted=interpreted)
                elif _is_known_answer(interpreted):
                    reply = _feedback_for_judgment(correct=False, interpreted=interpreted)
                elif not reply:
                    reply = _feedback_for_judgment(correct=False, interpreted=interpreted)

        if not reply:
            raise HTTPException(status_code=502, detail="GPT가 빈 답을 반환했습니다.")
    else:
        try:
            completion = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM},
                    {"role": "user", "content": transcript},
                ],
            )
        except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
            _raise_openai_http(e)

        reply = (completion.choices[0].message.content or "").strip()
        if not reply:
            raise HTTPException(status_code=502, detail="GPT가 빈 답을 반환했습니다.")

    # --- 3) TTS ---
    try:
        tts_response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=reply,
        )
    except (AuthenticationError, RateLimitError, APIConnectionError, APIError) as e:
        _raise_openai_http(e)

    audio_out = tts_response.content
    audio_b64 = base64.b64encode(audio_out).decode("utf-8")

    return VoiceResponse(
        transcript=transcript,
        interpreted=interpreted,
        expected=expected_norm,
        correct=correct,
        reply=reply,
        audio_base64=audio_b64,
    )


_STATIC = _ROOT / "static"
if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
