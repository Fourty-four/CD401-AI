"""FastAPI + OpenAI GPT 연습.

API 키 넣는 법 (택 1):
  1) 터미널: export OPENAI_API_KEY="sk-..."
  2) 프로젝트 루트에 `.env` 또는 `openAI_API_key.env` 파일:
       OPENAI_API_KEY=sk-...
       OPENAI_MODEL=gpt-5.5
     (둘 다 자동 로드. 비밀 파일은 .gitignore — 커밋 금지)

실행:
  source .venv/bin/activate
  pip install -r requirements.txt
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

호출 예:
  curl -s http://127.0.0.1:8000/gpt -H "Content-Type: application/json" \\
    -d '{"message":"한 문장으로 자기소개 해줘"}'

문서 UI: http://127.0.0.1:8000/docs

프롬프트: prompts.py 수정 (system 기본값). /docs 에서 system 생략 시 적용됨.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import APIError, APIConnectionError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from prompts import DEFAULT_SYSTEM

_ROOT = Path(__file__).resolve().parent
# 기본은 `.env`만 읽음. 같은 폴더의 `openAI_API_key.env`도 읽도록 함 (이미 설정된 변수는 덮어쓰지 않음).
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "openAI_API_key.env")

app = FastAPI(title="CD401 · GPT API", version="0.1.0")

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")


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


def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY 가 없습니다. export, .env, openAI_API_key.env 중 하나로 설정하세요.",
        )
    return OpenAI(api_key=key)


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "default_model": DEFAULT_MODEL,
    }


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
    except AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"API 키 인증 실패: {e}") from e
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"요청 한도: {e}") from e
    except APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI 연결 실패: {e}") from e
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API 오류: {e}") from e

    choice = completion.choices[0].message
    text = (choice.content or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="모델이 빈 답을 반환했습니다.")
    used = completion.model or model
    return GptResponse(reply=text, model=used)
