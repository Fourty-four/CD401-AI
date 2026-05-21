"""FastAPI + OpenAI GPT 연습.

API 키 넣는 법 (택 1):
  1) 터미널: export OPENAI_API_KEY="sk-..."
  2) 프로젝트 루트에 `.env` 또는 `openAI_API_key.env` 파일:
       OPENAI_API_KEY=sk-...
       OPENAI_MODEL=gpt-4o

실행:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError, APIConnectionError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel, Field

from prompts import DEFAULT_SYSTEM, SYSTEM_VISION_EVALUATOR

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "openAI_API_key.env")

app = FastAPI(title="CD401 · GPT API", version="0.1.0")

# CORS 허용 (site.html에서 API 호출 가능하도록)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

class GptRequest(BaseModel):
    message: str = Field(..., min_length=1, description="사용자 말(프롬프트)")
    system: str | None = Field(None, description="선택")
    model: str | None = Field(None, description="비우면 기본 모델")

class GptResponse(BaseModel):
    reply: str
    model: str

class EvalRequest(BaseModel):
    expected_type: str = Field(..., description="'number' 또는 'direction'")
    expected_value: str = Field(..., description="예: '3' 또는 '90'")
    user_spoken: str = Field(..., description="STT로 인식된 사용자 발화")
    model: str | None = None

class EvalResponse(BaseModel):
    result: str
    message: str
    model: str

def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY 가 없습니다.",
        )
    return OpenAI(api_key=key)

@app.post("/gpt", response_model=GptResponse)
def gpt(req: GptRequest) -> GptResponse:
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
        text = (completion.choices[0].message.content or "").strip()
        return GptResponse(reply=text, model=model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvalResponse)
def evaluate_answer(req: EvalRequest) -> EvalResponse:
    """사용자의 음성 답변(STT)이 정답인지 GPT로 판별합니다."""
    model = req.model or DEFAULT_MODEL
    client = _client()
    
    # 프롬프트에 정답 정보와 사용자 발화 전달
    user_message = (
        f"화면에 표시된 기호 타입: {req.expected_type}\n"
        f"정답 값 (방향은 0=우, 90=하, 180=좌, 270=상): {req.expected_value}\n"
        f"사용자 발화: {req.user_spoken}"
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": SYSTEM_VISION_EVALUATOR},
                {"role": "user", "content": user_message},
            ],
        )
        result_text = completion.choices[0].message.content or "{}"
        result_data = json.loads(result_text)
        
        return EvalResponse(
            result=result_data.get("result", "retry"),
            message=result_data.get("message", "다시 한번 말씀해주세요."),
            model=model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
