# src/ts_guard/api/main.py
from __future__ import annotations

import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .llm_provider import chat

load_dotenv()

APP = FastAPI()
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------- Lazy helpers ----------


def _get_rag():
    """Import RAG only when an endpoint needs it."""
    try:
        from .rag_qa import answer as rag_answer
        from .rag_qa import search as rag_search

        return rag_answer, rag_search
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")


def _detect_lang(text: str) -> str:
    """Best-effort language detection; default to 'en' if unavailable/fails."""
    try:
        from langdetect import detect as _detect
    except Exception:
        return "en"
    try:
        return _detect(text or "")
    except Exception:
        return "en"


# ---------- Model utilities ----------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib")
_model = None


def _load_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    return _model


# ---------- Schemas ----------


class CallMeta(BaseModel):
    caller: str = Field(..., examples=["+60123456789"])
    callee: str = Field(..., examples=["+60388888888"])
    duration_sec: int = 45
    hour_of_day: int = Field(..., ge=0, le=23)
    country_code: str = "MY"
    is_outbound: bool = False
    recent_calls_from_caller_24h: int = 8
    pct_answered_last_7d: float = Field(..., ge=0, le=1)
    complaints_last_7d: int = 1


class RiskResponse(BaseModel):
    risk_score: float
    risk_label: str


class TriageRequest(BaseModel):
    complaint_text: str
    meta: CallMeta | None = None


class TriageJSON(BaseModel):
    summary: str
    scam_type: str
    actions: list[str]
    sms_en: str
    sms_ms: str
    confidence: float


def risk_label_from_proba(proba: float) -> str:
    return "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"


# ---------- Routes ----------


@APP.get("/health", tags=["health"])
@APP.get("/healthz", tags=["health"])
def health():
    return {"ok": True}


@APP.post("/predict_call_risk", response_model=RiskResponse)
def predict_call_risk(meta: CallMeta):
    model = _load_model()
    df = pd.DataFrame([meta.model_dump()])
    X = df[
        [
            "duration_sec",
            "hour_of_day",
            "is_outbound",
            "recent_calls_from_caller_24h",
            "pct_answered_last_7d",
            "complaints_last_7d",
        ]
    ].astype({"is_outbound": int})
    proba = float(model.predict_proba(X)[0, 1])
    return {"risk_score": proba, "risk_label": risk_label_from_proba(proba)}


@APP.post("/triage")
def triage(req: TriageRequest):
    rag_answer, _ = _get_rag()
    lang = _detect_lang(req.complaint_text)
    out = rag_answer(req.complaint_text, lang_hint=lang, chat_fn=chat)
    try:
        tri = TriageJSON.model_validate(out).model_dump()
    except Exception:
        tri = {"raw": out}
    return {"triage": tri, "language": lang}


@APP.get("/rag/search")
def rag_search_endpoint(q: str, k: int = 5):
    _, rag_search = _get_rag()
    try:
        return {"results": [{"snippet": s, "source": src} for s, src in rag_search(q, k)]}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}")



@APP.get("/rag/answer")
def rag_answer_endpoint(q: str, k: int = 3):
    rag_answer, _ = _get_rag()
    lang = _detect_lang(q)
    return {"answer": rag_answer(q, k=k, lang_hint=lang, chat_fn=chat)}
