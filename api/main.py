
import os, joblib, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langdetect import detect
from .llm_provider import chat
from .rag_qa import answer as rag_answer, search as rag_search

load_dotenv()
APP = FastAPI(title="TS-Guard API")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "model.joblib")

_model = None
def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

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

@APP.get("/healthz")
def health():
    return {"ok": True}

@APP.post("/predict_call_risk", response_model=RiskResponse)
def predict_call_risk(meta: CallMeta):
    model = _load_model()
    df = pd.DataFrame([meta.model_dump()])
    X = df[[
        "duration_sec","hour_of_day","is_outbound",
        "recent_calls_from_caller_24h","pct_answered_last_7d","complaints_last_7d"
    ]].astype({"is_outbound": int})
    proba = model.predict_proba(X)[0,1]
    label = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"
    return {"risk_score": float(proba), "risk_label": label}

class TriageRequest(BaseModel):
    complaint_text: str
    meta: CallMeta

@APP.post("/triage")
def triage(req: TriageRequest):
    try:
        lang = detect(req.complaint_text or "en")
    except:
        lang = "en"
    out = rag_answer(req.complaint_text, lang_hint=lang, chat_fn=chat)
    return {"triage": out, "language": lang}

@APP.get("/search_kb")
def search_kb(q: str):
    res = rag_search(q, k=5)
    return {"results": [{"snippet": s, "source": src} for s,src in res]}
