![CI](https://github.com/fullcircle23/ai-telco-guardian/actions/workflows/ci.yml/badge.svg)
# TS-Guard (Telco Scam Early‑Warning & Triage)

A real, runnable reference system that blends a **tabular ML risk model** with **LLM‑powered triage and RAG**, exposed via **FastAPI** and a **Streamlit** UI.

## What's new in this patched build
- ✅ **CORS** enabled for local UI ↔ API
- ✅ **Structured JSON triage** (summary, scam_type, actions, sms_en, sms_ms, confidence)
- ✅ **RAG rebuild hygiene** (clear collection) + char-based chunking
- ✅ Streamlit shows triage with `st.json(...)`
- ✅ Added thresholds unit test

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.sample .env
python ml/train_tabular.py
python rag/build_index.py
uvicorn api.main:APP --reload --port 8000
# new terminal
streamlit run app/streamlit_app.py
```

## Docker
```bash
cd infra && docker-compose up --build
```

## Structure
```
ai-telco-guardian/
  app/ api/ ml/ rag/ infra/ tests/ .github/ .devcontainer/
```
