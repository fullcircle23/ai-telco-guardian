
# TS-Guard (Telco Scam Early‑Warning & Triage)

A real, runnable reference system that blends a **tabular ML risk model** with **LLM‑powered triage and RAG**, exposed via **FastAPI** and a **Streamlit** UI.

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

## GitHub Setup
```bash
git init
git add .
git commit -m "feat: init TS-Guard (LLM+RAG+tabular ML)"
git branch -M main
git remote add origin git@github.com:FULLCIRCLE23/ai-telco-guardian.git
git push -u origin main
```
Then open **Actions** to see CI.

## Structure
```
ai-telco-guardian/
  app/ api/ ml/ rag/ infra/ tests/ .github/ .devcontainer/
```
