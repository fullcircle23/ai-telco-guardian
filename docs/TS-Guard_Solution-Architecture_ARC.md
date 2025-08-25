# TS‑Guard: Telco Scam Early‑Warning & Triage
**Solution Architecture & Design (v1.0)**
*ARC Consultancy & Training • 25 Aug 2025*

---

## Table of Contents
- [1) Executive Summary](#1-executive-summary)
- [2) Goals, Non‑Goals, Success Criteria](#2-goals-nongoals-success-criteria)
- [3) Personas & User Journeys](#3-personas--user-journeys)
- [4) High‑Level Architecture](#4-highlevel-architecture)
- [5) Detailed Design](#5-detailed-design)
- [6) Non‑Functional Requirements](#6-nonfunctional-requirements)
- [7) Deployment Topologies](#7-deployment-topologies)
- [8) Build, Test, and CI/CD](#8-build-test-and-cicd)
- [9) Data & Model Lifecycle (MLOps)](#9-data--model-lifecycle-mlops)
- [10) Security (STRIDE‑lite)](#10-security-stridelite)
- [11) Configuration & Tuning](#11-configuration--tuning)
- [12) Operational Playbook](#12-operational-playbook)
- [13) Cost Considerations](#13-cost-considerations)
- [14) Extensibility Roadmap](#14-extensibility-roadmap)
- [15) Open Questions](#15-open-questions)
- [16) Appendix](#16-appendix)

---



## 1) Executive Summary
TS‑Guard blends a tabular ML risk model for real‑time call‑risk scoring with an LLM‑powered triage assistant grounded by RAG over internal knowledge (SOPs, advisories). Targeted at telco/customer‑care operations in Malaysia/SEA to reduce scam losses, shorten triage time, and improve agent guidance with bilingual (EN/BM) outputs.

**Key capabilities**
- Real‑time call risk scoring (RandomForest on engineered tabular features)
- LLM triage with Retrieval‑Augmented Generation (RAG) over your policies/SOPs
- Streamlit agent UI, FastAPI backend; pluggable LLM (OpenAI or local Ollama)
- Dockerized deployment, CI, basic tests, and extensibility hooks for enterprise hardening

**Outcomes**
- Lower false negatives while controlling false positives to avoid overload
- Faster first response with structured, bilingual guidance and citations
- Continuous improvement via feedback → model and prompt updates

---

## 2) Goals, Non‑Goals, Success Criteria
**Goals**
1. Score call events in near‑real‑time with low/medium/high risk classification.
2. Provide grounded triage guidance with citations to internal KB.
3. Offer simple, demo‑ready app + API with minimal infra overhead.
4. Support English and Bahasa Malaysia automatically.

**Non‑Goals (v1)**
- No automatic call blocking; TS‑Guard advises agents/upstream systems.
- No enterprise IAM/SSO in v1 (env‑based config only).
- No streaming ingestion (Kafka) in v1; batch/HTTP only.

**Success Criteria**
- >0.80 ROC‑AUC on held‑out synthetic (real data will differ)
- <300 ms p95 latency for /predict_call_risk at steady 50 RPS
- 3–8 s p95 for LLM triage at 1 RPS (provider‑dependent)
- ≥25% reduction in triage handle time, ≥15% reduction in escalation errors

---

## 3) Personas & User Journeys
**Personas**
- Contact‑centre agent: checks risk, runs triage, sends bilingual SMS
- Fraud analyst: audits flags, tunes thresholds, curates KB
- Ops/Platform engineer: deploys/monitors services, manages secrets
- Data scientist/ML engineer: retrains model, evaluates drift, iterates prompts

**Primary Journey**
1. Agent receives complaint → uses Risk Monitor to get a risk score.
2. Agent pastes transcript/notes → Triage returns summary, scam type, actions, SMS (EN/BM) with references.
3. Agent optionally searches KB directly when needed.

---

## 4) High‑Level Architecture
**Components**
- Streamlit UI: Risk Monitor, Agent Triage, Knowledge Search tabs
- FastAPI Backend: REST endpoints, feature mapping, LLM switch, RAG query
- Risk Model: RandomForestClassifier; features in ml/features.py
- RAG: Sentence‑Transformers embeddings → ChromaDB persistent store
- LLM Provider: OpenAI (default) or local Ollama (llama3)
- Infra: Dockerfiles, docker‑compose; CI; devcontainer for Codespaces

**Logical Flow**
Agent UI ↔ FastAPI → Risk Model (tabular) and RAG Orchestrator → Vector Store → LLM Provider → response back to UI with guidance and confidence.

---

## 5) Detailed Design
**API Endpoints**
- `GET /healthz` → `{ ok: true }`
- `POST /predict_call_risk` → `{ risk_score: float, risk_label: "low|medium|high" }`
- `POST /triage` → `{ triage: string, language: "en|ms" }`
- `GET /search_kb?q=…` → `[{ snippet, source }]`

**Triage Sequence**
1. UI → API: complaint text + optional meta
2. API → RAG: build query from complaint
3. RAG → Vector Store: kNN search (top‑k chunks)
4. RAG → LLM: prompt with snippets + guardrails
5. LLM → RAG → API → UI: structured triage text

**Data Model (v1)**
CallMeta: duration_sec, hour_of_day, is_outbound, recent_calls_from_caller_24h, pct_answered_last_7d, complaints_last_7d
(Extensible: prefix risk, reputation feeds, geo‑signals)

**Feature Engineering**
- Binary outbound flag
- Late‑night bucket
- Rolling aggregates
- Complaint density window

**Risk Thresholding & Actions**
- `<0.4` → Low: education SMS
- `0.4–0.7` → Medium: caution, KBA, SOP step‑ups
- `≥0.7` → High: escalate, restrict sensitive actions

**Prompting & RAG**
- System prompt fixes role, EN/BM, guardrails
- User prompt includes top‑k KB chunks
- JSON‑mode outputs in v1.1

---

## 6) Non‑Functional Requirements
- Performance: risk API <300ms; triage dominated by LLM
- Reliability: stateless API, persistent volume for Chroma, graceful degradation
- Security: PII minimization, redaction, secrets mgmt
- Observability: logs, metrics, drift detection, traces

---

## 7) Deployment Topologies
- Local: docker‑compose (api + app)
- Cloud: ECS/GKE; Option B with pgvector
- Networking: restrict egress, least privilege

---

## 8) Build, Test, and CI/CD
- CI: lint → quick train → build index → pytest
- Tests: unit/integration; prompt evals; later E2E
- Release: docker tags + versioned model.joblib

---

## 9) Data & Model Lifecycle (MLOps)
- Data: CDRs, complaints, reputation feeds
- Training: schedule retrain, calibrate, monitor drift
- Registry: simple v1; MLflow in v1.2
- Rollback: blue/green or canary

---

## 10) Security (STRIDE‑lite)
- Spoofing, Tampering, Repudiation, Disclosure, DoS, Elevation
- Mitigations: WAF, mTLS, logs, non‑root containers, encryption

---

## 11) Configuration & Tuning
```ini
LLM_PROVIDER=openai|ollama
API_PORT=8000
EMBED_MODEL=all‑MiniLM‑L6‑v2
```
- RAG k=3–6; chunk 600–800 tokens; tune thresholds

---

## 12) Operational Playbook
- Runbooks: LLM outage, VS restore, rollback
- Backups: nightly snapshots
- Dashboards: latency, error rate, LLM spend/day
- Alerts: 5xx, drift, VS timeouts

---

## 13) Cost Considerations
- LLM token usage (triage)
- Two containers + PV infra
- Embeddings: one‑time per chunk

---

## 14) Extensibility Roadmap
- v1.1: JSON outputs, Redis cache, pgvector
- v1.2: MLflow, drift, canary deploy
- v1.3: Kafka ingestion, notification connectors

---

## 15) Open Questions
1. Data access model for CDRs?
2. Approved LLM provider & residency?
3. IR SLAs?

---

## 16) Appendix
**Directory**
```
ai‑telco‑guardian/
 app/ api/ ml/ rag/ infra/ tests/
```

**Quick Commands**
```
python ml/train_tabular.py
python rag/build_index.py
uvicorn api.main:APP --reload --port 8000
streamlit run app/streamlit_app.py
```

**Sample SMS**
- EN: “For your safety, never share TAC/OTP with anyone.”
- BM: “Untuk keselamatan anda, jangan sesekali berkongsi TAC/OTP.”
