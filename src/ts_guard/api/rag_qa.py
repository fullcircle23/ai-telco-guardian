
import os, re, json
from typing import List, Tuple
import chromadb

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

JSON_SCHEMA = {
  "type": "object",
  "properties": {
    "summary": {"type":"string"},
    "scam_type": {"type":"string"},
    "actions": {"type":"array","items":{"type":"string"}},
    "sms_en": {"type":"string"},
    "sms_ms": {"type":"string"},
    "confidence": {"type":"number","minimum":0,"maximum":1}
  },
  "required": ["summary","scam_type","actions","sms_en","sms_ms","confidence"]
}

_model = None; _client=None; _coll=None

def _lazy_init():
    global _model, _client, _coll
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    if _coll is None:
        _coll = _client.get_or_create_collection("kb")

def embed(texts: List[str]):
    _lazy_init()
    return _model.encode(texts).tolist()

def _require_sbert():
    try:
        from sentence_transformers import SentenceTransformer  # noqa
    except Exception as e:
        raise RuntimeError("sentence-transformers not installed") from e

def search(query: str, k: int = 5) -> List[Tuple[str, str]]:
    _require_sbert()
    from sentence_transformers import SentenceTransformer
    _lazy_init()
    em = embed([query])[0]
    res = _coll.query(query_embeddings=[em], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, [m.get("source","kb") for m in metas]))

def answer(user_text: str, lang_hint: str = "en", chat_fn=None) -> dict:
    _require_sbert()
    from sentence_transformers import SentenceTransformer
    try:
        kb = search(user_text, k=4)
        kb_snips = [d for d,_ in kb]
    except Exception:
        kb_snips = []
    prompt = build_prompt(user_text, kb_snips, lang_hint)
    messages = [
        {"role":"system","content":"You output strictly JSON. No markdown, no prose."},
        {"role":"user","content":prompt}
    ]
    raw = (chat_fn or (lambda m: "{}"))(messages).strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
    try:
        return json.loads(raw)
    except Exception:
        return {"summary": raw[:400], "scam_type":"unknown","actions":[],
                "sms_en":"","sms_ms":"","confidence":0.2}

# def search(query: str, k: int = 4) -> List[Tuple[str, str]]:
#     _lazy_init()
#     em = embed([query])[0]
#     res = _coll.query(query_embeddings=[em], n_results=k)
#     docs = res.get("documents", [[]])[0]
#     metas = res.get("metadatas", [[]])[0]
#     return list(zip(docs, [m.get("source","kb") for m in metas]))

def build_prompt(user_text: str, kb_snippets: List[str], lang_hint: str = "en") -> str:
    kb_join = "\n\n".join("- " + re.sub(r"\s+", " ", s)[:700] for s in kb_snippets)
    import json as _json
    return f"""You are a telco fraud triage assistant for Malaysia. Use the knowledge snippets strictly.
Return ONLY JSON matching this JSON Schema (no commentary, no markdown):
{_json.dumps(JSON_SCHEMA)}

Fill fields with: (1) short summary, (2) likely scam type, (3) recommended actions with policy refs,
(4) bilingual SMS template fields: sms_en and sms_ms, (5) confidence 0-1.

Knowledge:
{kb_join}

Customer complaint/transcript:
{user_text}
"""
