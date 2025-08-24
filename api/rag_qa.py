
import os, re
from typing import List, Tuple
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
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

def search(query: str, k: int = 4) -> List[Tuple[str, str]]:
    _lazy_init()
    em = embed([query])[0]
    res = _coll.query(query_embeddings=[em], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, [m.get("source","kb") for m in metas]))

def build_prompt(user_text: str, kb_snippets: List[str], lang_hint: str = "en") -> str:
    kb_join = "\n\n".join([f"- {re.sub(r'\s+', ' ', s)[:600]}" for s in kb_snippets])
    return f"""You are a telco fraud triage assistant. Use the knowledge snippets and be concise.
Return: (1) short summary, (2) likely scam type, (3) recommended actions with policy references,
(4) bilingual SMS template (EN + Malay), (5) confidence 0-1.

Knowledge:
{kb_join}

Customer complaint / transcript:
{user_text}

If Malay is detected, prioritize Malay in outputs; otherwise produce EN+MS bilingual.
"""

def answer(user_text: str, lang_hint: str = "en", chat_fn=None) -> str:
    kb = search(user_text, k=4)
    kb_snips = [d for d,_ in kb]
    prompt = build_prompt(user_text, kb_snips, lang_hint)
    messages = [
        {"role":"system","content":"You are an expert telco fraud/scam triage assistant for Malaysia."},
        {"role":"user","content":prompt}
    ]
    return (chat_fn or (lambda m: "LLM not configured"))(messages)
