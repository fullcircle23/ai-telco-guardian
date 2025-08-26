import json
import os
import re
from typing import List, Tuple, TYPE_CHECKING

RE_SPACE = re.compile(r"\s+")

if TYPE_CHECKING:
    # for type hints only; not imported at runtime
    from sentence_transformers import SentenceTransformer

def _require_chroma():
    """
    Import chromadb only when needed and raise a clear error if missing.
    """
    try:
        import chromadb  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "ChromaDB is not installed. Install with either:\n"
            "  pip install -e '.[rag]'\n"
            "or\n"
            "  pip install 'chromadb>=1.0.13'\n"
        ) from e
    return chromadb


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "scam_type": {"type": "string"},
        "actions": {"type": "array", "items": {"type": "string"}},
        "sms_en": {"type": "string"},
        "sms_ms": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["summary", "scam_type", "actions", "sms_en", "sms_ms", "confidence"],
}

_model = None
_client = None
_coll = None


def _lazy_init():
    global _model, _client, _coll
    if _model is None:
        ST = _require_sbert()
        _model = ST(EMBED_MODEL)
    if _client is None:
        chroma = _require_chroma()
        try:
            _client = chroma.PersistentClient(path=CHROMA_DIR)
        except Exception as e:
            raise RuntimeError(
            f"Failed to open ChromaDB at {CHROMA_DIR!r}: {e}"
            ) from e
    if _coll is None:
        try:
            _coll = _client.get_or_create_collection("kb")
        except Exception as e:
            raise RuntimeError(
                "Failed to get or create ChromaDB collection 'kb': "
                f"{e}"
            ) from e


def embed(texts: List[str]):
    _lazy_init()
    return _model.encode(texts).tolist()


def _require_sbert():
    """
    Import SentenceTransformer only when needed.
    Raises a clear error if the optional dependency isn't installed.
    """
    try:
        from sentence_transformers import SentenceTransformer as ST  # local import
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install with: pip install -e '.[rag]'"
        ) from e
    return ST


def search(query: str, k: int = 5) -> List[Tuple[str, str]]:
    _lazy_init()
    em = embed([query])[0]
    res = _coll.query(query_embeddings=[em], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, [m.get("source", "kb") for m in metas]))


def answer(user_text: str, lang_hint: str = "en", chat_fn=None) -> dict:
    ST = _require_sbert()  # Using ST avoids the F401 “imported but unused” warning.
    try:
        kb = search(user_text, k=4)
        kb_snips = [d for d, _ in kb]
    except Exception:
        kb_snips = []
    prompt = build_prompt(user_text, kb_snips, lang_hint)
    messages = [
        {
            "role": "system",
            "content": "You output strictly JSON. No markdown, no prose.",
        },
        {"role": "user", "content": prompt},
    ]
    raw = (chat_fn or (lambda m: "{}"))(messages).strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
    try:
        return json.loads(raw)
    except Exception:
        return {
            "summary": raw[:400],
            "scam_type": "unknown",
            "actions": [],
            "sms_en": "",
            "sms_ms": "",
            "confidence": 0.2,
        }


def _clean_join_snippets(kb_snippets: list[str] | None, limit: int = 700) -> str:
    snippets = kb_snippets or []
    return "\n\n".join("- " + RE_SPACE.sub(" ", (s or ""))[:limit] for s in snippets)


def build_prompt(user_text: str, kb_snippets: List[str], lang_hint: str = "en") -> str:
    kb_join = _clean_join_snippets(kb_snippets)
    import json as _json

    return (
        "You are a telco fraud triage assistant for Malaysia. "
        "Use the knowledge snippets strictly.\n"
        "Return ONLY JSON matching this JSON Schema "
        "(no commentary, no markdown):\n"
        f"{_json.dumps(JSON_SCHEMA)}\n\n"
        "Fill fields with:\n"
        "  (1) short summary,\n"
        "  (2) likely scam type,\n"
        "  (3) recommended actions with policy refs,\n"
        "  (4) bilingual SMS fields: sms_en and sms_ms,\n"
        "  (5) confidence 0-1.\n\n"
        "Knowledge:\n"
        f"{kb_join}\n\n"
        "Customer complaint/transcript:\n"
        f"{user_text}\n"
    )
