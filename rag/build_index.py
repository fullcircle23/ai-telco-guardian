
import os, glob, uuid, chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

BASE_DIR = os.path.dirname(__file__)
KB_DIR = os.path.join(BASE_DIR, "kb")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_docs():
    docs = []
    for path in glob.glob(os.path.join(KB_DIR, "*")):
        if path.lower().endswith(".pdf"):
            reader = PdfReader(path)
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    docs.append((txt, f"{os.path.basename(path)}#p{i+1}"))
        elif path.lower().endswith((".md",".txt",".rtf",".markdown")):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                docs.append((f.read(), os.path.basename(path)))
    return docs

def chunk(text, n=1500, overlap=250):
    i = 0
    L = len(text)
    while i < L:
        yield text[i:i+n]
        i += max(1, n - overlap)
    return

def main():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        existing = [c.name for c in client.list_collections()]
        if "kb" in existing:
            client.delete_collection("kb")
    except Exception:
        pass
    coll = client.get_or_create_collection("kb")
    model = SentenceTransformer(EMBED_MODEL)
    docs = load_docs()
    ids, chunks, metas = [], [], []
    for text, src in docs:
        for ch in chunk(text):
            ids.append(str(uuid.uuid4()))
            chunks.append(ch)
            metas.append({"source": src})
    if not chunks:
        print("No documents found in rag/kb. Add PDFs/MD/TXT and rerun.")
        return
    embeds = model.encode(chunks).tolist()
    coll.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metas)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} docs.")

if __name__ == "__main__":
    main()
