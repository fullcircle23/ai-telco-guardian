
import os, requests

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def chat(messages, temperature=0.2, model="gpt-4o-mini") -> str:
    if PROVIDER == "openai" and OPENAI_API_KEY:
        import httpx
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": model, "messages": messages, "temperature": temperature}
        with httpx.Client(timeout=60.0) as client:
            r = client.post("https://api.openai.com/v1/chat/completions",
                            headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    try:
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json={
            "model": "llama3", "messages": messages, "options": {"temperature": temperature}
        }, timeout=120)
        if resp.ok:
            return resp.json()["message"]["content"].strip()
    except Exception:
        pass
    prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json={
        "model": "llama3", "prompt": prompt
    }, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()
