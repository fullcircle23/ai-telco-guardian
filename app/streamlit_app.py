
import os, requests, streamlit as st
API_BASE_DEFAULT = os.getenv("API_BASE_URL", "http://localhost:8000")
st.set_page_config(page_title="TS-Guard", layout="wide")
st.title("TS-Guard: Telco Scam Early‑Warning & Triage")
with st.sidebar:
    API_BASE = st.text_input("API Base URL", API_BASE_DEFAULT)
tabs = st.tabs(["📊 Risk Monitor", "🛟 Agent Triage", "📚 Knowledge Search"])
with tabs[0]:
    st.subheader("Quick Risk Score")
    with st.form("risk_form"):
        caller = st.text_input("Caller", "+60123456789")
        callee = st.text_input("Callee", "+60388888888")
        duration = st.number_input("Duration (sec)", 1, 3600, 45)
        hour = st.slider("Hour of day", 0, 23, 10)
        is_outbound = st.checkbox("Outbound?", False)
        recent_calls = st.number_input("Recent calls from caller (24h)", 0, 500, 8)
        pct_ans7 = st.number_input("Pct answered last 7d (0-1)", 0.0, 1.0, 0.5)
        complaints7 = st.number_input("Complaints last 7d", 0, 50, 0)
        submitted = st.form_submit_button("Score")
    if submitted:
        payload = {
            "caller": caller, "callee": callee, "duration_sec": duration,
            "hour_of_day": hour, "country_code": "MY", "is_outbound": is_outbound,
            "recent_calls_from_caller_24h": recent_calls, "pct_answered_last_7d": pct_ans7,
            "complaints_last_7d": complaints7
        }
        r = requests.post(f"{API_BASE}/predict_call_risk", json=payload, timeout=60)
        if r.ok:
            data = r.json()
            st.metric("Risk Score", f"{data['risk_score']:.2f}", data["risk_label"])
        else:
            st.error(f"Error {r.status_code}: {r.text[:300]}")

with tabs[1]:
    st.subheader("LLM‑powered Triage (EN + BM)")
    complaint = st.text_area("Complaint / transcript", height=180,
                             placeholder="Caller asked for TAC code and claimed to be from bank...")
    meta = {
        "caller":"+60123456789","callee":"+60388888888","duration_sec":45,"hour_of_day":10,
        "country_code":"MY","is_outbound":False,"recent_calls_from_caller_24h":8,
        "pct_answered_last_7d":0.5,"complaints_last_7d":1
    }
    if st.button("Run Triage"):
        r = requests.post(f"{API_BASE}/triage", json={"complaint_text": complaint, "meta": meta}, timeout=120)
        if r.ok:
            out = r.json()
            st.json(out["triage"])
            st.caption(f"Detected language: {out.get('language','n/a')}")
        else:
            st.error(f"Error {r.status_code}: {r.text[:300]}")

with tabs[2]:
    st.subheader("Search internal KB")
    q = st.text_input("Query", "tac code scam escalation")
    if st.button("Search"):
        r = requests.get(f"{API_BASE}/search_kb", params={"q": q}, timeout=60)
        if r.ok:
            for item in r.json().get("results", []):
                st.write("•", item["snippet"])
                st.caption(item.get("source","kb"))
        else:
            st.error(f"Error {r.status_code}: {r.text[:300]}")
