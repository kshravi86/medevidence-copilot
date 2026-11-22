"""MedEvidence Copilot front end (Week 4, Streamlit).

Talks to the API in app/server.py (API_URL, default http://127.0.0.1:8100). Each browser session gets its
own session_id, so follow-up questions ("and in children?") are resolved by the agent's planner.
Run: streamlit run app/streamlit_app.py
"""
import os
import uuid

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8100").rstrip("/")
VERDICT_COLOR = {"yes": "green", "no": "red", "maybe": "orange"}
REFUSAL_LABEL = {"emergency": "Emergency", "drug_dosing": "Drug dosing", "personal_diagnosis": "Personal diagnosis"}

st.set_page_config(page_title="MedEvidence Copilot", page_icon="🩺", layout="wide")

if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:16]
    st.session_state.turns = []


def api(method, path, **kw):
    try:
        r = requests.request(method, API + path, timeout=300,
                             headers={"X-Client-Id": st.session_state.sid}, **kw)
        if r.status_code == 429:
            return {"status": "error", "error": r.json().get("error", "Rate limit exceeded")}
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"status": "error", "error": f"API unreachable at {API}: {e}"}


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("MedEvidence Copilot")
    st.caption("Research evidence lookup only, **not clinical advice**. Each answer reflects ONE study "
               "from a fixed PubMedQA snapshot, not a systematic review.")
    health = api("GET", "/api/health")
    if health.get("ok"):
        st.success(f"API ok · {health['docs']} abstracts · mode: {health.get('mode')}")
        c = health.get("cache") or {}
        st.caption(f"Cache: {c.get('entries', 0)} entries, {c.get('hits', 0)} hits · "
                   f"rate limit {health.get('rate_limit_per_min')}/min")
    else:
        st.error(health.get("error") or "API not available")
    st.divider()
    st.caption(f"Session `{st.session_state.sid}`")
    if st.button("New conversation"):
        st.session_state.sid = uuid.uuid4().hex[:16]
        st.session_state.turns = []
        st.rerun()
    examples = api("GET", "/api/examples")
    if isinstance(examples, list):
        st.subheader("Examples")
        for ex in examples:
            if st.button(ex["label"], key="ex_" + ex["label"], use_container_width=True):
                st.session_state.pending = ex["question"]


# ---------------------------------------------------------------- render one answer
def render(res):
    status = res.get("status")
    if status == "refused":
        cat = res.get("refusal_category")
        box = st.error if cat == "emergency" else st.warning
        box(f"**Refused: {REFUSAL_LABEL.get(cat, cat)}**\n\n{res.get('message')}")
        return
    if status == "error":
        st.info(f"Something went wrong: {res.get('error')}")
        return
    if status == "insufficient_evidence":
        st.info(res.get("message") or "None of the retrieved abstracts addresses this question.")
    else:
        v = (res.get("verdict") or "").lower()
        conf = res.get("confidence")
        st.markdown(f"### Verdict: :{VERDICT_COLOR.get(v, 'gray')}[**{v.upper()}**]"
                    + (f"  ·  confidence {conf}" if conf else ""))
        st.write(res.get("explanation") or "")
        q = res.get("evidence_quote")
        if q:
            tag = "✅ verified verbatim in the abstract" if res.get("quote_verified") else "⚠️ NOT found verbatim in the abstract"
            st.markdown(f"> {q}")
            st.caption(tag)
        pmid = res.get("citation_pmid")
        if pmid:
            year = next((d.get("year") for d in res.get("retrieved", []) if d.get("pmid") == pmid), None)
            st.markdown(f"Source: [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" + (f" ({year})" if year else ""))
        cr = res.get("critic") or {}
        if cr.get("downgraded"):
            st.warning(f"The critic could not confirm the first verdict, so it was downgraded to **maybe**. "
                       f"Reason: {cr.get('reason')}")
        a, b = st.columns(2)
        if a.button("👍 Helpful", key=f"up_{res.get('_turn')}"):
            api("POST", "/api/feedback", json={"question": res.get("question"), "pmid": pmid,
                                                 "verdict": res.get("verdict"), "rating": "up"})
            st.toast("Thanks for the feedback")
        if b.button("👎 Wrong", key=f"down_{res.get('_turn')}"):
            api("POST", "/api/feedback", json={"question": res.get("question"), "pmid": pmid,
                                                 "verdict": res.get("verdict"), "rating": "down"})
            st.toast("Thanks, recorded for review")
    with st.expander(f"Retrieved abstracts ({len(res.get('retrieved', []))})"):
        for d in res.get("retrieved", []):
            mark = " ← cited" if d.get("pmid") == res.get("citation_pmid") else ""
            st.markdown(f"**#{d.get('rank')} PMID {d.get('pmid')}**{mark}")
            st.caption(d.get("text", "")[:1500])
    with st.expander("How the agent got here (plan, critic, trace)"):
        if res.get("plan"):
            st.write("**Plan**", res["plan"])
        if res.get("critic"):
            st.write("**Critic**", {k: v for k, v in res["critic"].items() if k != "notes"})
        u = res.get("usage") or {}
        st.write(f"LLM calls {u.get('llm_calls')}, tool calls {u.get('tool_calls')}, tokens {u.get('tokens')}, "
                 f"latency {res.get('latency_s')} s" + (" (cached)" if res.get("cached") else ""))
        if res.get("trace"):
            st.dataframe([{k: v for k, v in s.items() if k in ("step", "ms", "ok", "error", "prompt_tokens",
                                                                "completion_tokens", "cached")} for s in res["trace"]],
                         use_container_width=True)


# ---------------------------------------------------------------- chat
st.title("🩺 MedEvidence Copilot")
st.caption("Ask a population-level research question. Follow-up questions use the earlier turns of this conversation.")

for i, (q, res) in enumerate(st.session_state.turns):
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        render(dict(res, _turn=i))

question = st.chat_input("e.g. Does intravenous magnesium relieve acute migraine in the emergency department?")
question = question or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Safety check → planning → searching → drafting → critic review …"):
            res = api("POST", "/api/ask", json={"question": question, "session_id": st.session_state.sid})
        st.session_state.turns.append((question, res))
        render(dict(res, _turn=len(st.session_state.turns) - 1))
