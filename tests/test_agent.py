"""Unit tests for the Week 3 agent: safety gate, tool errors, malformed JSON, loop and budget caps,
critic send-backs and persistent sessions. A scripted fake LLM and an in-memory corpus are used,
so no endpoint, index or model is needed."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medevidence.agent import prompts_agent as P  # noqa: E402
from medevidence.agent import safety  # noqa: E402
from medevidence.agent.agent import Agent  # noqa: E402
from medevidence.agent.state import SessionStore  # noqa: E402
from medevidence.agent.tools import ToolResult, Tools  # noqa: E402
from medevidence.llm import ChatResult  # noqa: E402

ABS = {
    "111": "OBJECTIVE: To test drug A in adults with asthma. RESULTS: Drug A reduced exacerbations by 40% (p<0.01). "
           "Drug A significantly reduced asthma exacerbations compared with placebo.",
    "222": "OBJECTIVE: To study diet in diabetes. RESULTS: Diet B had no effect on HbA1c.",
}
GOOD = {"status": "answered", "citation_pmid": "111", "verdict": "yes",
        "explanation": "Drug A reduced exacerbations versus placebo. The reduction was significant.",
        "evidence_quote": "Drug A significantly reduced asthma exacerbations compared with placebo.", "confidence": "high"}
APPROVE = {"abstract_matches_question": True, "verdict_supported": True, "correct_verdict": "yes",
           "explanation_faithful": True, "feedback": "ok"}
REJECT = {"abstract_matches_question": True, "verdict_supported": False, "correct_verdict": "no",
          "explanation_faithful": True, "feedback": "results show no effect"}


class FakeLLM:
    """Routes by system prompt; each role pops from its own script (last item repeats)."""

    def __init__(self, **scripts):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls = []

    def chat(self, messages, **kw):
        sysmsg = messages[0]["content"]
        role = ("safety" if sysmsg == safety.CLASSIFIER else "planner" if sysmsg == P.PLANNER else
                "executor" if sysmsg == P.EXECUTOR else "critic" if sysmsg == P.CRITIC else "?")
        self.calls.append(role)
        seq = self.scripts.get(role, [])
        if not seq:
            raise AssertionError(f"unexpected {role} call")
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return ChatResult([text], prompt_tokens=100, completion_tokens=20)


def fake_tools(fail_search=0):
    t = Tools.__new__(Tools)  # skip loading the real index
    t.abs, t.year = dict(ABS), {p: 2020 for p in ABS}
    state = {"fails": fail_search}

    def search(query, k=5):
        if state["fails"] > 0:
            state["fails"] -= 1
            raise ConnectionError("index unavailable")
        return ToolResult(True, [{"pmid": "111", "score": 2.0}, {"pmid": "222", "score": 1.0}][:k])
    t._t_search_abstracts = search
    return t


def agent(llm, tools=None, **budget):
    return Agent(llm, tools or fake_tools(), budget=budget or None)


PLAN = {"research_question": "Does drug A reduce asthma exacerbations in adults?", "queries": ["drug A asthma"], "n_read": 3}
INSCOPE = {"category": "in_scope", "reason": "research"}


# ------------------------------------------------------------------ safety gate
@pytest.mark.parametrize("q", [
    "Can we predict outcome in chest pain patients?",
    "Is epinephrine effective for anaphylaxis in children?",
    "Do higher loading doses of vancomycin improve target attainment?",
    "Does CBT reduce suicidal ideation in adolescents?",
    "Is HER2 status prognostic in breast cancer?",
])
def test_rules_do_not_flag_research_questions(q):
    assert safety.rule_check(q) == (None, None)


@pytest.mark.parametrize("q,cat", [
    ("My dad just collapsed and is not breathing", "emergency"),
    ("I have crushing chest pain right now", "emergency"),
    ("I keep thinking about killing myself", "emergency"),
    ("How many mg of ibuprofen should I give my son?", "drug_dosing"),
    ("My PSA went from 4 to 7, does that mean I have cancer?", "personal_diagnosis"),
])
def test_rules_flag_personal_requests(q, cat):
    assert safety.rule_check(q)[0] == cat


def test_emergency_rule_refuses_without_model_call():
    llm = FakeLLM()
    r = agent(llm).ask("My dad just collapsed and is not breathing, what do I do?")
    assert r["status"] == "refused" and r["refusal_category"] == "emergency"
    assert r["message"] == safety.refusal("emergency") and "911" in r["message"]
    assert llm.calls == [] and r["retrieved"] == []


def test_classifier_refusal_uses_code_template_and_precedence():
    llm = FakeLLM(safety=[{"category": "drug_dosing", "reason": "dose"}])
    r = agent(llm).ask("As a nurse, what is the right starting amount of metoprolol for Mrs K?")
    assert r["status"] == "refused" and r["refusal_category"] == "drug_dosing"
    assert r["message"] == safety.refusal("drug_dosing")
    assert safety.combine("personal_diagnosis", "emergency") == "emergency"


# ------------------------------------------------------------------ happy path
def test_answer_with_verified_citation():
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[GOOD], critic=[APPROVE])
    r = agent(llm).ask("Does drug A help asthma?")
    assert r["status"] == "answered" and r["verdict"] == "yes" and r["citation_pmid"] == "111"
    assert r["quote_verified"] and r["citation_valid"] and r["critic"]["approved"]
    assert r["usage"]["llm_calls"] == 4 and r["usage"]["tool_calls"] >= 3
    assert {s["step"] for s in r["trace"]} >= {"safety_gate", "llm.planner", "tool.search_abstracts", "llm.critic"}


# ------------------------------------------------------------------ error handling
def test_malformed_json_is_repaired_once():
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=["Sure! The verdict is yes.", GOOD], critic=[APPROVE])
    r = agent(llm).ask("Does drug A help asthma?")
    assert r["status"] == "answered" and r["usage"]["json_repairs"] == 1


def test_planner_garbage_falls_back_to_user_question():
    llm = FakeLLM(safety=[INSCOPE], planner=["not json"], executor=[GOOD], critic=[APPROVE])
    r = agent(llm).ask("Does drug A help asthma?")
    assert r["plan"]["fallback"] and r["plan"]["queries"] == ["Does drug A help asthma?"]
    assert r["status"] == "answered"


def test_tool_failure_is_retried_and_reported():
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[GOOD], critic=[APPROVE])
    r = agent(llm, fake_tools(fail_search=1)).ask("Does drug A help asthma?")
    assert r["status"] == "answered"  # retry succeeded
    llm2 = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[GOOD], critic=[APPROVE])
    r2 = agent(llm2, fake_tools(fail_search=99)).ask("Does drug A help asthma?")
    assert r2["status"] == "insufficient_evidence"
    assert r2["usage"]["tool_errors"] and r2["usage"]["tool_errors"][0]["error"].startswith("tool_failed")


def test_bad_tool_args_rejected():
    t = fake_tools()
    assert t.call("search_abstracts", query="", k=5).error.startswith("bad_args")
    assert t.call("get_abstract", pmid="12; DROP").error.startswith("bad_args")
    assert t.call("rm_rf").error.startswith("unknown_tool")


def test_llm_endpoint_down_does_not_raise():
    llm = FakeLLM(safety=[RuntimeError("HTTP 503")], planner=[RuntimeError("HTTP 503")], executor=[RuntimeError("HTTP 503")])
    r = agent(llm).ask("Does drug A help asthma?")
    assert r["status"] in ("insufficient_evidence", "error")


# ------------------------------------------------------------------ critic loop + caps
def test_critic_sends_back_twice_then_downgrades_to_maybe():
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[GOOD], critic=[REJECT])
    r = agent(llm).ask("Does drug A help asthma?")
    assert r["critic"]["revisions"] == 2 and not r["critic"]["approved"]
    assert r["critic"]["downgraded"] and r["verdict"] == "maybe"
    assert llm.calls.count("executor") == 3 and llm.calls.count("critic") == 3


def test_hallucinated_pmid_caught_by_code_check():
    bad = dict(GOOD, citation_pmid="999999")
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[bad, GOOD], critic=[APPROVE])
    r = agent(llm).ask("Does drug A help asthma?")
    assert r["citation_pmid"] == "111" and r["critic"]["revisions"] == 1


def test_step_cap_stops_loops():
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[GOOD], critic=[REJECT])
    r = agent(llm, max_steps=6).ask("Does drug A help asthma?")
    assert r["error"] == "stopped: step_limit" and r["usage"]["steps"] <= 6


def test_token_budget_cap():
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[GOOD], critic=[REJECT])
    r = agent(llm, max_tokens=300).ask("Does drug A help asthma?")
    assert r["error"] == "stopped: token_budget" and r["usage"]["tokens"] < 300 + 120


def test_input_validation():
    r = agent(FakeLLM()).ask("   ")
    assert r["status"] == "error"
    r = agent(FakeLLM()).ask("x" * 1501)
    assert r["status"] == "error" and "1500" in r["error"]


# ------------------------------------------------------------------ persistent state
def test_session_history_reaches_planner(tmp_path):
    store = SessionStore(path=tmp_path / "s.sqlite")
    llm = FakeLLM(safety=[INSCOPE], planner=[PLAN], executor=[GOOD], critic=[APPROVE])
    ag = Agent(llm, fake_tools(), sessions=store)
    ag.ask("Does drug A help asthma?", session_id="abc")
    store2 = SessionStore(path=tmp_path / "s.sqlite")  # new process, same file
    assert store2.history("abc")[0]["verdict"] == "yes"
    seen = {}
    orig = llm.chat

    def spy(messages, **kw):
        if messages[0]["content"] == P.PLANNER:
            seen["planner_user"] = messages[1]["content"]
        return orig(messages, **kw)
    llm.chat = spy
    Agent(llm, fake_tools(), sessions=store2).ask("and in children?", session_id="abc")
    assert "Does drug A help asthma?" in seen["planner_user"]


# ------------------------------------------------------------------ FAISS store + ROUGE-L
def test_faiss_flat_matches_numpy_ranking():
    import numpy as np
    from medevidence.vectorstore import FaissStore
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 32)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[7] + 0.01
    _, rows = FaissStore.build(emb, "flat").search(q, 10)
    assert list(rows[0]) == list(np.argsort(-(emb @ q))[:10])
    _, hrows = FaissStore.build(emb, "hnsw").search(q, 10)
    assert hrows[0][0] == 7


def test_rouge_l():
    from medevidence.rouge import rouge_l
    ref = "Drug A reduced asthma exacerbations in adults."
    assert rouge_l(ref, ref)["f"] == 1.0
    assert rouge_l(None, ref)["f"] == 0.0
    part = rouge_l("Drug A reduced exacerbations.", ref)
    assert 0 < part["f"] < 1 and part["p"] == 1.0  # every predicted word is in the reference, in order
