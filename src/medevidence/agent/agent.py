"""MedEvidence agent: safety gate -> planner -> executor (tools) -> critic loop (spec section 2, brief IV).

Hand-rolled orchestration (no framework), so every guard is explicit:
- tool failures: Tools.call retries once and returns an error result; search falls back to BM25
- malformed JSON: one repair turn per LLM call, then a safe fallback for that step
- infinite loops: hard caps on steps, critic revisions (2) and re-plans (1); repeated queries are skipped
- token budget: every LLM call is charged; when the cap is hit the agent stops and returns its best safe answer
- context window: abstracts are dropped (lowest rank first) until the executor prompt fits the 8192-token limit
"""
import time

from .. import metrics
from ..hybrid import rrf
from . import prompts_agent as P
from . import safety
from .state import AgentState, Budget, SessionStore
from .tracing import Trace

CONTEXT_LIMIT = 8192        # endpoint limit, prompt + completion
EXEC_MAX_OUT = 600


def approx_tokens(messages):
    return sum(len(m["content"]) for m in messages) // 3 + 20 * len(messages)


class BudgetExceeded(Exception):
    pass


class Agent:
    def __init__(self, llm, tools, sessions: SessionStore | None = None, budget: dict | None = None,
                 use_critic=True):
        self.llm, self.tools = llm, tools
        self.sessions = sessions
        self.budget_cfg = budget or {}
        self.use_critic = use_critic

    # ---------------------------------------------------------------- LLM + tool wrappers
    def _chat(self, st, tr, name, messages, max_tokens=300):
        """One JSON LLM call with budget accounting and a single repair turn. Returns (obj|None, raw)."""
        raw = None
        for attempt in range(2):
            why = st.budget.exhausted()
            if why:
                raise BudgetExceeded(why)
            with tr.span(f"llm.{name}", attempt=attempt) as sp:
                try:
                    res = self.llm.chat(messages, temperature=0.0, max_tokens=max_tokens, json_mode=True)
                except Exception as e:  # endpoint down / 4xx after retries
                    st.budget.charge(1)
                    st.llm_calls += 1
                    sp["error"] = f"llm_error: {str(e)[:150]}"
                    return None, None
                st.llm_calls += 1
                st.budget.charge(1, res.prompt_tokens + res.completion_tokens)
                sp.update(prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens, cached=res.cached)
                raw = res.text
                obj, err = metrics.parse_json(raw)
                if obj is not None:
                    return obj, raw
                sp["parse_error"] = err
            st.json_repairs += 1
            messages = messages + [{"role": "assistant", "content": raw or ""},
                                   {"role": "user", "content": P.JSON_REPAIR}]
        return None, raw

    def _tool(self, st, tr, name, **kw):
        why = st.budget.exhausted()
        if why:
            raise BudgetExceeded(why)
        with tr.span(f"tool.{name}", **{k: (v if not isinstance(v, str) else v[:120]) for k, v in kw.items()}) as sp:
            res = self.tools.call(name, **kw)
            st.tool_calls += 1
            st.budget.charge(1)
            sp.update(ok=res.ok, **{f"meta_{k}": v for k, v in res.meta.items()})
            if not res.ok:
                sp["error"] = res.error
                st.tool_errors.append({"tool": name, "error": res.error})
        return res

    # ---------------------------------------------------------------- public API
    def ask(self, question, session_id=None):
        t0 = time.time()
        tr = Trace()
        st = AgentState(question=question, session_id=session_id, budget=Budget(**self.budget_cfg))
        out = {"question": question, "session_id": session_id, "status": "error", "refusal_category": None,
               "verdict": None, "explanation": None, "evidence_quote": None, "citation_pmid": None,
               "quote_verified": False, "citation_valid": False, "message": None, "retrieved": [],
               "plan": None, "critic": None, "safety": None, "error": None}
        try:
            with tr.span("agent.ask", session=bool(session_id)):
                if not isinstance(question, str) or not question.strip():
                    out["error"] = "Question is empty."
                elif len(question) > 1500:
                    out["error"] = f"Question exceeds 1500 characters (length: {len(question)})."
                else:
                    if self.sessions and session_id:
                        st.history = self.sessions.history(session_id)
                    self._run(st, tr, out)
        except BudgetExceeded as e:
            self._budget_stop(st, out, str(e))
        except Exception as e:  # never raise to the caller
            out["status"], out["error"] = "error", f"{type(e).__name__}: {e}"
        out["usage"] = {"llm_calls": st.llm_calls, "tool_calls": st.tool_calls, "tokens": st.budget.tokens,
                        "steps": st.budget.steps, "json_repairs": st.json_repairs, "tool_errors": st.tool_errors}
        out["latency_s"] = round(time.time() - t0, 3)
        out["trace"], out["trace_id"] = tr.steps, tr.trace_id
        if self.sessions and session_id and out["status"] != "error":
            self.sessions.append(session_id, out)
        return out

    # ---------------------------------------------------------------- pipeline
    def _run(self, st, tr, out):
        # 1) safety gate (code rules + classifier), before any retrieval or planning
        with tr.span("safety_gate") as sp:
            rule_cat, rule = safety.rule_check(st.question)
            llm_cat = None
            if rule_cat != "emergency":  # emergency rule hit: refuse immediately, no model call
                o, _ = self._chat(st, tr, "safety", safety.classifier_messages(st.question, st.history), max_tokens=60)
                c = (o or {}).get("category")
                llm_cat = c if c in safety.PRECEDENCE else None
            cat = safety.combine(rule_cat, llm_cat)
            sp.update(rule=rule, rule_category=rule_cat, llm_category=llm_cat, category=cat)
        out["safety"] = {"rule": rule, "rule_category": rule_cat, "llm_category": llm_cat, "category": cat}
        if cat:
            out.update(status="refused", refusal_category=cat, message=safety.refusal(cat))
            return

        # 2) planner
        plan = self._plan(st, tr, P.planner_messages(st.question, st.history))
        out["plan"] = plan
        rq = plan["research_question"]

        # 3) executor: search -> read -> draft; one re-plan if nothing relevant was found
        docs = self._gather(st, tr, plan)
        self._set_retrieved(out, docs)
        draft = self._draft(st, tr, rq, docs)
        while draft.get("status") == "insufficient_evidence" and st.budget.max_replans > 0:
            st.budget.max_replans -= 1
            plan2 = self._plan(st, tr, P.replan_messages(st.question, st.history, st.queries_tried), replan=True)
            out["plan"]["replanned_queries"] = plan2["queries"]
            docs = self._gather(st, tr, plan2, keep=docs)
            self._set_retrieved(out, docs)
            draft = self._draft(st, tr, rq, docs)

        if draft.get("status") != "answered":
            out.update(status="insufficient_evidence", message="None of the retrieved abstracts addresses this question.",
                       critic={"approved": None, "revisions": 0, "downgraded": False, "notes": []})
            return

        # 4) critic loop: up to 2 send-backs, then downgrade to "maybe"
        base = P.executor_messages(rq, docs)
        revisions, notes = 0, []
        while True:
            ok, feedback, checks = self._critique(st, tr, rq, docs, draft)
            notes.append({"draft": revisions, "approved": ok, "feedback": feedback, "checks": checks})
            if ok:
                break
            if revisions >= st.budget.max_revisions:
                break
            revisions += 1
            new = self._draft(st, tr, rq, docs, messages=P.revise_messages(base, draft, feedback))
            if new.get("status") == "answered":
                draft = new
        downgraded = not ok and draft.get("verdict") != "maybe"
        if not ok:
            draft = dict(draft, verdict="maybe", confidence="low")
        out["critic"] = {"approved": ok, "revisions": revisions, "downgraded": downgraded,
                         "reason": None if ok else notes[-1]["feedback"], "notes": notes,
                         "draft_verdicts": [d.get("verdict") for d in st.drafts]}
        self._finalize(st, out, draft, docs)

    def _set_retrieved(self, out, docs):
        years = getattr(self.tools, "year", {})
        out["retrieved"] = [{"pmid": p, "rank": i + 1, "text": t, "year": years.get(p)} for i, (p, t) in enumerate(docs)]

    def _plan(self, st, tr, messages, replan=False):
        o, _ = self._chat(st, tr, "replan" if replan else "planner", messages, max_tokens=200)
        o = o or {}
        rq = o.get("research_question") if isinstance(o.get("research_question"), str) else None
        qs = [q.strip() for q in (o.get("queries") or []) if isinstance(q, str) and q.strip()][:3]
        n = o.get("n_read") if o.get("n_read") in (3, 4, 5) else 4
        if not replan:
            qs = [st.question] + [q for q in qs if q.lower() != st.question.lower()]  # always search the user's words too
        qs = [q[:300] for q in qs if q.lower() not in {x.lower() for x in st.queries_tried}][:3]  # no repeated searches
        return {"research_question": rq or st.question, "queries": qs, "n_read": n, "fallback": not o}

    def _gather(self, st, tr, plan, keep=None):
        ranked_lists = []
        for q in plan["queries"]:
            st.queries_tried.append(q)
            r = self._tool(st, tr, "search_abstracts", query=q, k=8)
            if r.ok and r.value:
                ranked_lists.append([(d["pmid"], d["score"]) for d in r.value])
        fused = [p for p, _ in rrf(ranked_lists, top=plan["n_read"])] if ranked_lists else []
        docs = list(keep or [])
        have = {p for p, _ in docs}
        for p in fused:
            if p in have:
                continue
            a = self._tool(st, tr, "get_abstract", pmid=p)
            if a.ok:
                docs.append((p, a.value["text"]))
                have.add(p)
        return docs[: plan["n_read"] + (len(keep) if keep else 0)]

    def _draft(self, st, tr, rq, docs, messages=None):
        if not docs:
            return {"status": "insufficient_evidence"}
        docs = list(docs)
        msgs = messages or P.executor_messages(rq, docs)
        while messages is None and len(docs) > 1 and approx_tokens(msgs) + EXEC_MAX_OUT > CONTEXT_LIMIT:
            docs.pop()  # context-window guard: drop the lowest-ranked abstract
            msgs = P.executor_messages(rq, docs)
        o, raw = self._chat(st, tr, "executor", msgs, max_tokens=EXEC_MAX_OUT)
        if o is None:
            return {"status": "insufficient_evidence", "_error": "malformed_json"}
        d = {k: o.get(k) for k in ("status", "citation_pmid", "verdict", "explanation", "evidence_quote", "confidence")}
        if d["citation_pmid"] is not None:
            d["citation_pmid"] = str(d["citation_pmid"]).strip().removeprefix("PMID").strip()
        st.drafts.append(d)
        return d

    def _critique(self, st, tr, rq, docs, draft):
        """Code checks first (cheap, deterministic), then the LLM critic."""
        fb, checks = [], {}
        pmids = {p for p, _ in docs}
        cited = draft.get("citation_pmid")
        checks["citation_in_results"] = cited in pmids
        if not checks["citation_in_results"]:
            fb.append(f"citation_pmid {cited!r} is not one of the retrieved PMIDs {sorted(pmids)}; cite one of them exactly.")
        fmt_ok, fmt_err = metrics.check_format(draft, None)
        checks["format_ok"] = fmt_ok
        if not fmt_ok:
            fb.append(f"format problem: {fmt_err} (verdict must be yes/no/maybe; explanation 2-4 sentences).")
        if checks["citation_in_results"]:
            v = self._tool(st, tr, "verify_quote", pmid=cited, quote=draft.get("evidence_quote") or "")
            checks["quote_verbatim"] = bool(v.ok and v.value)
            if not checks["quote_verbatim"]:
                fb.append("evidence_quote is not copied word for word from the cited abstract; copy an exact sentence.")
        if fb or not self.use_critic:
            return not fb, fb, checks
        abstract = dict(docs)[cited]
        concl = self._tool(st, tr, "conclusion_sentences", pmid=cited, n=2)
        o, _ = self._chat(st, tr, "critic",
                          P.critic_messages(rq, cited, abstract, concl.value if concl.ok else [], draft), max_tokens=150)
        st.critiques.append(o)
        if o is None:  # critic unavailable: keep the draft that passed the code checks
            checks["critic"] = "unavailable"
            return True, [], checks
        checks.update(matches=o.get("abstract_matches_question"), verdict_supported=o.get("verdict_supported"),
                      critic_verdict=o.get("correct_verdict"), faithful=o.get("explanation_faithful"))
        if o.get("abstract_matches_question") is False:
            fb.append("the cited abstract does not study this question; choose the abstract that does, or answer insufficient_evidence.")
        if o.get("verdict_supported") is False and o.get("correct_verdict") in metrics.VERDICTS \
                and o.get("correct_verdict") != draft.get("verdict"):
            fb.append(f"the verdict does not match the study's results (reviewer reads it as '{o['correct_verdict']}'): "
                      f"{o.get('feedback') or ''}".strip())
        if o.get("explanation_faithful") is False:
            fb.append(f"the explanation adds claims not in the abstract: {o.get('feedback') or ''}".strip())
        return not fb, fb, checks

    def _finalize(self, st, out, draft, docs):
        cited = draft.get("citation_pmid")
        pm = dict(docs)
        out.update(status="answered", verdict=draft.get("verdict"), explanation=draft.get("explanation"),
                   evidence_quote=draft.get("evidence_quote"), citation_pmid=cited, confidence=draft.get("confidence"),
                   citation_valid=cited in pm,
                   quote_verified=bool(cited in pm and metrics.quote_in_abstract(draft.get("evidence_quote"), pm[cited])))

    def _budget_stop(self, st, out, why):
        """Budget/step cap hit: return the last draft only if it is grounded, as a low-confidence 'maybe'."""
        out["error"] = f"stopped: {why}"
        d = st.drafts[-1] if st.drafts else None
        if out.get("status") == "refused":
            out["error"] = None
            return
        if d and d.get("status") == "answered" and d.get("citation_pmid") in {r["pmid"] for r in out["retrieved"]}:
            docs = [(r["pmid"], r["text"]) for r in out["retrieved"]]
            self._finalize(st, out, dict(d, verdict="maybe", confidence="low"), docs)
            out["critic"] = {"approved": False, "revisions": None, "downgraded": True, "reason": f"stopped: {why}"}
        else:
            out["status"] = "error"
