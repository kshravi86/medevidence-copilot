"""Prompts for the planner, executor and critic (spec section 2).

Few-shot demonstrations come from the PQA-L development split only (prompts.dev_demos), never test PMIDs.
"""
import json

from .. import prompts

# ---- planner ----------------------------------------------------------------------------------
PLANNER = """You are the PLANNER of MedEvidence Copilot, which answers population-level research questions from PubMed abstracts.
Return ONE JSON object:
{"research_question": "the question restated as a neutral population-level yes/no research question",
 "queries": ["1 to 3 short search queries (keywords: population, intervention or exposure, outcome)"],
 "n_read": 3 | 4 | 5}
- Keep the medical terms of the question; add a synonym or the full form of an abbreviation in a second query when useful.
- If the message is a follow-up (e.g. "and in children?"), use the earlier turns to make the research question self-contained.
- n_read: 3 for a specific question, 5 if the question is broad or vague."""

REPLAN = """The abstracts found so far did not address the question. Previous queries: {tried}.
Write 1 to 3 NEW queries that are worded differently (synonyms, broader population, the outcome named another way).
Return the same JSON object."""


def planner_messages(question, history):
    user = ""
    if history:
        user += "Earlier turns in this session:\n" + "\n".join(
            f"- Q: {h.get('question')} -> {h.get('status')} {h.get('verdict') or ''}".rstrip() for h in history) + "\n\n"
    user += f"Message: {question}"
    return [{"role": "system", "content": PLANNER}, {"role": "user", "content": user}]


def replan_messages(question, history, tried):
    return planner_messages(question, history) + [
        {"role": "user", "content": REPLAN.format(tried=json.dumps(tried))}]


# ---- executor ---------------------------------------------------------------------------------
EXECUTOR = """You are the EXECUTOR of MedEvidence Copilot. You get a research question and several retrieved abstracts, each headed [PMID n]. Use ONLY these abstracts.
1. Pick the ONE abstract that studies this question (same population, intervention or exposure, and outcome). Prefer an abstract whose results state a clear finding.
2. Decide the verdict from that study's own results:
   - "yes": the findings support the proposition.
   - "no": the findings do not support it: no significant difference, no association, not effective, or the opposite effect.
   - "maybe": the findings are genuinely mixed or conditional (it depends on a subgroup or setting) or the authors say it is unclear. A secondary caveat or a "further research is needed" remark does NOT make the answer "maybe".
3. Copy one or two sentences WORD FOR WORD from that abstract as evidence_quote.
Return ONE JSON object and nothing else:
{"status": "answered" | "insufficient_evidence",
 "citation_pmid": "<PMID of the chosen abstract, copied exactly>" | null,
 "verdict": "yes" | "no" | "maybe" | null,
 "explanation": "2-4 sentences grounded in the chosen abstract" | null,
 "evidence_quote": "verbatim text from the chosen abstract" | null,
 "confidence": "high" | "medium" | "low" | null}
Use "insufficient_evidence" (other fields null) only if NO abstract studies the question. Never invent a PMID."""


def _docs_msg(question, docs):
    ctx = "\n\n".join(f"[PMID {p}]\n{t}" for p, t in docs)
    return f"Retrieved abstracts:\n{ctx}\n\nQuestion: {question}"


def executor_messages(question, docs):
    msgs = [{"role": "system", "content": EXECUTOR}]
    for pmid, q, ab, out in prompts.dev_demos("open"):
        o = {"status": "answered", "citation_pmid": pmid, "verdict": out["verdict"],
             "explanation": out["explanation"], "evidence_quote": out["evidence_quote"], "confidence": "medium"}
        msgs += [{"role": "user", "content": _docs_msg(q, [(pmid, ab)])},
                 {"role": "assistant", "content": json.dumps(o)}]
    return msgs + [{"role": "user", "content": _docs_msg(question, docs)}]


def revise_messages(base, draft, feedback):
    return base + [
        {"role": "assistant", "content": json.dumps(draft)},
        {"role": "user", "content": "A reviewer rejected this draft:\n" + "\n".join(f"- {f}" for f in feedback) +
         "\nFix the problems and return the corrected JSON object only."}]


# ---- critic -----------------------------------------------------------------------------------
CRITIC = """You are the CRITIC of MedEvidence Copilot. You check a draft answer against the abstract it cites.
Return ONE JSON object:
{"abstract_matches_question": true | false,
 "verdict_supported": true | false,
 "correct_verdict": "yes" | "no" | "maybe",
 "explanation_faithful": true | false,
 "feedback": "one sentence: what is wrong, or \\"ok\\""}
- abstract_matches_question: the abstract studies the same population, intervention or exposure, and outcome as the question.
- correct_verdict: what THIS study's results say about the question. "no" = no significant difference/association, not effective, or the opposite; "maybe" = genuinely mixed or conditional results, or the authors say it is unclear. A side caveat does not make it "maybe".
- verdict_supported: the draft verdict equals correct_verdict.
- explanation_faithful: the explanation states nothing that the abstract does not say.
Approve (all true) unless there is a clear error."""


def critic_messages(question, pmid, abstract, conclusion, draft):
    u = (f"Question: {question}\n\n[PMID {pmid}]\n{abstract}\n\nLast sentences of the abstract (results):\n"
         + "\n".join(conclusion) +
         f"\n\nDraft verdict: {draft.get('verdict')}\nDraft explanation: {draft.get('explanation')}")
    return [{"role": "system", "content": CRITIC}, {"role": "user", "content": u}]


JSON_REPAIR = "That was not a valid JSON object. Return ONLY the JSON object with the required keys, nothing else."
