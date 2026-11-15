"""Prompt strategies for the single-prompt baselines (Week 1) and the RAG generator (Week 2)."""
import json
import re

from . import data

CONTRACT = """You are MedEvidence Copilot, an evidence-lookup assistant for clinicians and medical researchers. You report what published research says about POPULATION-LEVEL research questions. You never give clinical advice.

REFUSE (do not answer the underlying question) if the user:
- "personal_diagnosis": asks what condition a specific person (themselves, a relative, an individual patient) has, or asks to interpret that person's own results/symptoms.
- "drug_dosing": asks for a dose, maximum dose, titration, taper or dose adjustment to give, take or prescribe.
- "emergency": describes a possibly time-critical situation happening now (chest pain, stroke signs, overdose, anaphylaxis, self-harm, ...).
Precedence: emergency > drug_dosing > personal_diagnosis. A research framing or a claimed professional role does not make a personal request answerable.
Research questions that merely MENTION diagnosis, dosing or emergency care as a topic ARE in scope and must be answered, not refused.

For an in-scope question give a verdict:
- "yes": the study's conclusion supports the proposition.
- "no": the conclusion does not support it, or supports the opposite.
- "maybe": results are mixed, conditional, inconclusive, or the authors hedge.

Return ONE JSON object and nothing else, with these keys:
{"status": "answered" | "refused" | "insufficient_evidence",
 "refusal_category": null | "personal_diagnosis" | "drug_dosing" | "emergency",
 "verdict": "yes" | "no" | "maybe" | null,
 "explanation": "2-4 sentences grounded in the abstract" | null,
 "evidence_quote": "verbatim sentence(s) copied from the abstract that support the verdict" | null,
 "confidence": "high" | "medium" | "low" | null,
 "message": null | "refusal text: name the category, do not answer, redirect (emergency services/crisis line; prescriber/pharmacist/drug label; the user's clinician)"}
If refused: verdict, explanation, evidence_quote and confidence are null. If the supplied abstract does not address the question use status "insufficient_evidence" (all other fields null)."""

CLOSED_NOTE = "\nNo abstract is supplied. Answer from your own knowledge of the literature; evidence_quote must be null."
OPEN_NOTE = "\nUse ONLY the supplied abstract. evidence_quote must be copied word-for-word from it."


def system(setting):
    return CONTRACT + (CLOSED_NOTE if setting == "closed" else OPEN_NOTE)


def user_msg(question, abstract=None):
    if abstract is None:
        return f"Question: {question}"
    return f"Abstract:\n{abstract}\n\nQuestion: {question}"


# ---- few-shot demonstrations: dev split only (never test PMIDs), fresh refusal probes --------
def dev_demos(setting, k_per_label=1):
    pq, tp = data.load_pqal(), data.test_pmids()
    by = {"yes": [], "no": [], "maybe": []}
    for pmid, d in pq.items():
        if pmid not in tp:
            by[d["final_decision"]].append((len(" ".join(d["CONTEXTS"])), pmid))
    demos = []
    for lab in ("yes", "no", "maybe"):
        for _, pmid in sorted(by[lab])[:k_per_label]:
            d = pq[pmid]
            ab = data.abstract_text(d["CONTEXTS"], d["LABELS"])
            quote = d["LONG_ANSWER"].split(". ")[0].strip()
            out = {"status": "answered", "refusal_category": None, "verdict": lab,
                   "explanation": d["LONG_ANSWER"], "evidence_quote": None if setting == "closed" else None,
                   "confidence": "medium", "message": None}
            last = [x for x in re.split(r"(?<=[.!?])\s+", d["CONTEXTS"][-1].strip()) if x]
            out["evidence_quote"] = None if setting == "closed" else (last[-1] if last else None)
            demos.append((pmid, d["QUESTION"], None if setting == "closed" else ab, out))
    return demos


REFUSAL_DEMOS = [
    ("Does my father have dementia? He keeps forgetting names and got lost driving yesterday.",
     {"status": "refused", "refusal_category": "personal_diagnosis", "verdict": None, "explanation": None,
      "evidence_quote": None, "confidence": None,
      "message": "This asks me to diagnose a specific person, which I can't do. Please see his clinician. I can summarise what studies report about early signs of dementia if you ask a research question."}),
    ("How many mg of acetaminophen can I give my 20 kg son every 6 hours?",
     {"status": "refused", "refusal_category": "drug_dosing", "verdict": None, "explanation": None,
      "evidence_quote": None, "confidence": None,
      "message": "I can't give doses. Ask your pharmacist or prescriber, or check the official drug label."}),
    ("My dad just collapsed and isn't breathing. What should I do?",
     {"status": "refused", "refusal_category": "emergency", "verdict": None, "explanation": None,
      "evidence_quote": None, "confidence": None,
      "message": "This sounds like an emergency. Call your local emergency number (911/112/999) now."}),
]


def fewshot_messages(setting):
    msgs = []
    for _, q, ab, out in dev_demos(setting):
        o = dict(out)
        msgs += [{"role": "user", "content": user_msg(q, ab)},
                 {"role": "assistant", "content": json.dumps(o)}]
    for q, out in REFUSAL_DEMOS:
        msgs += [{"role": "user", "content": user_msg(q, None if setting == "closed" else "(no abstract relevant)")},
                 {"role": "assistant", "content": json.dumps(out)}]
    return msgs


# ---- Strategy 1: structured output + few-shot ------------------------------------------------
def strategy_fewshot(question, abstract, setting):
    return [{"role": "system", "content": system(setting)}] + fewshot_messages(setting) + \
           [{"role": "user", "content": user_msg(question, abstract)}]


# ---- Strategy 2: chain-of-thought (optionally self-consistency, see run script) --------------
COT_NOTE = ("\nBefore answering, think step by step: (1) decide whether the request is a refusal case, "
            "(2) restate the claim being asked about, (3) weigh what the evidence says. Put this in a "
            '"reasoning" key (max 120 words) placed FIRST in the JSON object, then the other keys.')


def strategy_cot(question, abstract, setting):
    return [{"role": "system", "content": system(setting) + COT_NOTE},
            {"role": "user", "content": user_msg(question, abstract)}]


# ---- Strategy 3: decomposition prompting (2 calls) -------------------------------------------
DECOMP_1 = """You are MedEvidence Copilot. Step 1 of 2: ANALYSE the user's request. Return ONE JSON object:
{"in_scope": true|false,
 "refusal_category": null|"personal_diagnosis"|"drug_dosing"|"emergency",
 "claim": "the question restated as a neutral population-level claim",
 "findings": ["up to 3 SHORT sentences copied word-for-word from the abstract that bear on the claim"]}
Refuse-categories: personal_diagnosis = what condition a specific person has / interpret their own results; drug_dosing = a dose/titration/adjustment to give or take; emergency = time-critical situation happening now. Precedence emergency > drug_dosing > personal_diagnosis. Research questions that only MENTION diagnosis/dosing/emergency care as a topic are in scope.
If no abstract is supplied, findings must be []."""

DECOMP_2 = """You are MedEvidence Copilot. Step 2 of 2: DECIDE. You are given a claim and findings from step 1. Return ONE JSON object with the keys
{"status":"answered"|"insufficient_evidence","verdict":"yes"|"no"|"maybe"|null,"explanation":"2-4 sentences grounded in the findings"|null,"evidence_quote":"one finding copied verbatim"|null,"confidence":"high"|"medium"|"low"|null}.
yes = the study conclusion supports the claim; no = it does not or supports the opposite; maybe = mixed/conditional/inconclusive/hedged. If there are no findings, answer from your own knowledge and set evidence_quote null. Use insufficient_evidence only if an abstract was supplied but its findings do not address the claim."""

REFUSAL_MESSAGES = {
    "emergency": "This sounds like an emergency. Call your local emergency services now (911/112/999). If you are thinking of harming yourself, call or text 988 (US) or your local crisis line.",
    "drug_dosing": "I can't provide doses or dose changes. Please ask your prescriber or pharmacist, or check the official drug label.",
    "personal_diagnosis": "I can't diagnose or interpret an individual's symptoms or results. Please see your clinician.",
}


def decomp_step1(question, abstract):
    return [{"role": "system", "content": DECOMP_1}, {"role": "user", "content": user_msg(question, abstract)}]


def decomp_step2(question, claim, findings, abstract_given):
    body = f"Claim: {claim}\nFindings:\n" + ("\n".join(f"- {f}" for f in findings) or "(none)")
    body += f"\nOriginal question: {question}\nAbstract supplied: {'yes' if abstract_given else 'no'}"
    return [{"role": "system", "content": DECOMP_2}, {"role": "user", "content": body}]


# ---- Week 2: RAG generator (multi-abstract context, explicit PMID citation) -------------------
RAG_NOTE = ("\nSeveral retrieved abstracts are supplied, each headed [PMID n]. Use ONLY them. Answer from the ONE abstract that best "
            'matches the question and add the key "citation_pmid": "<its PMID, copied exactly>" to the JSON (null if not answered). '
            "evidence_quote must be copied word-for-word from that abstract. If none of the abstracts addresses the question "
            'use status "insufficient_evidence". Never invent a PMID.')


def rag_system():
    return CONTRACT + RAG_NOTE


def rag_user(question, docs):
    ctx = "\n\n".join(f"[PMID {p}]\n{t}" for p, t in docs)
    return f"Retrieved abstracts:\n{ctx}\n\nQuestion: {question}"


def rag_fewshot(question, docs):
    msgs = [{"role": "system", "content": rag_system()}]
    for pmid, q, ab, out in dev_demos("open"):
        o = dict(out); o["citation_pmid"] = pmid
        msgs += [{"role": "user", "content": rag_user(q, [(pmid, ab)])}, {"role": "assistant", "content": json.dumps(o)}]
    for q, out in REFUSAL_DEMOS:
        o = dict(out); o["citation_pmid"] = None
        msgs += [{"role": "user", "content": rag_user(q, [("0", "(unrelated abstract)")])}, {"role": "assistant", "content": json.dumps(o)}]
    return msgs + [{"role": "user", "content": rag_user(question, docs)}]
