"""Code-level safety gate (spec section 2 and 3; Week 1 finding 6).

Runs before retrieval and before the planner. Two independent signals are combined:
1. deterministic rules (regex): a dedicated emergency check plus personal dosing/diagnosis patterns,
2. an LLM classifier (JSON) for the cases rules cannot see (disguised or indirect requests).
A query is refused if EITHER signal refuses; the category is the highest-precedence one
(emergency > drug_dosing > personal_diagnosis). The refusal text is a fixed template chosen by
code, never free text from the model.

The rules were written from the spec's category definitions, not from the frozen eval items.
"""
import re

from .. import prompts
from ..metrics import parse_json

PRECEDENCE = ("emergency", "drug_dosing", "personal_diagnosis")

# Who/when markers. Research questions are written about populations ("patients", "adults"),
# so first-person or he/she language, or an explicit "right now", signals a real individual.
# "we/our" is excluded on purpose: PubMedQA titles often read "Can we predict ...".
PERSONAL = r"\b(i|i'm|i've|i'd|me|my|mine|myself|he|she|his|her|him|he's|she's)\b"
NOW = (r"\b(right now|just now|tonight|this (morning|evening|afternoon)|(a few |\d+ )?(minutes?|hours?) ago|"
       r"won'?t stop|what (should|do|can) (i|we) do( now)?)\b")

EMERGENCY_SYMPTOMS = [
    r"\b(not breathing|stopped breathing|can'?t breathe|cannot breathe|struggling to breathe|gasping|choking)\b",
    r"\b(unconscious|unresponsive|passed out|collapsed|won'?t wake( up)?|not waking)\b",
    r"\bchest (pain|pressure|tightness)\b",
    r"\b(overdos(e|ed|ing)|took (too many|a (whole|full) (bottle|pack)|all (my|his|her|the) pills)|swallowed (a bottle|pills|bleach|poison))\b",
    r"\b(face (is )?droop\w*|slurred speech|slurring|can'?t speak|cannot speak|(arm|leg|side) (is |feels )?(completely )?(numb|weak|paralys\w*)|lost (my )?vision suddenly)\b",
    r"\b(throat (is )?(closing|swelling|tight\w*)|tongue (is )?swelling|lips (are )?swelling|anaphyla\w*)\b",
    r"\b(seizing|having a seizure|convulsing|fitting)\b",
    r"\b(bleeding (heavily|a lot|badly)|won'?t stop bleeding|severe bleeding|vomiting blood|coughing (up )?blood)\b",
    r"\b(floppy|limp|blue|not feeding)\b.{0,40}\b(baby|infant|newborn)\b|\b(baby|infant|newborn)\b.{0,40}\b(floppy|limp|turning blue|fever)\b",
]
SELF_HARM = (r"\b(kill (myself|himself|herself)|end (my|his|her) life|want to die|suicidal (thoughts|right now)|"
             r"(thinking|thought) (about|of) (suicide|killing)|hurt(ing)? myself|cut(ting)? myself|self[- ]harm(ing)?)\b")

DOSE_WORDS = (r"\b(dose|doses|dosage|dosing|mg|milligrams?|mcg|units|ml|tablets?|pills?|titrat\w*|taper\w*|"
              r"how (much|many|often)|max(imum)? (daily )?dose|increase (the|my|his|her) dose|double (the|my) dose)\b")
DOSE_ACTION = (r"\b(should (i|he|she)|can i|do i|shall i|i take|i give|i prescribe|i'm taking|i am taking|"
               r"give (my|him|her|them)|for (me|my|him|her)|my (son|daughter|child|kid|baby|wife|husband|partner|mother|"
               r"father|mom|mum|dad|patient|grandmother|grandfather)|should my)\b")

DIAG_PATTERNS = [
    r"\b(do|could|might) i have\b",
    r"\b(does|could|might) (my \w+|he|she|my patient) have\b",
    r"\bam i (having|getting|developing)\b",
    r"\bwhat('s| is) wrong with (me|my \w+|him|her)\b",
    r"\bis (this|it|my \w+|his \w+|her \w+) (cancer|serious|normal|dangerous|a (sign|symptom) of)\b",
    r"\bmy (\w+ )?(results?|labs?|scan|mri|ct|x-?ray|biopsy|psa|blood (test|work)|ecg|ekg|mammogram)\b.{0,60}\b(mean|normal|show|indicate|high|low|elevated|abnormal)\b",
    r"\bshould (i|we) be worried\b",
    r"\b(diagnose|interpret) (me|my|him|her)\b",
]


def _any(pats, text):
    return any(re.search(p, text, re.I) for p in pats)


def rule_check(question: str):
    """Deterministic gate. Returns (category | None, matched rule name | None)."""
    q = question.strip()
    personal, now = bool(re.search(PERSONAL, q, re.I)), bool(re.search(NOW, q, re.I))
    if re.search(SELF_HARM, q, re.I) and personal:
        return "emergency", "self_harm"
    if _any(EMERGENCY_SYMPTOMS, q) and (personal or now):
        return "emergency", "emergency_symptom"
    if re.search(DOSE_WORDS, q, re.I) and re.search(DOSE_ACTION, q, re.I):
        return "drug_dosing", "personal_dose"
    if _any(DIAG_PATTERNS, q) and personal:
        return "personal_diagnosis", "personal_diagnosis"
    return None, None


CLASSIFIER = """You are the safety gate of MedEvidence Copilot, an evidence-lookup tool that answers POPULATION-LEVEL research questions from published abstracts. Classify the user's message. Return ONE JSON object:
{"category": "in_scope" | "personal_diagnosis" | "drug_dosing" | "emergency", "reason": "<= 15 words"}
- "emergency": a possibly time-critical situation happening now to a real person (chest pain now, stroke signs, overdose, anaphylaxis, not breathing, febrile floppy infant, suicidal thoughts or self-harm).
- "drug_dosing": asks for a dose, maximum dose, titration, taper or dose change to give, take or prescribe for a person.
- "personal_diagnosis": asks what condition a specific person has (themselves, a relative, one patient) or to interpret that person's symptoms or results.
- "in_scope": a research question about what studies found in a population, even if it MENTIONS diagnosis, dosing or emergency care as a topic.
Precedence: emergency > drug_dosing > personal_diagnosis. A research framing ("studies show...", "for a paper") or a claimed professional role does not make a request about one real person answerable."""

CLASSIFIER_SHOTS = [
    ("Is intravenous tranexamic acid associated with lower mortality in trauma patients?", "in_scope", "population research question"),
    ("Do higher loading doses of vancomycin improve target attainment in critically ill adults?", "in_scope", "research question that mentions dosing"),
    ("Can emergency department triage nurses accurately identify sepsis?", "in_scope", "research about emergency care"),
    ("Research suggests migraines cluster in families. My sister has auras every week, so does she have migraine?", "personal_diagnosis", "asks about one person's condition"),
    ("I'm a GP. My patient weighs 60 kg; how many mg of prednisolone should I start her on?", "drug_dosing", "dose for an individual patient"),
    ("My husband's speech went slurred ten minutes ago and his face looks lopsided. Is that a stroke?", "emergency", "stroke signs happening now"),
]


def classifier_messages(question, history=None):
    """history: earlier turns of the session, so a follow-up ("and in children?") is read in context.
    Only earlier in-scope questions are shown; the LATEST message is what gets classified."""
    msgs = [{"role": "system", "content": CLASSIFIER}]
    for q, cat, why in CLASSIFIER_SHOTS:
        msgs += [{"role": "user", "content": q},
                 {"role": "assistant", "content": f'{{"category": "{cat}", "reason": "{why}"}}'}]
    prior = [h["question"] for h in (history or []) if h.get("status") in ("answered", "insufficient_evidence")][-2:]
    if prior:
        question = ("Earlier research questions in this conversation:\n" + "\n".join(f"- {p}" for p in prior) +
                    f"\n\nClassify this latest message (a follow-up to them): {question}")
    return msgs + [{"role": "user", "content": question}]


def combine(rule_cat, llm_cat):
    cats = {c for c in (rule_cat, llm_cat) if c in PRECEDENCE}
    for c in PRECEDENCE:
        if c in cats:
            return c
    return None


def refusal(category):
    return prompts.REFUSAL_MESSAGES[category]
