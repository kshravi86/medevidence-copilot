"""Writes docs/agent_system_prompts.txt: every prompt the Week 3 agent sends to the model, read from the code.

Run from the repo root:  python scripts/export_agent_prompts.py
Re-run it after changing a prompt so the text file stays in step with the code.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medevidence import prompts  # noqa: E402
from medevidence.agent import prompts_agent as P, safety  # noqa: E402

OUT = ROOT / "docs" / "agent_system_prompts.txt"
RULE = "=" * 100


def section(title, body):
    return f"{RULE}\n{title}\n{RULE}\n{body.strip(chr(10))}\n\n"


shots = "\n".join(f'  user:      {q}\n  assistant: {{"category": "{c}", "reason": "{why}"}}\n'
                  for q, c, why in safety.CLASSIFIER_SHOTS)
refusals = "\n".join(f"  {cat}:\n    {msg}\n" for cat, msg in prompts.REFUSAL_MESSAGES.items())

text = f"""MedEvidence Copilot: system prompts of every agent (Week 3)
Generated from the code by scripts/export_agent_prompts.py. Do not edit by hand.

Model calls per question, in order:
  1. Safety classifier   (skipped when the code rules already found an emergency)
  2. Planner             (and once more as a re-plan if no retrieved abstract addresses the question)
  3. Executor            (and again for each critic send-back, up to 2)
  4. Critic              (skipped when a code check fails; the draft goes straight back)
Every call asks for JSON (temperature 0). If the reply is not valid JSON, the JSON repair message is sent once.
Search, reading abstracts, quote checks and the refusal text are code, not model calls.

""" + section("1. SAFETY CLASSIFIER  (src/medevidence/agent/safety.py: CLASSIFIER)", safety.CLASSIFIER) + section(
    "1a. SAFETY CLASSIFIER: few-shot examples sent before the question (CLASSIFIER_SHOTS)", shots) + section(
    "1b. SAFETY CLASSIFIER: follow-up wrapping (classifier_messages)",
    "When the session has earlier in-scope questions, the latest message is sent as:\n\n"
    "Earlier research questions in this conversation:\n- <up to 2 earlier questions>\n\n"
    "Classify this latest message (a follow-up to them): <message>") + section(
    "2. PLANNER  (src/medevidence/agent/prompts_agent.py: PLANNER)", P.PLANNER) + section(
    "2a. RE-PLAN: added after the planner messages when nothing relevant was found (REPLAN)", P.REPLAN) + section(
    "3. EXECUTOR  (src/medevidence/agent/prompts_agent.py: EXECUTOR)",
    P.EXECUTOR + "\n\nFew-shot: 3 worked examples (one yes, one no, one maybe) from the PubMedQA labelled set,\n"
    "outside the test split that the evaluation questions come from, are sent before the real question (prompts.dev_demos).\n"
    "User message format:\n\nRetrieved abstracts:\n[PMID n]\n<abstract>\n\n[PMID m]\n<abstract>\n\nQuestion: <research question>") + section(
    "3a. EXECUTOR REVISION: sent after a critic send-back (revise_messages)",
    "The previous draft is replayed as the assistant turn, then:\n\n"
    "A reviewer rejected this draft:\n- <feedback line>\n- <feedback line>\n"
    "Fix the problems and return the corrected JSON object only.") + section(
    "4. CRITIC  (src/medevidence/agent/prompts_agent.py: CRITIC)",
    P.CRITIC + "\n\nUser message format (critic_messages):\n\nQuestion: <research question>\n\n[PMID n]\n<abstract>\n\n"
    "Last sentences of the abstract (results):\n<2 sentences from conclusion_sentences>\n\n"
    "Draft verdict: <verdict>\nDraft explanation: <explanation>") + section(
    "5. JSON REPAIR: sent once after any reply that is not valid JSON (JSON_REPAIR)", P.JSON_REPAIR) + section(
    "6. FIXED REFUSAL MESSAGES: chosen by code, never written by the model (prompts.REFUSAL_MESSAGES)", refusals)

OUT.write_text(text)
print("wrote", OUT.relative_to(ROOT), f"({len(text.splitlines())} lines)")
