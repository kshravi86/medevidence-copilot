"""C4 faithfulness judge (spec 5): does the explanation agree with the authors' conclusion
and add no claims the abstract does not support? Judge model = the same Qwen3-8B, temperature 0
(known limitation, see report; a 20% human spot-check file is exported alongside)."""
import json

from .metrics import parse_json

SYS = """You are a strict evaluator. You get a research question, an abstract, the authors' conclusion, and a candidate EXPLANATION written by an assistant.
Return ONE JSON object: {"agrees": true|false, "unsupported": true|false}
- agrees: true only if the explanation's overall message is consistent with the authors' conclusion (same direction: supports / does not support / mixed).
- unsupported: true if the explanation asserts a specific fact, number or claim that is NOT stated in the abstract or the conclusion."""


def judge(llm, item, abstract, explanation):
    if not explanation:
        return {"agrees": False, "unsupported": True, "raw": None}
    u = (f"Question: {item['original_question'] or item['query']}\n\nAbstract:\n{abstract}\n\n"
         f"Authors' conclusion: {item['gold_rationale']}\n\nEXPLANATION: {explanation}")
    res = llm.chat([{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                   temperature=0.0, max_tokens=60, json_mode=True)
    o, _ = parse_json(res.text)
    if not o:
        return {"agrees": False, "unsupported": True, "raw": res.text}
    return {"agrees": bool(o.get("agrees")), "unsupported": bool(o.get("unsupported")), "raw": res.text}


def faithful(j):
    return j["agrees"] and not j["unsupported"]
