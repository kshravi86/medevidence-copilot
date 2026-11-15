"""Week 2 analysis: failure taxonomy per RAG config, refusal breakdown, paired bootstrap incl. vs Week 1 baseline (b)."""
import collections, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from medevidence import data, metrics

ev = {i["id"]: i for i in data.load_eval()}
ans = [i for i, x in ev.items() if x["expected_behavior"] == "answer"]
W2 = ROOT / "results" / "week2"
names = ["bm25_1", "bm25_3", "hybrid_3", "rerank_3"]
R = {n: {r["id"]: r for r in map(json.loads, open(W2 / f"rag__abstract__{n}.jsonl"))} for n in names}
W1 = {r["id"]: r for r in map(json.loads, open(ROOT / "results/week1/bm25__fewshot.jsonl"))}
out = {"taxonomy": {}, "refusal": {}, "bootstrap": {}}
for n in names:
    t = collections.Counter(); hit = [i for i in ans if R[n][i]["gold_in_context"]]
    for i in ans:
        r = R[n][i]
        if r.get("strict"): t["pass"] += 1
        elif not r["c6_no_overrefusal"]: t["over-refusal"] += 1
        elif not r["gold_in_context"]: t["retrieval miss (gold not in context)"] += 1
        elif r["status"] == "insufficient_evidence": t["insufficient_evidence despite gold in context"] += 1
        elif not r["c2_citation"]: t["gold in context but cited another abstract"] += 1
        elif not r["c5_format"]: t["format error"] += 1
        elif not r["c1_verdict"]: t["wrong verdict (gold cited)"] += 1
        elif not r["c3_grounded"]: t["quote not verbatim"] += 1
        elif not r.get("c4_faithful"): t["unfaithful explanation (judge)"] += 1
        else: t["other"] += 1
    t = dict(t)
    t["_verdict_acc_when_gold_in_context"] = round(sum(R[n][i]["c1_verdict"] for i in hit) / len(hit), 3)
    t["_n_gold_in_context"] = len(hit)
    t["_cited_not_in_retrieved"] = sum(1 for i in ans if R[n][i]["cited_pmid"] and R[n][i]["cited_pmid"] not in R[n][i]["retrieved"])
    out["taxonomy"][n] = t
    rf = collections.Counter()
    for i, it in ev.items():
        if it["expected_behavior"] != "refuse": continue
        r = R[n][i]; o = r["pred"] or {}
        if r["correct"]: rf["correct"] += 1
        elif not r["refused"]: rf[f"not refused ({it['refusal_category']}: {it['notes']})"] += 1
        elif o.get("refusal_category") != it["refusal_category"]: rf[f"wrong category {it['refusal_category']}->{o.get('refusal_category')}"] += 1
        else: rf["message lacks redirect"] += 1
    out["refusal"][n] = dict(rf)

def vec(R_, key): return [int(bool(R_[i].get(key))) for i in ans]
pairs = [("rerank_3", "bm25_1"), ("hybrid_3", "bm25_1"), ("bm25_3", "bm25_1"), ("rerank_3", "hybrid_3")]
for a, b in pairs:
    for key in ("strict", "c1_verdict", "c2_citation"):
        d, ci = metrics.paired_bootstrap(vec(R[a], key), vec(R[b], key))
        out["bootstrap"][f"{key}: {a} - {b}"] = {"diff": round(d, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)]}
for a in names:
    for key in ("strict", "c1_verdict"):
        d, ci = metrics.paired_bootstrap(vec(R[a], key), vec(W1, key))
        out["bootstrap"][f"{key}: week2 {a} - week1 baseline(b) few-shot"] = {"diff": round(d, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)]}
json.dump(out, open(W2 / "analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
