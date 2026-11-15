"""Week 1 analysis: confusion matrices, failure taxonomy, paired bootstrap, judge spot-check sample."""
import csv, collections, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from medevidence import data, metrics

ev = {i["id"]: i for i in data.load_eval()}
ans_ids = [i for i, x in ev.items() if x["expected_behavior"] == "answer"]
absmap = {d["pmid"]: d["text"] for d in data.load_corpus()}
W = ROOT / "results" / "week1"
names = ["closed__fewshot", "closed__cot", "closed__cot_sc", "closed__decompose",
         "bm25__fewshot", "bm25__cot", "bm25__cot_sc", "bm25__decompose"]
R = {n: {r["id"]: r for r in map(json.loads, open(W / f"{n}.jsonl"))} for n in names}
out = {"confusion": {}, "taxonomy": {}, "bootstrap": {}}

for n in names:
    cm = collections.Counter((ev[i]["gold_label"], (R[n][i]["verdict"] if R[n][i]["status"] == "answered" else "none")) for i in ans_ids)
    out["confusion"][n] = {f"{g}->{p}": c for (g, p), c in sorted(cm.items())}

# failure taxonomy for the BM25 settings (answerable items)
for n in [x for x in names if x.startswith("bm25")]:
    t = collections.Counter()
    for i in ans_ids:
        r = R[n][i]
        if r.get("strict"):
            t["pass"] += 1; continue
        if not r["c6_no_overrefusal"]: t["over-refusal"] += 1
        elif not r["c2_citation"]: t["retrieval miss (top-1 not gold)"] += 1
        elif r["status"] == "insufficient_evidence": t["said insufficient_evidence despite gold abstract"] += 1
        elif not r["c5_format"]: t["format error"] += 1
        elif not r["c1_verdict"]: t["wrong verdict (gold abstract retrieved)"] += 1
        elif not r["c3_grounded"]: t["quote not verbatim in abstract"] += 1
        elif not r.get("c4_faithful"): t["unfaithful explanation (judge)"] += 1
        else: t["other"] += 1
    out["taxonomy"][n] = dict(t)
    hit = [i for i in ans_ids if R[n][i]["c2_citation"]]; miss = [i for i in ans_ids if not R[n][i]["c2_citation"]]
    out["taxonomy"][n]["_verdict_acc_when_top1_hit"] = round(sum(R[n][i]["c1_verdict"] for i in hit) / max(len(hit), 1), 3)
    out["taxonomy"][n]["_verdict_acc_when_top1_miss"] = round(sum(R[n][i]["c1_verdict"] for i in miss) / max(len(miss), 1), 3)
    out["taxonomy"][n]["_n_hit_miss"] = [len(hit), len(miss)]

# paired bootstrap on C1 (verdict correct) over 78 answerable items
def c1(n): return [int(R[n][i]["c1_verdict"]) for i in ans_ids]
pairs = [("bm25__fewshot", "closed__fewshot"), ("bm25__fewshot", "bm25__cot"), ("bm25__fewshot", "bm25__decompose"),
         ("bm25__decompose", "closed__decompose"), ("closed__decompose", "closed__fewshot")]
for a, b in pairs:
    d, ci = metrics.paired_bootstrap(c1(a), c1(b))
    out["bootstrap"][f"{a} - {b}"] = {"diff": round(d, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)]}
# strict accuracy among bm25 strategies
def st(n): return [int(bool(R[n][i]["strict"])) for i in ans_ids]
for a, b in [("bm25__fewshot", "bm25__cot"), ("bm25__fewshot", "bm25__decompose")]:
    d, ci = metrics.paired_bootstrap(st(a), st(b))
    out["bootstrap"][f"STRICT {a} - {b}"] = {"diff": round(d, 3), "ci95": [round(ci[0], 3), round(ci[1], 3)]}

json.dump(out, open(W / "analysis.json", "w"), indent=1)

# 20% human spot-check of the C4 judge (best BM25 config), fixed seed
rng = random.Random(42)
cand = [i for i in ans_ids if R["bm25__fewshot"][i]["status"] == "answered"]
pick = sorted(rng.sample(cand, round(0.2 * len(ans_ids))))
with open(W / "judge_spotcheck.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "question", "gold_label", "gold_rationale", "model_explanation", "cited_pmid", "judge_faithful", "human_faithful (y/n)", "human_notes"])
    for i in pick:
        r = R["bm25__fewshot"][i]
        w.writerow([i, ev[i]["query"], ev[i]["gold_label"], ev[i]["gold_rationale"], (r["pred"] or {}).get("explanation"), r["cited_pmid"], r.get("c4_faithful"), "", ""])
print(json.dumps(out, indent=1))
print("spot-check rows:", len(pick))
