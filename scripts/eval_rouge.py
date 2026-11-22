"""ROUGE-L of the generated explanation against the authors' conclusion, for every system run so far.

Reads the per-item results of Weeks 1-3 (no model calls needed: each row stores the prediction) and scores
the 78 answerable items. Two averages are reported:
  - rougeL_all:      over all 78 items, counting a missing explanation (refused / insufficient / error) as 0
  - rougeL_answered: over items that produced an explanation
plus by slice, and for items whose verdict is right vs wrong (does a closer wording go with a right verdict?).
Results: results/rouge_l.json.  Usage: python scripts/eval_rouge.py
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from medevidence import data  # noqa: E402
from medevidence.rouge import rouge_l  # noqa: E402

RUNS = {f"week1/{p.stem}": p for p in sorted((ROOT / "results/week1").glob("*.jsonl"))}
RUNS.update({f"week2/{p.stem}": p for p in sorted((ROOT / "results/week2").glob("*.jsonl"))})
RUNS.update({f"week3/{p.stem}": p for p in sorted((ROOT / "results/week3").glob("*.jsonl"))})


def mean(x):
    return round(statistics.mean(x), 4) if x else None


def score_run(path, ev):
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    per = []
    for it in ev:
        if it["expected_behavior"] != "answer" or it["id"] not in rows:
            continue
        r = rows[it["id"]]
        pred = r.get("pred") or {}
        expl = pred.get("explanation") if isinstance(pred, dict) and pred.get("status") == "answered" else None
        s = rouge_l(expl, it["gold_rationale"])
        per.append({"id": it["id"], "slice": it["query_type"], "has_expl": bool(expl),
                    "verdict_ok": bool(r.get("c1_verdict")), **{f"rougeL_{k}": round(v, 4) for k, v in s.items()}})
    out = {
        "n": len(per),
        "rougeL_all": mean([p["rougeL_f"] for p in per]),
        "rougeL_answered": mean([p["rougeL_f"] for p in per if p["has_expl"]]),
        "precision_answered": mean([p["rougeL_p"] for p in per if p["has_expl"]]),
        "recall_answered": mean([p["rougeL_r"] for p in per if p["has_expl"]]),
        "n_answered": sum(p["has_expl"] for p in per),
        "verdict_right": mean([p["rougeL_f"] for p in per if p["has_expl"] and p["verdict_ok"]]),
        "verdict_wrong": mean([p["rougeL_f"] for p in per if p["has_expl"] and not p["verdict_ok"]]),
    }
    for sl in ("verbatim", "paraphrase", "boundary"):
        out[f"rougeL_{sl}"] = mean([p["rougeL_f"] for p in per if p["slice"] == sl])
    return out, per


def main():
    ev = data.load_eval()
    summary, items = {}, {}
    for name, path in RUNS.items():
        summary[name], items[name] = score_run(path, ev)
    json.dump({"summary": summary, "items": items}, open(ROOT / "results" / "rouge_l.json", "w"), indent=1)
    print(f"{'run':38s} {'all78':>6s} {'answ':>6s} {'n':>3s} {'P':>6s} {'R':>6s} {'right':>6s} {'wrong':>6s}")
    for name, s in summary.items():
        print(f"{name:38s} {s['rougeL_all']:6.3f} {s['rougeL_answered'] or 0:6.3f} {s['n_answered']:3d} "
              f"{s['precision_answered'] or 0:6.3f} {s['recall_answered'] or 0:6.3f} "
              f"{s['verdict_right'] or 0:6.3f} {s['verdict_wrong'] or 0:6.3f}")


if __name__ == "__main__":
    main()
