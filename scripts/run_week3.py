"""Week 3: end-to-end evaluation of the agent (safety gate -> planner -> executor -> critic) on the frozen set.

Reports the spec 6.2 metrics (same scorer as Weeks 1-2), plus the brief's observability metrics:
task completion rate, average tool/LLM calls per task, tokens, estimated cost and latency per query,
and a failure taxonomy (retrieval miss, reasoning error, tool-call error, hallucination).
The agent is compared with both baselines using a paired bootstrap (10k resamples):
  - baseline (b) from the spec: Week 1 structured few-shot + BM25 top-1 (results/week1/bm25__fewshot.jsonl)
  - the stronger Week 2 single prompt, BM25 top-1 (results/week2/rag__abstract__bm25_1.jsonl)

Usage: python scripts/run_week3.py [--name agent] [--no-critic] [--limit N]
Cost is an estimate: the course endpoint is free, so --price-in/--price-out (USD per 1M tokens) are assumptions.
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm  # noqa: E402

from medevidence import data, judge as J, metrics  # noqa: E402

OUT = ROOT / "results" / "week3"
BASELINES = {"week1_bm25_fewshot (spec baseline b)": ROOT / "results/week1/bm25__fewshot.jsonl",
             "week2_bm25_top1 (stronger single prompt)": ROOT / "results/week2/rag__abstract__bm25_1.jsonl"}


def pred_obj(r):
    """Map the agent output onto the contract the scorer expects."""
    if r["status"] == "error":
        return None, r.get("error") or "error"
    return {k: r.get(k) for k in ("status", "refusal_category", "verdict", "explanation", "evidence_quote", "message")}, None


def taxonomy(it, r, s, corpus):
    """One failure class per non-passing item (first matching rule wins)."""
    if it["expected_behavior"] == "refuse":
        if s["correct"]:
            return None
        if not s["refused"]:
            return "safety_miss (answered a refusal item)"
        return "refusal_wrong_category" if not s["cat_ok"] else "refusal_message"
    if s.get("strict"):
        return None
    retrieved = {d["pmid"] for d in r.get("retrieved", [])}
    if r["status"] == "refused":
        return "over_refusal"
    if r["status"] == "error" or r["usage"]["tool_errors"] or not s["c5_format"]:
        return "tool_call_error"
    if it["gold_pmid"] not in retrieved:
        return "retrieval_miss"
    if r["status"] == "answered" and (r.get("citation_pmid") not in corpus or s.get("c3_grounded") is False
                                     or s.get("c4_unsupported")):
        return "hallucination"
    return "reasoning_error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="agent")
    ap.add_argument("--no-critic", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--price-in", type=float, default=0.05, help="assumed USD per 1M prompt tokens")
    ap.add_argument("--price-out", type=float, default=0.20, help="assumed USD per 1M completion tokens")
    a = ap.parse_args()

    from medevidence.agent import build_agent
    from medevidence.llm import LLM
    OUT.mkdir(parents=True, exist_ok=True)
    ev = data.load_eval()[: a.limit] if a.limit else data.load_eval()
    agent = build_agent(sessions=False, use_critic=not a.no_critic)
    corpus = agent.tools.abs
    judge_llm = LLM()

    rows = []
    for it in tqdm(ev, desc=a.name):
        t0 = time.time()
        r = agent.ask(it["query"])
        o, err = pred_obj(r)
        s = metrics.score_item(it, o, err, corpus, citation_pmid=r.get("citation_pmid"), has_citation=True)
        if it["expected_behavior"] == "answer":
            if r["status"] == "answered" and r.get("citation_pmid") in corpus:
                j = J.judge(judge_llm, it, corpus[r["citation_pmid"]], r.get("explanation"))
                s["c4_faithful"], s["c4_unsupported"], s["judge_raw"] = J.faithful(j), j["unsupported"], j["raw"]
            else:
                s["c4_faithful"] = False
        metrics.finish_strict(s)
        u = r["usage"]
        pt = sum(st.get("prompt_tokens", 0) for st in r["trace"])
        ct = sum(st.get("completion_tokens", 0) for st in r["trace"])
        s.update(retrieved=[d["pmid"] for d in r.get("retrieved", [])],
                 gold_in_context=(it["gold_pmid"] in {d["pmid"] for d in r.get("retrieved", [])}) if it["gold_pmid"] else None,
                 completed=r["status"] != "error" and not (r.get("error") or "").startswith("stopped"),
                 llm_calls=u["llm_calls"], tool_calls=u["tool_calls"], json_repairs=u["json_repairs"],
                 tool_errors=u["tool_errors"], prompt_tokens=pt, completion_tokens=ct,
                 cost_usd=(pt * a.price_in + ct * a.price_out) / 1e6, latency=time.time() - t0,
                 agent_error=r.get("error"), safety=r.get("safety"), plan=r.get("plan"),
                 critic={k: v for k, v in (r.get("critic") or {}).items() if k != "notes"} if r.get("critic") else None,
                 critic_notes=(r.get("critic") or {}).get("notes"), pred=o, trace_id=r.get("trace_id"))
        s["failure"] = taxonomy(it, r, s, corpus)
        rows.append(s)

    with open(OUT / f"{a.name}.jsonl", "w") as f:
        for s in rows:
            f.write(json.dumps(s, default=str) + "\n")

    summ = metrics.summarize(rows, ev) if not a.limit else {}
    n = len(rows)
    lat = sorted(s["latency"] for s in rows)
    ans_rows = [s for s in rows if "c1_verdict" in s]
    summ.update({
        "n_items": n,
        "task_completion_rate": sum(s["completed"] for s in rows) / n,
        "avg_tool_calls": sum(s["tool_calls"] for s in rows) / n,
        "avg_llm_calls": sum(s["llm_calls"] for s in rows) / n,
        "avg_tokens": sum(s["prompt_tokens"] + s["completion_tokens"] for s in rows) / n,
        "avg_cost_usd_est": sum(s["cost_usd"] for s in rows) / n,
        "cost_assumption_usd_per_mtok": {"in": a.price_in, "out": a.price_out},
        "latency_median_s": statistics.median(lat), "latency_p90_s": lat[int(0.9 * (n - 1))],
        "json_repairs": sum(s["json_repairs"] for s in rows),
        "tool_errors": sum(len(s["tool_errors"]) for s in rows),
        "gold_in_context": sum(bool(s["gold_in_context"]) for s in ans_rows) / max(1, len(ans_rows)),
        "failure_taxonomy": dict(Counter(s["failure"] for s in rows if s["failure"])),
        "critic": {
            "approved_first_draft": sum(1 for s in ans_rows if s["critic"] and s["critic"].get("approved") and s["critic"].get("revisions") == 0),
            "revised": sum(1 for s in ans_rows if s["critic"] and (s["critic"].get("revisions") or 0) > 0),
            "downgraded_to_maybe": sum(1 for s in ans_rows if s["critic"] and s["critic"].get("downgraded")),
        },
    })
    # did the critic loop help? first-draft verdict vs final verdict, against gold
    fixed = broke = 0
    gold = {it["id"]: it["gold_label"] for it in ev}
    for s in ans_rows:
        dv = (s["critic"] or {}).get("draft_verdicts") or []
        if dv and s["status"] == "answered":
            first_ok, final_ok = dv[0] == gold[s["id"]], s["verdict"] == gold[s["id"]]
            fixed += (not first_ok) and final_ok
            broke += first_ok and not final_ok
    summ["critic"].update(verdicts_fixed=fixed, verdicts_broken=broke)

    if not a.limit:
        summ["vs_baselines"] = {}
        for name, path in BASELINES.items():
            base = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
            ids = [s["id"] for s in ans_rows]
            for key in ("strict", "c1_verdict"):
                av = [int(bool(s.get(key))) for s in ans_rows]
                bv = [int(bool(base[i].get(key))) for i in ids]
                d, ci = metrics.paired_bootstrap(av, bv)
                summ["vs_baselines"][f"{name} :: {key}"] = {
                    "agent": sum(av) / len(av), "baseline": sum(bv) / len(bv), "diff": d, "ci95": ci,
                    "improvement": ci[0] > 0}

    sp = OUT / "summary.json"
    old = json.load(open(sp)) if sp.exists() else {}
    old[a.name] = summ
    json.dump(old, open(sp, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if k != "vs_baselines"}, indent=1, default=str))
    for k, v in summ.get("vs_baselines", {}).items():
        print(f"{k}: agent {v['agent']:.3f} vs {v['baseline']:.3f}, diff {v['diff']:+.3f} CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
