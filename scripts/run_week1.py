"""Week 1: prompting baselines on the frozen eval set.

settings  : closed = baseline (a) closed-book; bm25 = baseline (b) top-1 BM25 abstract
strategies: fewshot (structured JSON + few-shot) | cot (chain-of-thought, greedy)
            | cot_sc (CoT + self-consistency, n=5) | decompose (2-step decomposition)
Usage     : python scripts/run_week1.py [--only closed__fewshot ...]
"""
import argparse, collections, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tqdm import tqdm
from medevidence import data, prompts, metrics, judge as J
from medevidence.config import RESULTS
from medevidence.llm import LLM
from medevidence.retrieval import BM25

OUT = RESULTS / "week1"
OUT.mkdir(parents=True, exist_ok=True)
SETTINGS = ["closed", "bm25"]
STRATS = ["fewshot", "cot", "cot_sc", "decompose"]


def vote(objs):
    """Majority vote on (status, verdict, refusal_category); return the first sample with that key."""
    keys = [(o.get("status"), o.get("verdict"), o.get("refusal_category")) if o else None for o in objs]
    c = collections.Counter(k for k in keys if k)
    if not c:
        return None, 0
    top, n = c.most_common(1)[0]
    return objs[keys.index(top)], n


def run_item(llm, strat, setting, item, abstract):
    """Returns (parsed obj, parse error, token usage, latency, extra)."""
    q = item["query"]
    ab = abstract if setting == "bm25" else None
    s = "closed" if setting == "closed" else "open"
    tok, lat, extra = 0, 0.0, {}
    if strat == "fewshot":
        r = llm.chat(prompts.strategy_fewshot(q, ab, s), json_mode=True, max_tokens=500)
        o, e = metrics.parse_json(r.text); tok += r.prompt_tokens + r.completion_tokens; lat += r.latency
        extra["raw"] = r.text
    elif strat == "cot":
        r = llm.chat(prompts.strategy_cot(q, ab, s), json_mode=True, max_tokens=700)
        o, e = metrics.parse_json(r.text); tok += r.prompt_tokens + r.completion_tokens; lat += r.latency
        extra["raw"] = r.text
    elif strat == "cot_sc":
        r = llm.chat(prompts.strategy_cot(q, ab, s), json_mode=True, max_tokens=700, n=5, temperature=0.7, seed=7)
        objs = [metrics.parse_json(t)[0] for t in r.texts]
        o, agree = vote(objs); e = None if o else "no_valid_sample"
        tok += r.prompt_tokens + r.completion_tokens; lat += r.latency
        extra.update(raw=r.texts, vote_agreement=agree, n_valid=sum(x is not None for x in objs))
    elif strat == "decompose":
        r1 = llm.chat(prompts.decomp_step1(q, ab), json_mode=True, max_tokens=500)
        a, e1 = metrics.parse_json(r1.text); tok += r1.prompt_tokens + r1.completion_tokens; lat += r1.latency
        extra["step1"] = r1.text
        if a is None:
            o, e = None, "step1_" + (e1 or "fail")
        elif not a.get("in_scope", True) and a.get("refusal_category") in metrics.CATS:
            cat = a["refusal_category"]
            o, e = {"status": "refused", "refusal_category": cat, "verdict": None, "explanation": None,
                    "evidence_quote": None, "confidence": None, "message": prompts.REFUSAL_MESSAGES[cat]}, None
        else:
            r2 = llm.chat(prompts.decomp_step2(q, a.get("claim") or q, a.get("findings") or [], ab is not None),
                          json_mode=True, max_tokens=400)
            o, e = metrics.parse_json(r2.text); tok += r2.prompt_tokens + r2.completion_tokens; lat += r2.latency
            extra["step2"] = r2.text
            if o is not None:
                o.setdefault("refusal_category", None); o.setdefault("message", None)
    return o, e, tok, lat, extra


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    ev = data.load_eval()
    corpus = data.load_corpus()
    absmap = {d["pmid"]: d["text"] for d in corpus}
    bm = BM25([d["text"] for d in corpus], [d["pmid"] for d in corpus])
    top1 = {it["id"]: corpus[bm.search(it["query"], 1)[0][0]]["pmid"] for it in ev}
    llm = LLM()
    summary = {}
    for setting in SETTINGS:
        for strat in STRATS:
            name = f"{setting}__{strat}"
            if args.only and name not in args.only:
                continue
            rows = []
            for it in tqdm(ev, desc=name):
                pmid = top1[it["id"]] if setting == "bm25" else None
                o, e, tok, lat, extra = run_item(llm, strat, setting, it, absmap.get(pmid))
                r = metrics.score_item(it, o, e, absmap, citation_pmid=pmid, has_citation=(setting == "bm25"))
                r.update(tokens=tok, latency=lat, pred=o, **extra)
                if setting == "bm25" and it["expected_behavior"] == "answer":
                    r["top1_is_gold"] = (pmid == it["gold_pmid"])
                    if o and o.get("status") == "answered":
                        j = J.judge(llm, it, absmap[pmid], o.get("explanation")) if not args.no_judge else None
                        if j:
                            r["c4_faithful"] = J.faithful(j); r["judge_raw"] = j["raw"]
                    else:
                        r["c4_faithful"] = False
                metrics.finish_strict(r)
                rows.append(r)
            with open(OUT / f"{name}.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            s = metrics.summarize(rows, ev)
            s["avg_tokens"] = sum(r["tokens"] for r in rows) / len(rows)
            s["median_latency_s"] = sorted(r["latency"] for r in rows)[len(rows) // 2]
            if setting == "bm25":
                s["bm25_top1_hit"] = sum(r.get("top1_is_gold", False) for r in rows if "top1_is_gold" in r) / 78
            summary[name] = s
            print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    sp = OUT / "summary.json"
    old = json.load(open(sp)) if sp.exists() else {}
    old.update(summary)
    json.dump(old, open(sp, "w"), indent=1)
    (OUT / "eval_sha256.txt").write_text(hashlib.sha256(open(ROOT / "eval/eval_set.jsonl", "rb").read()).hexdigest() + "\n")


if __name__ == "__main__":
    main()
