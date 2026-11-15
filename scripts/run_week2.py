"""Week 2: end-to-end RAG (retrieve -> generate with citation) on the frozen eval set.
Retrieval configs are ablations; answer quality (C1-C6) is reported separately from recall@k.
Usage: python scripts/run_week2.py --chunking section --configs bm25@1 bm25@3 hybrid@3 rerank@3 ..."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from tqdm import tqdm
from medevidence import data, prompts, metrics, judge as J
from medevidence.hybrid import IDX, Index, Reranker, rrf
from medevidence.llm import LLM

OUT = ROOT / "results" / "week2"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--chunking", default="section")
ap.add_argument("--reranker", default="medcpt")
ap.add_argument("--configs", nargs="+", default=["bm25@1", "bm25@3", "hybrid@3", "rerank@3"])
args = ap.parse_args()

ev = data.load_eval()
absmap = {d["pmid"]: d["text"] for d in data.load_corpus()}
ix = Index(args.chunking)
Qx = np.load(IDX / "q_prefixed.npy")
rer = Reranker(args.reranker, absmap)
llm = LLM()
summary = {}
for cfg in args.configs:
    method, k = cfg.split("@"); k = int(k)
    rows = []
    for n, it in enumerate(tqdm(ev, desc=cfg)):
        q = it["query"]
        if method == "bm25":
            ranked = ix.bm25(q)
        elif method == "dense":
            ranked = ix.dense(Qx[n])
        elif method == "hybrid":
            ranked = ix.hybrid(q, Qx[n])
        else:  # rerank: hybrid top-20 -> cross-encoder
            ranked = rer.rerank(q, ix.hybrid(q, Qx[n])[:20])
        top = [p for p, _ in ranked[:k]]
        docs = [(p, absmap[p]) for p in top]
        res = llm.chat(prompts.rag_fewshot(q, docs), json_mode=True, max_tokens=500)
        o, e = metrics.parse_json(res.text)
        cited = str(o.get("citation_pmid")) if o and o.get("citation_pmid") else None
        r = metrics.score_item(it, o, e, absmap, citation_pmid=cited, has_citation=True)
        r.update(retrieved=top, gold_in_context=(it["gold_pmid"] in top) if it["gold_pmid"] else None,
                 tokens=res.prompt_tokens + res.completion_tokens, latency=res.latency, raw=res.text, pred=o)
        if it["expected_behavior"] == "answer":
            if o and o.get("status") == "answered" and cited in absmap:
                j = J.judge(llm, it, absmap[cited], o.get("explanation"))
                r["c4_faithful"] = J.faithful(j); r["judge_raw"] = j["raw"]
            else:
                r["c4_faithful"] = False
        metrics.finish_strict(r)
        rows.append(r)
    name = f"rag__{args.chunking}__{cfg.replace('@', '_')}"
    with open(OUT / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    s = metrics.summarize(rows, ev)
    s["gold_in_context"] = sum(bool(r["gold_in_context"]) for r in rows if r["gold_in_context"] is not None) / 78
    s["avg_tokens"] = sum(r["tokens"] for r in rows) / len(rows)
    s["median_latency_s"] = sorted(r["latency"] for r in rows)[len(rows) // 2]
    summary[name] = s
    print(name, json.dumps({a: (round(b, 3) if isinstance(b, float) else b) for a, b in s.items()}), flush=True)
sp = OUT / "summary.json"
old = json.load(open(sp)) if sp.exists() else {}
old.update(summary); json.dump(old, open(sp, "w"), indent=1)
