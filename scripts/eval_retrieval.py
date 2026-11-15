"""Week 2: retrieval-only evaluation (recall@k, MRR at PMID level), separate from answer quality.
chunking x retriever; slices verbatim / paraphrase. Needs data/index (scripts/build_index.py)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from medevidence import data
from medevidence.hybrid import IDX, Index, Reranker, recall_at

ev = [i for i in data.load_eval() if i["expected_behavior"] == "answer"]
allq = data.load_eval()
qpos = {i["id"]: n for n, i in enumerate(allq)}
Qp, Qx = np.load(IDX / "q_plain.npy"), np.load(IDX / "q_prefixed.npy")
absmap = {d["pmid"]: d["text"] for d in data.load_corpus()}
rer = {k: Reranker(k, absmap) for k in ("medcpt", "minilm")}
DEPTH = 20  # candidates passed to the reranker

rows = []
for name in ("abstract", "section", "window"):
    ix = Index(name)
    runs = {k: {} for k in ("bm25", "dense_plain", "dense_prefixed", "hybrid", "hybrid+rerank(medcpt)", "hybrid+rerank(minilm)", "hybrid_cand@20")}
    for it in ev:
        q, n = it["query"], qpos[it["id"]]
        b = ix.bm25(q)
        dp, dx = ix.dense(Qp[n]), ix.dense(Qx[n])
        from medevidence.hybrid import rrf
        hy = rrf([b, dx], top=100)
        runs["bm25"][it["id"]] = b
        runs["dense_plain"][it["id"]] = dp
        runs["dense_prefixed"][it["id"]] = dx
        runs["hybrid"][it["id"]] = hy
        for key, label in (("medcpt", "hybrid+rerank(medcpt)"), ("minilm", "hybrid+rerank(minilm)")):
            runs[label][it["id"]] = rer[key].rerank(q, hy[:DEPTH]) + hy[DEPTH:]
    for method, rr in runs.items():
        if method == "hybrid_cand@20":
            continue
        for sl in ("verbatim", "paraphrase", "boundary", "all"):
            items = [i for i in ev if sl == "all" or i["query_type"] == sl]
            m = [recall_at(rr[i["id"]], i["gold_pmid"], (1, 3, 5, 10, 20)) for i in items]
            rows.append({"chunking": name, "method": method, "slice": sl, "n": len(items),
                         **{k: round(sum(x[k] for x in m) / len(m), 4) for k in m[0]}})
    print("done", name, flush=True)
out = ROOT / "results" / "week2"; out.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(out / "retrieval_results.json", "w"), indent=1)
print("{:9s} {:24s} {:11s} {:>3s} {:>6s} {:>6s} {:>6s} {:>6s} {:>6s}".format("chunking", "method", "slice", "n", "R@1", "R@5", "R@10", "R@20", "MRR"))
for r in rows:
    if r["slice"] in ("verbatim", "paraphrase"):
        print("{:9s} {:24s} {:11s} {:3d} {:6.3f} {:6.3f} {:6.3f} {:6.3f} {:6.3f}".format(r["chunking"], r["method"], r["slice"], r["n"], r["R@1"], r["R@5"], r["R@10"], r["R@20"], r["MRR"]))
