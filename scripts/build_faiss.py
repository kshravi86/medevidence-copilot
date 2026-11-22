"""Build the FAISS vector stores and check them against the numpy brute-force search.

For each chunking (abstract, section, window) and index type (flat = exact, hnsw = approximate):
  - writes data/index/faiss_<chunking>_<kind>.index
  - runs the 100 frozen eval queries (data/index/q_prefixed.npy) and compares the top-10 PMIDs with
    the numpy scan, gold-PMID recall@1/@5 on the 78 answerable items, and search latency.
Results: results/faiss_check.json.  Usage: python scripts/build_faiss.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from medevidence import data  # noqa: E402
from medevidence.hybrid import IDX, pmid_rank  # noqa: E402
from medevidence.vectorstore import KINDS, FaissStore, store_path  # noqa: E402


def faiss_rank(store, pmids, q, k=10):
    scores, rows = store.search(q, k * 8)
    best = {}
    for s, i in zip(scores[0], rows[0]):
        if i >= 0 and pmids[i] not in best:
            best[pmids[i]] = float(s)
            if len(best) >= k:
                break
    return [p for p, _ in sorted(best.items(), key=lambda x: -x[1])]


def main():
    ev = data.load_eval()
    Q = np.load(IDX / "q_prefixed.npy")
    out = {}
    for name in ("abstract", "section", "window"):
        emb = np.load(IDX / f"emb_{name}.npy")
        pmids = [c["pmid"] for c in json.load(open(IDX / f"chunks_{name}.json"))]
        t = time.perf_counter()
        ref = [[p for p, _ in pmid_rank(emb @ Q[n], pmids, 10)] for n in range(len(ev))]
        np_ms = (time.perf_counter() - t) * 1000 / len(ev)
        for kind in KINDS:
            t = time.perf_counter()
            store = FaissStore.build(emb, kind)
            build_s = time.perf_counter() - t
            store.save(store_path(name, kind))
            t = time.perf_counter()
            got = [faiss_rank(store, pmids, Q[n]) for n in range(len(ev))]
            ms = (time.perf_counter() - t) * 1000 / len(ev)
            ans = [n for n, it in enumerate(ev) if it["gold_pmid"]]
            r = {
                "vectors": store.size, "build_s": round(build_s, 3),
                "file_mb": round(store_path(name, kind).stat().st_size / 1e6, 2),
                "top10_identical_to_numpy": sum(g == r_ for g, r_ in zip(got, ref)) / len(ev),
                "top1_same_as_numpy": sum(g[:1] == r_[:1] for g, r_ in zip(got, ref)) / len(ev),
                "R@1": sum(ev[n]["gold_pmid"] in got[n][:1] for n in ans) / len(ans),
                "R@5": sum(ev[n]["gold_pmid"] in got[n][:5] for n in ans) / len(ans),
                "numpy_R@1": sum(ev[n]["gold_pmid"] in ref[n][:1] for n in ans) / len(ans),
                "numpy_R@5": sum(ev[n]["gold_pmid"] in ref[n][:5] for n in ans) / len(ans),
                "ms_per_query": round(ms, 3), "numpy_ms_per_query": round(np_ms, 3),
            }
            out[f"{name}_{kind}"] = r
            print(f"{name:8s} {kind:5s} vectors={r['vectors']:5d} top10_identical={r['top10_identical_to_numpy']:.2f} "
                  f"R@1={r['R@1']:.3f} (numpy {r['numpy_R@1']:.3f}) R@5={r['R@5']:.3f} "
                  f"{r['ms_per_query']:.2f} ms/q (numpy {r['numpy_ms_per_query']:.2f})", flush=True)
    json.dump(out, open(ROOT / "results" / "faiss_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
