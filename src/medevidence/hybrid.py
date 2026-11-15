"""PMID-level retrieval: BM25 + dense over chunks, RRF fusion, cross-encoder reranking."""
import json

import numpy as np

from .config import ROOT
from .retrieval import BM25

IDX = ROOT / "data" / "index"
RERANKERS = {"medcpt": "ncbi/MedCPT-Cross-Encoder", "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2"}


def pmid_rank(scores, pmids, k=100):
    """Aggregate chunk scores to PMIDs by max; return ranked [(pmid, score)]."""
    best = {}
    for i in np.argsort(-scores)[: k * 8]:
        p = pmids[i]
        if p not in best:
            best[p] = float(scores[i])
            if len(best) >= k:
                break
    return sorted(best.items(), key=lambda x: -x[1])


def rrf(rank_lists, k=60, top=100):
    s = {}
    for rl in rank_lists:
        for r, (p, _) in enumerate(rl):
            s[p] = s.get(p, 0.0) + 1.0 / (k + r + 1)
    return sorted(s.items(), key=lambda x: -x[1])[:top]


class Index:
    """One chunking strategy: chunk texts, PMIDs, BM25 and dense matrix."""

    def __init__(self, name):
        ch = json.load(open(IDX / f"chunks_{name}.json"))
        self.name = name
        self.pmids = [c["pmid"] for c in ch]
        self.texts = [c["text"] for c in ch]
        self.bm = BM25(self.texts, self.pmids)
        self.emb = np.load(IDX / f"emb_{name}.npy")

    def bm25(self, q, k=100):
        return pmid_rank(self.bm.scores(q), self.pmids, k)

    def dense(self, qvec, k=100):
        return pmid_rank(self.emb @ qvec, self.pmids, k)

    def hybrid(self, q, qvec, k=100):
        return rrf([self.bm25(q, k), self.dense(qvec, k)], top=k)


class Reranker:
    def __init__(self, key, abstracts):
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(RERANKERS[key], max_length=512, device="mps")
        self.abs = abstracts  # pmid -> text
        self.cache = {}

    def rerank(self, q, cands):
        """cands: ranked [(pmid, score)] -> reranked [(pmid, ce_score)]."""
        todo = [p for p, _ in cands if (q, p) not in self.cache]
        if todo:
            sc = self.ce.predict([(q, self.abs[p]) for p in todo], batch_size=16, show_progress_bar=False)
            for p, s in zip(todo, sc):
                self.cache[(q, p)] = float(s)
        return sorted(((p, self.cache[(q, p)]) for p, _ in cands), key=lambda x: -x[1])


def recall_at(ranked, gold, ks=(1, 3, 5, 10)):
    ids = [p for p, _ in ranked]
    out = {f"R@{k}": float(gold in ids[:k]) for k in ks}
    out["MRR"] = 1.0 / (ids.index(gold) + 1) if gold in ids else 0.0
    return out
