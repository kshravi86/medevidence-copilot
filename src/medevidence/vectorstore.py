"""FAISS vector store for dense retrieval (replaces the brute-force numpy `emb @ q` scan).

One store per chunking strategy, built from the existing BGE embeddings (data/index/emb_<name>.npy,
L2-normalised, so inner product = cosine similarity). Row i of the store is chunk i of
chunks_<name>.json, so PMIDs come from the chunk list.

Index types
- "flat": IndexFlatIP, exact search. Same ranking as the numpy scan; the default.
- "hnsw": IndexHNSWFlat (inner product), approximate graph search for a larger corpus.
Build with: python scripts/build_faiss.py
"""
import os
from pathlib import Path

import numpy as np

from .config import ROOT

IDX = ROOT / "data" / "index"
KINDS = ("flat", "hnsw")


def store_path(name, kind):
    return IDX / f"faiss_{name}_{kind}.index"


class FaissStore:
    def __init__(self, index, kind):
        self.index, self.kind = index, kind

    @property
    def size(self):
        return self.index.ntotal

    @classmethod
    def build(cls, emb, kind="flat", hnsw_m=32, ef_construction=200, ef_search=128):
        import faiss
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        d = emb.shape[1]
        if kind == "flat":
            index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
        else:
            raise ValueError(f"unknown kind {kind!r}, expected one of {KINDS}")
        index.add(emb)
        return cls(index, kind)

    def save(self, path: Path):
        import faiss
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, name, kind=None):
        """Load data/index/faiss_<name>_<kind>.index, or None if FAISS is off, missing or not built."""
        kind = kind or os.environ.get("FAISS_KIND", "flat")
        if os.environ.get("USE_FAISS", "1") == "0":
            return None
        p = store_path(name, kind)
        if not p.exists():
            return None
        try:
            import faiss
        except ImportError:
            return None
        index = faiss.read_index(str(p))
        if kind == "hnsw":
            index.hnsw.efSearch = int(os.environ.get("FAISS_EF_SEARCH", "128"))
        return cls(index, kind)

    def search(self, qvec, k):
        """qvec: (d,) or (n, d). Returns (scores, row_ids) as (n, k) arrays; row id -1 = no result."""
        q = np.ascontiguousarray(np.atleast_2d(qvec), dtype=np.float32)
        return self.index.search(q, min(k, self.index.ntotal))
