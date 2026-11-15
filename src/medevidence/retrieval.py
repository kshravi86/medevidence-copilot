"""BM25 index over documents or chunks, mapped back to PMIDs."""
import re

from rank_bm25 import BM25Okapi

STOP = set("a an the of in on at to for with and or is are was were be been by from as that this it its do does did "
           "can could should would may might not no than then there their which who whom what when where how".split())


def tokenize(t):
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP]


class BM25:
    def __init__(self, texts, ids):
        self.ids = ids
        self.bm = BM25Okapi([tokenize(t) for t in texts])

    def scores(self, q):
        return self.bm.get_scores(tokenize(q))

    def search(self, q, k=10):
        s = self.scores(q)
        top = s.argsort()[::-1][:k]
        return [(int(i), float(s[i])) for i in top]
