"""Eval set, PubMedQA loading and the retrieval corpus.

Leakage rules (spec 6.1): the index holds abstracts only. The authors' conclusion
(LONG_ANSWER) is never indexed. PQA-L test PMIDs are never used for few-shot/tuning.
"""
import json
import random

import pandas as pd

from .config import CORPUS_PATH, EVAL_PATH, HF_DIR, RAW_DIR


def load_eval():
    return [json.loads(l) for l in open(EVAL_PATH)]


def load_pqal():
    return json.load(open(RAW_DIR / "ori_pqal.json"))


def test_pmids():
    return set(json.load(open(RAW_DIR / "test_ground_truth.json")))


def abstract_text(contexts, labels=None):
    if labels and len(labels) == len(contexts):
        return "\n".join(f"{l}: {c}" for l, c in zip(labels, contexts))
    return "\n".join(contexts)


def build_corpus(seed=42, n_artificial=500, n_unlabeled=500):
    """1000 PQA-L abstracts (incl. all 500 test abstracts) + random PQA-A/U distractors."""
    pq = load_pqal()
    docs = []
    for pmid, d in pq.items():
        docs.append({"pmid": pmid, "source": "pqal", "sections": d["CONTEXTS"], "labels": d["LABELS"],
                     "text": abstract_text(d["CONTEXTS"], d["LABELS"]), "year": d.get("YEAR")})
    seen = set(pq)
    rng = random.Random(seed)
    for name, n, src in (("pqa_artificial", n_artificial, "pqaa"), ("pqa_unlabeled", n_unlabeled, "pqau")):
        df = pd.read_parquet(HF_DIR / f"{name}.parquet")
        ids = [i for i in range(len(df)) if str(df.iloc[i]["pubid"]) not in seen]
        for i in rng.sample(ids, n):
            r = df.iloc[i]
            ctx = [str(x) for x in r["context"]["contexts"]]
            lab = [str(x) for x in r["context"]["labels"]]
            pmid = str(r["pubid"])
            if pmid in seen or not ctx:
                continue
            seen.add(pmid)
            docs.append({"pmid": pmid, "source": src, "sections": ctx, "labels": lab,
                         "text": abstract_text(ctx, lab), "year": None})
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return docs


def load_corpus():
    if not CORPUS_PATH.exists():
        return build_corpus()
    return [json.loads(l) for l in open(CORPUS_PATH)]
