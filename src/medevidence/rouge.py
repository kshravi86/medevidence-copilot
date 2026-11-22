"""ROUGE-L (longest common subsequence) between a model explanation and the authors' conclusion.

Reference = the eval item's `gold_rationale` (PubMedQA LONG_ANSWER). It is used only for scoring:
the conclusion is never indexed or shown to the model (spec 6.1 leakage rules).
Uses Google's rouge-score package (Porter stemming, standard tokenisation); reports F1, precision, recall.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _scorer():
    from rouge_score import rouge_scorer
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction, reference):
    """Returns {"f": F1, "p": precision, "r": recall}; all 0 if either text is missing."""
    if not prediction or not reference:
        return {"f": 0.0, "p": 0.0, "r": 0.0}
    s = _scorer().score(reference, prediction)["rougeL"]
    return {"f": s.fmeasure, "p": s.precision, "r": s.recall}
