"""Agent tools with defined interfaces (course brief IV: at least three tools + error handling).

Each tool takes typed keyword arguments and returns a ToolResult. Tools never raise to the agent:
exceptions become ToolResult(ok=False, error=...), are retried once, and recorded in the trace.

Tools
- search_abstracts(query, k)   hybrid BM25 + dense retrieval, MedCPT cross-encoder rerank -> [{pmid, score}]
                                (falls back to BM25 only if the embedding endpoint fails)
- get_abstract(pmid)           full abstract text + year for one PMID, or not_found
- verify_quote(pmid, quote)    is the quote verbatim in that abstract (spec C3 check)
- conclusion_sentences(pmid)   the abstract's last sentences, where the authors' result is usually stated
"""
import re
from dataclasses import dataclass, field

from .. import data
from ..metrics import quote_in_abstract

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class ToolResult:
    ok: bool
    value: object = None
    error: str | None = None
    meta: dict = field(default_factory=dict)


SCHEMAS = {
    "search_abstracts": {"query": "str (1-300 chars)", "k": "int 1-10"},
    "get_abstract": {"pmid": "str of digits"},
    "verify_quote": {"pmid": "str of digits", "quote": "str"},
    "conclusion_sentences": {"pmid": "str of digits", "n": "int 1-3"},
}


class Tools:
    """Owns the corpus, index, embedder and reranker. Loaded once per process."""

    def __init__(self, rerank=True, chunking="abstract"):
        from ..embed import Embedder
        from ..hybrid import Index, Reranker

        corpus = data.load_corpus()
        self.abs = {str(d["pmid"]): d["text"] for d in corpus}
        self.year = {str(d["pmid"]): d.get("year") for d in corpus}
        self.index = Index(chunking)
        self.embedder = Embedder()
        self.reranker = None
        if rerank:
            self.reranker = Reranker("medcpt", self.abs)  # device from RERANK_DEVICE (default mps)

    # --- dispatcher -------------------------------------------------------------------------
    def call(self, name, retries=1, **kwargs) -> ToolResult:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None or name not in SCHEMAS:
            return ToolResult(False, error=f"unknown_tool:{name}")
        bad = _validate(name, kwargs)
        if bad:
            return ToolResult(False, error=f"bad_args:{bad}")
        last = None
        for attempt in range(retries + 1):
            try:
                res = fn(**kwargs)
                res.meta["attempts"] = attempt + 1
                return res
            except Exception as e:  # tool failure -> retried, then reported, never raised
                last = f"{type(e).__name__}: {str(e)[:200]}"
        return ToolResult(False, error=f"tool_failed:{last}", meta={"attempts": retries + 1})

    # --- tools ------------------------------------------------------------------------------
    def _t_search_abstracts(self, query, k=5):
        meta = {}
        try:
            qv = self.embedder.embed([BGE_QUERY_PREFIX + query])[0]
            ranked = self.index.hybrid(query, qv)
            meta["mode"] = "hybrid"
        except Exception as e:
            ranked = self.index.bm25(query)
            meta.update(mode="bm25_fallback", fallback_reason=f"{type(e).__name__}: {str(e)[:120]}")
        if self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked[:20])
            meta["reranked"] = True
        return ToolResult(True, [{"pmid": str(p), "score": round(float(s), 4)} for p, s in ranked[:k]], meta=meta)

    def _t_get_abstract(self, pmid):
        if pmid not in self.abs:
            return ToolResult(False, error="not_found")
        return ToolResult(True, {"pmid": pmid, "text": self.abs[pmid], "year": self.year.get(pmid)})

    def _t_verify_quote(self, pmid, quote):
        if pmid not in self.abs:
            return ToolResult(False, error="not_found")
        return ToolResult(True, bool(quote_in_abstract(quote, self.abs[pmid])))

    def _t_conclusion_sentences(self, pmid, n=2):
        if pmid not in self.abs:
            return ToolResult(False, error="not_found")
        sents = [s for s in re.split(r"(?<=[.!?])\s+", self.abs[pmid].strip()) if s]
        return ToolResult(True, sents[-n:])


def _validate(name, kw):
    if name == "search_abstracts":
        q, k = kw.get("query"), kw.get("k", 5)
        if not isinstance(q, str) or not q.strip() or len(q) > 300:
            return "query must be a non-empty string <= 300 chars"
        if not isinstance(k, int) or not 1 <= k <= 10:
            return "k must be an int in 1..10"
    elif name in ("get_abstract", "verify_quote", "conclusion_sentences"):
        if not isinstance(kw.get("pmid"), str) or not kw["pmid"].isdigit():
            return "pmid must be a string of digits"
        if name == "verify_quote" and not isinstance(kw.get("quote"), str):
            return "quote must be a string"
        if name == "conclusion_sentences" and not isinstance(kw.get("n", 2), int):
            return "n must be an int"
    return None
