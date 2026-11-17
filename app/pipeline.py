import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from medevidence import data, metrics, prompts
from medevidence.embed import Embedder
from medevidence.hybrid import Index
from medevidence.llm import LLM

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Pipeline:
    def __init__(self):
        # Load corpus, index, embedder, and LLM ONCE
        corpus = data.load_corpus()
        self.absmap = {str(d["pmid"]): d["text"] for d in corpus}
        self.index = Index("abstract")
        self.embedder = Embedder()
        self.llm = LLM()

    def ask(self, question: str) -> dict:
        t0 = time.time()
        retrieved = []

        try:
            # Validate question
            if not isinstance(question, str) or not question.strip():
                return {
                    "question": str(question) if question is not None else "",
                    "status": "error",
                    "refusal_category": None,
                    "verdict": None,
                    "explanation": None,
                    "evidence_quote": None,
                    "citation_pmid": None,
                    "quote_verified": False,
                    "citation_valid": False,
                    "message": None,
                    "retrieved": [],
                    "latency_s": round(time.time() - t0, 3),
                    "error": "Question is empty or whitespace only.",
                }

            if len(question) > 1500:
                return {
                    "question": question,
                    "status": "error",
                    "refusal_category": None,
                    "verdict": None,
                    "explanation": None,
                    "evidence_quote": None,
                    "citation_pmid": None,
                    "quote_verified": False,
                    "citation_valid": False,
                    "message": None,
                    "retrieved": [],
                    "latency_s": round(time.time() - t0, 3),
                    "error": f"Question exceeds maximum allowed length of 1500 characters (length: {len(question)}).",
                }

            # Hybrid top-3 retrieval (BM25 + BGE dense with query prefix via RRF)
            q_emb = self.embedder.embed([f"{BGE_QUERY_PREFIX}{question}"])[0]
            ranked = self.index.hybrid(question, q_emb)
            top3_pmids = [str(p) for p, _ in ranked[:3]]

            retrieved = [
                {
                    "pmid": p,
                    "rank": i + 1,
                    "text": self.absmap.get(p, ""),
                }
                for i, p in enumerate(top3_pmids)
            ]

            # Generation with RAG prompt
            docs = [(r["pmid"], r["text"]) for r in retrieved]
            messages = prompts.rag_fewshot(question, docs)
            res = self.llm.chat(messages, json_mode=True, max_tokens=500)

            o, parse_err = metrics.parse_json(res.text)
            if o is None:
                return {
                    "question": question,
                    "status": "error",
                    "refusal_category": None,
                    "verdict": None,
                    "explanation": None,
                    "evidence_quote": None,
                    "citation_pmid": None,
                    "quote_verified": False,
                    "citation_valid": False,
                    "message": None,
                    "retrieved": retrieved,
                    "latency_s": round(time.time() - t0, 3),
                    "error": f"Failed to parse LLM JSON output: {parse_err}",
                }

            raw_status = o.get("status")
            if raw_status == "refused":
                raw_cat = o.get("refusal_category")
                refusal_category = (
                    raw_cat if raw_cat in ("personal_diagnosis", "drug_dosing", "emergency") else None
                )
                message = prompts.REFUSAL_MESSAGES.get(refusal_category, o.get("message") or "Request refused.")
                return {
                    "question": question,
                    "status": "refused",
                    "refusal_category": refusal_category,
                    "verdict": None,
                    "explanation": None,
                    "evidence_quote": None,
                    "citation_pmid": None,
                    "quote_verified": False,
                    "citation_valid": False,
                    "message": message,
                    "retrieved": retrieved,
                    "latency_s": round(time.time() - t0, 3),
                    "error": None,
                }

            elif raw_status == "insufficient_evidence":
                explanation = str(o["explanation"]) if o.get("explanation") is not None else None
                message = str(o["message"]) if o.get("message") is not None else None
                return {
                    "question": question,
                    "status": "insufficient_evidence",
                    "refusal_category": None,
                    "verdict": None,
                    "explanation": explanation,
                    "evidence_quote": None,
                    "citation_pmid": None,
                    "quote_verified": False,
                    "citation_valid": False,
                    "message": message,
                    "retrieved": retrieved,
                    "latency_s": round(time.time() - t0, 3),
                    "error": None,
                }

            elif raw_status == "answered":
                raw_verdict = o.get("verdict")
                verdict = raw_verdict if raw_verdict in ("yes", "no", "maybe") else None
                explanation = str(o["explanation"]) if o.get("explanation") is not None else None
                evidence_quote = str(o["evidence_quote"]) if o.get("evidence_quote") is not None else None

                raw_citation = o.get("citation_pmid")
                citation_pmid = (
                    str(raw_citation).strip()
                    if raw_citation is not None and str(raw_citation).strip() not in ("", "null", "None")
                    else None
                )

                retrieved_pmids = {r["pmid"] for r in retrieved}
                citation_valid = bool(citation_pmid and citation_pmid in retrieved_pmids)

                quote_verified = False
                if citation_pmid and citation_pmid in self.absmap and evidence_quote:
                    quote_verified = bool(metrics.quote_in_abstract(evidence_quote, self.absmap[citation_pmid]))

                return {
                    "question": question,
                    "status": "answered",
                    "refusal_category": None,
                    "verdict": verdict,
                    "explanation": explanation,
                    "evidence_quote": evidence_quote,
                    "citation_pmid": citation_pmid,
                    "quote_verified": quote_verified,
                    "citation_valid": citation_valid,
                    "message": None,
                    "retrieved": retrieved,
                    "latency_s": round(time.time() - t0, 3),
                    "error": None,
                }

            else:
                return {
                    "question": question,
                    "status": "error",
                    "refusal_category": None,
                    "verdict": None,
                    "explanation": None,
                    "evidence_quote": None,
                    "citation_pmid": None,
                    "quote_verified": False,
                    "citation_valid": False,
                    "message": None,
                    "retrieved": retrieved,
                    "latency_s": round(time.time() - t0, 3),
                    "error": f"Unexpected status from model: {raw_status}",
                }

        except Exception as e:
            return {
                "question": str(question) if question is not None else "",
                "status": "error",
                "refusal_category": None,
                "verdict": None,
                "explanation": None,
                "evidence_quote": None,
                "citation_pmid": None,
                "quote_verified": False,
                "citation_valid": False,
                "message": None,
                "retrieved": retrieved,
                "latency_s": round(time.time() - t0, 3),
                "error": str(e),
            }


if __name__ == "__main__":
    pipeline = Pipeline()
    sample_question = "Do statins reduce all-cause mortality in patients with heart failure?"
    result = pipeline.ask(sample_question)
    print(json.dumps(result, indent=2))
