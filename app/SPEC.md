# Doctor test app (MedEvidence Copilot) - build spec

Goal: tiny local web app so clinicians can try the Week 2 RAG pipeline: type a research question, see verdict,
explanation, PMID citation, verbatim quote, retrieved abstracts; and see refusals for personal/dosing/emergency requests.
Stdlib only for the server (http.server), plain HTML/CSS/JS for the UI (no CDN, no build step). Port 8100. Never print or log secrets.
Python: /Users/aakashr/medevidence-copilot/.venv/bin/python . Package: src/medevidence (add `src` to sys.path).

## Files (each owned by ONE agent; do not edit files you do not own)
- app/pipeline.py  (agent "pipeline")
- app/server.py    (agent "server")
- app/static/index.html (agent "ui")  -- single file with inline CSS+JS

## Contract: app/pipeline.py
`class Pipeline:` `__init__(self)` loads corpus (medevidence.data.load_corpus), Index("abstract") from medevidence.hybrid,
Embedder (medevidence.embed) and LLM (medevidence.llm) ONCE. Look at scripts/run_week2.py and src/medevidence/*.py for exact APIs
(bge query prefix, how q_prefixed.npy was produced, RRF hybrid). Use hybrid top-3 retrieval (no reranker: keep startup fast).
`ask(self, question: str) -> dict` returns EXACTLY:
{"question": str,
 "status": "answered"|"refused"|"insufficient_evidence"|"error",
 "refusal_category": null|"personal_diagnosis"|"drug_dosing"|"emergency",
 "verdict": null|"yes"|"no"|"maybe", "explanation": null|str, "evidence_quote": null|str,
 "citation_pmid": null|str, "quote_verified": bool,   # evidence_quote found verbatim in cited abstract (metrics.quote_in_abstract)
 "citation_valid": bool,                              # cited PMID is among retrieved docs
 "message": null|str,                                 # refusal text; if refused use prompts.REFUSAL_MESSAGES[category] (code chooses, not the model)
 "retrieved": [{"pmid": str, "rank": int, "text": str}],
 "latency_s": float, "error": null|str}
Generation: prompts.rag_fewshot(question, docs), llm.chat(..., json_mode=True, max_tokens=500), metrics.parse_json.
Never raise: catch exceptions and return status "error". Empty/too-long (>1500 chars) question -> status "error".
Add `if __name__ == "__main__":` that asks one sample question and prints the JSON.

## Contract: app/server.py
Imports Pipeline lazily at startup, serves app/static/index.html at "/", `GET /api/health` -> {"ok":true,"docs":N},
`POST /api/ask` body {"question": str} -> pipeline.ask JSON. Because the LLM key allows ONE active request, guard ask with a threading.Lock
and use ThreadingHTTPServer. Bind 127.0.0.1:8100. Also `GET /api/examples` -> list of {"label","question"}: 4 research questions, plus
1 personal_diagnosis, 1 drug_dosing, 1 emergency example (invent them; do not use eval-set items).

## Contract: app/static/index.html
Calls the endpoints above. Layout: header with title + disclaimer ("Research evidence lookup only, not clinical advice"), textarea + Ask button,
example chips from /api/examples, result card: big verdict badge (yes/no/maybe), explanation, quote highlighted with a green "verified verbatim"
or red "NOT found in abstract" tag, PMID linked to https://pubmed.ncbi.nlm.nih.gov/<pmid>/, collapsible "Retrieved abstracts" with the cited one marked.
Refusals: amber card with category and message (emergency = red). Error/insufficient_evidence: neutral card. Loading state, latency shown,
a thumbs up/down + optional comment feedback that POSTs /api/feedback {question,pmid,verdict,rating,comment}; the UI must still work if it 404s.
Responsive, accessible (labels, focus, contrast), light+dark via prefers-color-scheme. Escape all model text (textContent, never innerHTML with model output).
Also: server.py must implement POST /api/feedback appending a JSON line to app/feedback.jsonl and returning {"ok":true}.
