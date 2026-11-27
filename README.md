# MedEvidence Copilot

An agentic evidence-lookup assistant over 2,000 PubMed abstracts, built entirely on a hosted 8B model
(Qwen3-8B behind the course endpoint). Ask a population-level research question ("Does X improve Y in Z?") and
it answers **yes / no / maybe** with a short explanation, the **PMID** of one retrieved study and a
**word-for-word quote** from it. It refuses personal diagnosis, drug dosing for an individual and emergencies.

Capstone project, Track B (API side), Group 50: Executive PG Programme in Generative AI & Agentic AI
(IIT Kharagpur × upGrad).

| | |
|---|---|
| **Live system** | <https://huggingface.co/spaces/aakash108/medevidence-copilot> (CPU; the first question after it wakes from sleep takes about a minute) |
| **Technical report** | 12-page PDF submitted on upGrad; trace screenshots in its section 7 |
| **Deployment files** | `deploy/hf-space/` (Hugging Face Docker Space), `Dockerfile` + `docker-compose.yml` (local) |

## Results (frozen 100-question set: 78 answerable, 22 must-refuse)

| | Week 1 baseline (BM25 top-1, few-shot) | Week 2 best single prompt | **Week 3 agent** |
|---|---|---|---|
| Strict accuracy (verdict + citation + verbatim quote + faithful + format + not refused) | 51.3% | 60.3% | **69.2%** |
| Verdict accuracy | 57.7% | 64.1% | **70.5%** |
| Citation correct | 88.5% | 88.5% | **96.2%** |
| Refusal recall | 90.9% | 81.8% | **100%** (7/7 emergencies) |
| Hallucinated PMIDs | 0 | 0 | **0** |

Agent vs the Week 1 baseline: **+17.9 points strict, 95% CI [+7.7, +28.2]** (paired bootstrap). Task completion 100%,
3.4 LLM calls and 7.8 tool calls per question, about 4,000 tokens, 12.4 s median latency. The full failure taxonomy,
CIs and per-week analysis are in `docs/week1_report.md`, `docs/week2_report.md`, `docs/week3_report.md` and `results/`.

## Setup from a fresh clone

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cp .env.example .env            # fill in the course endpoint URLs, keys and CA certificate paths (never committed)

# Data: public PubMedQA files (PQA-L from GitHub, PQA-A/PQA-U distractors from Hugging Face); prints SHA-256s
.venv/bin/python scripts/download_data.py
.venv/bin/python scripts/build_eval_set.py      # rebuilds eval/eval_set.jsonl byte for byte (sha256 1a5bef…6e56)
.venv/bin/python scripts/build_corpus.py        # 2,000 abstracts -> data/corpus.jsonl
.venv/bin/python scripts/build_index.py         # BM25 chunks + BGE embeddings (three chunkings)
.venv/bin/python scripts/build_faiss.py         # FAISS indexes

.venv/bin/python -m pytest tests -q             # 29 unit tests, no endpoint needed
```

`data/` and `.cache/` are git-ignored. Model and embedding calls are cached in `.cache/*.sqlite`, so re-running a
script reproduces the committed results without new API calls.

## Week 1: specification, frozen eval set and prompting baselines

- `docs/week1_report.md`: the specification (what the copilot answers; what it refuses: personal diagnosis, drug
  dosing, emergencies, with precedence emergency > dosing > diagnosis), the answer contract and the baseline results.
- `scripts/build_eval_set.py` → `eval/eval_set.jsonl`: 78 answerable PQA-L test questions (verbatim, paraphrased,
  boundary) and 22 hand-written refusal probes, frozen before any agent code. Few-shot examples come only from the
  development split.
- `scripts/run_week1.py`: four prompting strategies (structured + few-shot, chain-of-thought, self-consistency,
  decomposition), closed-book and with the BM25 top-1 abstract → `results/week1/`.

## Week 2: retrieval-augmented generation

- Three chunkings (whole abstract, labelled section, 100-word window), BM25 + dense (`bge-small-en-v1.5`) fused
  with reciprocal rank fusion, and a MedCPT cross-encoder reranker.
- Retrieval measured separately from answers: `scripts/eval_retrieval.py` → `results/week2/retrieval_results.json`.
- `scripts/run_week2.py --chunking abstract --configs bm25@1 bm25@3 hybrid@3 rerank@3` → `results/week2/`.

## Week 3: agent (safety gate → planner → executor → critic)

Code: `src/medevidence/agent/` (hand-rolled, no framework). Every prompt it sends is listed in
`docs/agent_system_prompts.txt`.

| Part | File | What it does |
|---|---|---|
| Safety gate | `agent/safety.py` | Runs first. Regex rules (dedicated emergency check, personal dosing/diagnosis) **plus** an LLM classifier; refuses if either does, precedence emergency > dosing > diagnosis. The refusal text is a fixed template chosen by code. Emergency rule hits skip the model entirely. |
| Planner | `agent/agent.py`, `agent/prompts_agent.py` | Restates the question as a population-level research question, writes up to 3 queries (the user's own words are always one), picks how many abstracts to read. Uses session history for follow-ups. Re-plans once with new wording if nothing relevant was found. |
| Tools | `agent/tools.py` | `search_abstracts` (BM25 + dense RRF, MedCPT rerank; BM25-only fallback if the embedding endpoint fails), `get_abstract`, `verify_quote`, `conclusion_sentences`. Argument validation, one retry, errors returned (never raised). |
| Executor | `agent/agent.py` | Drafts verdict, explanation, verbatim quote and PMID from the retrieved abstracts (dev-split few-shots). |
| Critic | `agent/agent.py` | Code checks first (PMID among results, quote verbatim, format), then an LLM check of the verdict and faithfulness. Sends the draft back up to 2 times, then downgrades to `maybe` and records the reason. |
| State | `agent/state.py` | Per-request budget (30 steps, 30k tokens, 2 revisions, 1 re-plan) and SQLite sessions (`.cache/sessions.sqlite`) that persist across turns and restarts. |
| Tracing | `agent/tracing.py` | OpenTelemetry spans for every step, exported as JSON lines to `results/traces/spans.jsonl` (`MEDEVIDENCE_TRACE_FILE`) and, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, to Jaeger. |

```bash
.venv/bin/python scripts/run_week3.py --name agent   # full eval on eval/eval_set.jsonl -> results/week3/
.venv/bin/python scripts/run_week3.py --name agent_nocritic --no-critic   # critic ablation
```

`run_week3.py` reports the spec metrics plus task completion, tool/LLM calls, tokens, estimated cost and
latency per query, a failure taxonomy (retrieval miss / reasoning error / tool-call error / hallucination),
and paired-bootstrap CIs against both baselines (Week 1 BM25 few-shot, and the stronger Week 2 BM25 top-1 prompt).

**Trace dashboard (Jaeger):**
```bash
docker run -d -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one:1.62.0
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 .venv/bin/python app/server.py   # ask a question, then open http://localhost:16686
```

## Week 4: ship it

| | |
|---|---|
| API | `app/server.py`: `POST /api/ask {"question", "session_id"?}`, `POST /api/ask_stream` (server-sent events, one per step), `GET /api/health`, `GET /api/examples`, `POST /api/feedback`. `MEDEVIDENCE_MODE=agent` (default) or `rag` (Week 2 pipeline). |
| Rate limiting | Token bucket per client (`RATE_LIMIT_PER_MIN`=10, `RATE_LIMIT_BURST`=5; the public Space uses 6 and 3) → HTTP 429 with `Retry-After`. |
| Caching | TTL + LRU answer cache for stateless questions (`CACHE_TTL_S`=3600, `CACHE_SIZE`=256), normalised by case/whitespace; plus the on-disk LLM and embedding caches. Session follow-ups bypass it. |
| Front-ends | `app/streamlit_app.py` (chat with per-browser session, verdict, verified quote, PMID link, critic/plan/trace panel, feedback) and `app/static/eval.html` (live page that streams every agent step; the front page of the public Space). |
| Containers | `Dockerfile` (CPU torch, reranker and tokenizer baked in, non-root user) and `docker-compose.yml` (`api` on 8100, `ui` on 8501). |
| Public deployment | `deploy/hf-space/`: `prepare.sh <dir>` stages code, eval set, abstract index and the PubMedQA file used for few-shot examples; API keys and CA certificates are Space secrets, never in the image. |
| Responsible AI | `scripts/redteam_week4.py` → `results/week4/`: 22 new adversarial cases (prompt injection planted in a retrieved abstract 8/8, jailbreaks in the message 8/8, personal data in the question 3/6 with the gap documented). |

Local:
```bash
.venv/bin/python app/server.py                       # http://127.0.0.1:8100
API_URL=http://127.0.0.1:8100 .venv/bin/streamlit run app/streamlit_app.py   # http://127.0.0.1:8501
```

Docker (needs `.env`, `certs/` and `data/` in the repo folder; they are mounted, never copied into the image):
```bash
docker compose up --build    # UI: http://127.0.0.1:8501, API: http://127.0.0.1:8100
```

## Vector store (FAISS) and ROUGE-L

**FAISS.** Dense retrieval uses a FAISS index per chunking (`src/medevidence/vectorstore.py`) instead of a numpy scan.
`IndexFlatIP` (exact, default) or `IndexHNSWFlat` (approximate, `FAISS_KIND=hnsw`) over the BGE embeddings;
set `USE_FAISS=0` to fall back to numpy. Build once after `build_index.py` (the files live in the git-ignored `data/index/`):
```bash
.venv/bin/python scripts/build_faiss.py    # writes data/index/faiss_<chunking>_<flat|hnsw>.index, checks vs numpy
```
Check on the 100 eval queries (`results/faiss_check.json`): flat returns the identical top-10 PMIDs as numpy for every query;
HNSW 97–100% identical, same R@1/R@5; under 0.3 ms per query.

**ROUGE-L.** Longest-common-subsequence overlap between the generated explanation and the authors' conclusion
(`gold_rationale`, used only for scoring), via Google's `rouge-score` with stemming (`src/medevidence/rouge.py`).
```bash
.venv/bin/python scripts/eval_rouge.py     # scores every Week 1-3 run from its saved rows -> results/rouge_l.json
```
`run_week3.py` also reports it per item (`rougeL`) and on average.

## Data and licences

PubMedQA (Jin et al., 2019; MIT licence): public, de-identified research abstracts. Nothing is fine-tuned and no
patient data is used. This is a research evidence-lookup tool, not clinical advice.
