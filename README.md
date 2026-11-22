# medevidence-copilot
MedEvidence Copilot: agentic medical research assistant (PubMedQA, 8B model) - capstone project

## Week 3: agent (safety gate → planner → executor → critic)

Code: `src/medevidence/agent/` (hand-rolled, no framework).

| Part | File | What it does |
|---|---|---|
| Safety gate | `agent/safety.py` | Runs first. Regex rules (dedicated emergency check, personal dosing/diagnosis) **plus** an LLM classifier; refuses if either does, precedence emergency > dosing > diagnosis. The refusal text is a fixed template chosen by code. Emergency rule hits skip the model entirely. |
| Planner | `agent/agent.py`, `agent/prompts_agent.py` | Restates the question as a population-level research question, writes up to 3 queries (the user's own words are always one), picks how many abstracts to read. Uses session history for follow-ups. Re-plans once with new wording if nothing relevant was found. |
| Tools | `agent/tools.py` | `search_abstracts` (BM25 + dense RRF, MedCPT rerank; BM25-only fallback if the embedding endpoint fails), `get_abstract`, `verify_quote`, `conclusion_sentences`. Argument validation, one retry, errors returned (never raised). |
| Executor | `agent/agent.py` | Drafts verdict, explanation, verbatim quote and PMID from the retrieved abstracts (dev-split few-shots). |
| Critic | `agent/agent.py` | Code checks first (PMID among results, quote verbatim, format), then an LLM check of the verdict and faithfulness. Sends the draft back up to 2 times, then downgrades to `maybe` and records the reason. |
| State | `agent/state.py` | Per-request budget (30 steps, 30k tokens, 2 revisions, 1 re-plan) and SQLite sessions (`.cache/sessions.sqlite`) that persist across turns and restarts. |
| Tracing | `agent/tracing.py` | OpenTelemetry spans for every step, exported as JSON lines to `results/traces/spans.jsonl` (`MEDEVIDENCE_TRACE_FILE`). |

```bash
.venv/bin/python -m pytest tests -q                  # 27 unit tests, no endpoint needed
.venv/bin/python scripts/run_week3.py --name agent   # full eval on eval/eval_set.jsonl -> results/week3/
.venv/bin/python scripts/run_week3.py --name agent_nocritic --no-critic   # critic ablation
```

`run_week3.py` reports the spec 6.2 metrics plus task completion, tool/LLM calls, tokens, estimated cost and
latency per query, a failure taxonomy (retrieval miss / reasoning error / tool-call error / hallucination),
and paired-bootstrap CIs against both baselines (Week 1 BM25 few-shot, and the stronger Week 2 BM25 top-1 prompt).

## Week 4: ship it

| | |
|---|---|
| API | `app/server.py`: `POST /api/ask {"question", "session_id"?}`, `GET /api/health`, `GET /api/examples`, `POST /api/feedback`. `MEDEVIDENCE_MODE=agent` (default) or `rag` (Week 2 pipeline). |
| Rate limiting | Token bucket per client (`RATE_LIMIT_PER_MIN`=10, `RATE_LIMIT_BURST`=5) → HTTP 429 with `Retry-After`. |
| Caching | TTL + LRU answer cache for stateless questions (`CACHE_TTL_S`=3600, `CACHE_SIZE`=256), normalised by case/whitespace; plus the on-disk LLM and embedding caches. Session follow-ups bypass it. |
| Front-end | `app/streamlit_app.py` (chat with per-browser session, verdict, verified quote, PMID link, critic/plan/trace panel, feedback). The plain HTML test UI is still served at `/`. |
| Container | `Dockerfile` (CPU torch, reranker and tokenizer baked in, non-root user) and `docker-compose.yml` (`api` on 8100, `ui` on 8501). |

Local:
```bash
.venv/bin/python app/server.py                       # http://127.0.0.1:8100
API_URL=http://127.0.0.1:8100 .venv/bin/streamlit run app/streamlit_app.py   # http://127.0.0.1:8501
```

Docker (needs `.env`, `certs/` and `data/` in the repo folder; they are mounted, never copied into the image):
```bash
docker compose up --build    # UI: http://127.0.0.1:8501, API: http://127.0.0.1:8100
```
