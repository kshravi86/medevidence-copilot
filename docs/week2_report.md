# Week 2 report: RAG over a real corpus

Track B (API-side), MedEvidence Copilot. Same frozen eval set as Week 1 (`eval/eval_set.jsonl`). Single-pass RAG: retrieve, then generate a cited answer. There is **no planner, critic or safety gate yet** (Week 3).

## 1. Week 2 deliverables

| Requirement (course brief) | Status | Where |
|---|---|---|
| RAG over a real corpus of 500-2,000 documents | done: 2,000 abstracts (1,000 PQA-L + 500 PQA-A + 500 PQA-U distractors) | `scripts/build_corpus.py` |
| Chunking experiments | done: 3 strategies compared at retrieval level (section 3); end-to-end only for whole-abstract chunks | `scripts/build_index.py`, `scripts/eval_retrieval.py` |
| Hybrid retrieval (BM25 + dense) | done: reciprocal rank fusion, k=60 | `src/medevidence/hybrid.py` |
| A reranker | done: MedCPT cross-encoder (biomedical) vs MiniLM (general) | `src/medevidence/hybrid.py` |
| Citation-grounded answers | done: PMID + verbatim evidence quote, checked automatically | `src/medevidence/prompts.py`, `metrics.py` |
| Retrieval recall@k measured separately from end-to-end answer quality | done: sections 3 and 5 | `results/week2/` |

## 2. System and documented decisions

- **Dense model:** `bge-small-en-v1.5` (384-d) from the shared endpoint. The endpoint rejects more than 512 tokens, so inputs are cut at 500 tokens with the model's own tokenizer: 79 of 2,000 whole abstracts (4.0%) were truncated.
  Queries use the recommended BGE instruction prefix; it made little difference (section 3).
- **Chunking:** `abstract` (1 chunk per abstract), `section` (1 per labelled section, 6,557 chunks), `window` (100 words, stride 50, 7,120 chunks); `abstract` has 2,000. Chunk scores are aggregated to PMID by max. **Decision: whole-abstract chunks**, because the three strategies are indistinguishable here and it is the simplest (details below).
- **Fusion:** BM25 and dense lists fused with RRF (k=60); no weight tuning.
- **Reranker:** local cross-encoder over the whole abstract of the top-20 fused candidates. **Decision: MedCPT** (trained on PubMed search logs). Note: this model runs locally, not through the hosted API; the brief's API-only constraint applies to the generator.
- **Generator:** `Qwen/Qwen3-8B`, structured JSON + 3 answered and 3 refusal few-shot demos, T=0. It sees the top-k full abstracts, each headed `[PMID n]`, and must cite one PMID and copy a verbatim quote.
- **Leakage:** the authors' conclusion is not indexed; demos come from the development split only.

## 3. Retrieval experiments (recall@k, PMID level, 78 answerable items)

Targets (spec 6.2): R@5 at least 0.95 verbatim and 0.85 paraphrase.

| Chunking | Method | verbatim R@1 | verbatim R@5 | paraphrase R@1 | paraphrase R@5 | R@10 (all) | MRR (all) |
|---|---|---|---|---|---|---|---|
| abstract | bm25 | 93.3% | 97.8% | 76.0% | 96.0% | 0.987 | 0.925 |
| abstract | dense_plain | 97.8% | 97.8% | 96.0% | 100.0% | 1.000 | 0.980 |
| abstract | dense_prefixed | 97.8% | 100.0% | 96.0% | 100.0% | 1.000 | 0.981 |
| abstract | hybrid | 97.8% | 100.0% | 88.0% | 96.0% | 0.987 | 0.966 |
| abstract | hybrid+rerank(minilm) | 97.8% | 100.0% | 96.0% | 96.0% | 1.000 | 0.983 |
| abstract | hybrid+rerank(medcpt) | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 | 1.000 |
| section | bm25 | 91.1% | 95.6% | 80.0% | 96.0% | 0.974 | 0.917 |
| section | dense_plain | 95.6% | 97.8% | 100.0% | 100.0% | 1.000 | 0.982 |
| section | dense_prefixed | 97.8% | 97.8% | 100.0% | 100.0% | 1.000 | 0.989 |
| section | hybrid | 95.6% | 97.8% | 88.0% | 96.0% | 0.987 | 0.957 |
| section | hybrid+rerank(minilm) | 97.8% | 100.0% | 92.0% | 92.0% | 0.987 | 0.971 |
| section | hybrid+rerank(medcpt) | 100.0% | 100.0% | 96.0% | 96.0% | 0.987 | 0.988 |
| window | bm25 | 93.3% | 97.8% | 80.0% | 96.0% | 0.974 | 0.920 |
| window | dense_plain | 97.8% | 97.8% | 96.0% | 100.0% | 0.987 | 0.982 |
| window | dense_prefixed | 95.6% | 100.0% | 100.0% | 100.0% | 1.000 | 0.983 |
| window | hybrid | 97.8% | 100.0% | 88.0% | 96.0% | 0.987 | 0.965 |
| window | hybrid+rerank(minilm) | 97.8% | 100.0% | 92.0% | 92.0% | 0.987 | 0.971 |
| window | hybrid+rerank(medcpt) | 100.0% | 100.0% | 96.0% | 96.0% | 0.987 | 0.988 |

**Reading.**
- Almost every configuration meets both targets, so the corpus is **easy**: distractors are random abstracts, not topical neighbours. On the 25-item paraphrase slice one item is 4 points, so most differences between chunking strategies are noise and no chunking strategy should be called better.
- **Dense beats BM25 on paraphrases** (R@1 about 96-100% vs 76-80%). **Plain RRF fusion is worse than dense alone on paraphrases** (R@1 88%) because BM25 drags the fused list down when the wording differs; the reranker then recovers that loss.
- **The MedCPT reranker** reaches 100% R@1 on both slices with whole-abstract chunks. With section or window chunks it gets 96% on paraphrases, where dense alone gets 96-100%, so it is not uniformly better than dense retrieval. It does beat the general MiniLM reranker in every chunking (paraphrase R@1 100% vs 96% for abstract, 96% vs 92% for the others). The query prefix made little difference.

## 4. End-to-end answer quality (all 100 items, frozen eval set)

| Retrieval config | Gold in context | Citation (C2) | Verdict acc | Macro-F1 | Strict acc | Grounded (C3) | Faithful (C4) | Halluc. PMIDs | Over-refusal | Refusal recall | Refusal fully correct | Avg tokens | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BM25, top-1 context | 88.5% | 88.5% | 64.1% | 0.601 | 60.3% | 84.6% | 88.5% | 0 | 0.0% | 81.8% | 72.7% | 2230 | 3.2s |
| BM25, top-3 | 96.2% | 94.9% | 62.8% | 0.557 | 57.7% | 88.5% | 89.7% | 0 | 0.0% | 90.9% | 86.4% | 2911 | 3.5s |
| BM25 + dense (RRF), top-3 | 97.4% | 94.9% | 61.5% | 0.555 | 59.0% | 88.5% | 92.3% | 0 | 0.0% | 81.8% | 77.3% | 2925 | 3.6s |
| BM25 + dense (RRF) + MedCPT rerank, top-3 | 100.0% | 97.4% | 65.4% | 0.552 | 59.0% | 87.2% | 93.6% | 0 | 0.0% | 86.4% | 81.8% | 2910 | 3.6s |

Week 1 baseline (b), for reference: strict 51.3%, verdict 57.7%, citation 88.5%.

## 5. Retrieval quality vs answer quality (measured separately)

| Config | Retrieval recall (abstract chunking) | Gold in generator context | Cited PMID correct | Strict acc |
|---|---|---|---|---|
| BM25, top-1 context | 88.5% (R@1) | 88.5% | 88.5% | 60.3% |
| BM25, top-3 | 96.2% (R@3) | 96.2% | 94.9% | 57.7% |
| BM25 + dense (RRF), top-3 | 97.4% (R@3) | 97.4% | 94.9% | 59.0% |
| BM25 + dense (RRF) + MedCPT rerank, top-3 | 100.0% (R@3) | 100.0% | 97.4% | 59.0% |

Failure taxonomy (answerable items, strict accuracy; counts out of 78):

| Cause | BM25 top-1 | BM25 top-3 | BM25 + dense top-3 | BM25 + dense + rerank top-3 |
|---|---|---|---|---|
| pass | 47 | 45 | 46 | 46 |
| retrieval miss (gold not in context) | 9 | 3 | 2 | 0 |
| wrong verdict (gold cited) | 20 | 26 | 26 | 25 |
| insufficient_evidence despite gold in context | 0 | 1 | 2 | 2 |
| quote not verbatim | 2 | 2 | 2 | 4 |
| unfaithful explanation (judge) | 0 | 1 | 0 | 1 |
| verdict accuracy when the gold abstract is in context | 71.0% | 64.0% | 63.2% | 65.4% |

Paired bootstrap (10,000 resamples):

| Comparison | Difference | 95% CI |
|---|---|---|
| strict: rerank_3 - bm25_1 | -1.3 pts | [-11.5, +9.0] |
| c2_citation: rerank_3 - bm25_1 | +9.0 pts | [+2.6, +16.7] |
| c2_citation: hybrid_3 - bm25_1 | +6.4 pts | [+0.0, +12.8] |
| c2_citation: bm25_3 - bm25_1 | +6.4 pts | [+1.3, +12.8] |
| strict: rerank_3 - hybrid_3 | +0.0 pts | [-9.0, +9.0] |
| c2_citation: rerank_3 - hybrid_3 | +2.6 pts | [+0.0, +6.4] |
| strict: week2 bm25_1 - week1 baseline(b) few-shot | +9.0 pts | [+3.8, +15.4] |
| strict: week2 bm25_3 - week1 baseline(b) few-shot | +6.4 pts | [-1.3, +14.1] |
| strict: week2 hybrid_3 - week1 baseline(b) few-shot | +7.7 pts | [-1.3, +16.7] |
| strict: week2 rerank_3 - week1 baseline(b) few-shot | +7.7 pts | [-2.6, +17.9] |

## 6. Findings

1. **Better retrieval raises citation accuracy but not strict accuracy.** The reranker lifts the share of questions whose gold abstract reaches the generator from 88.5% to 100.0% and citation accuracy by 9.0 points (95% CI +2.6 to +16.7, significant), yet strict accuracy stays flat at about 58-60% (differences between retrieval configs are within 3 points, CIs contain 0).
2. **The verdict step is the bottleneck, as in Week 1.** With the gold abstract in context, the 8B model still gets only 63 to 71% of verdicts right; wrong verdicts with the right abstract cited are the largest failure class in every config (20 to 26 of 78). Better retrieval cannot fix that; the Week 3 critic and planner must target it.
3. **Grounding is solid.** No hallucinated PMIDs and no cited PMID outside the retrieved set in any config. The evidence quote is verbatim for 85-89% of the 78 answerable items.
4. **The Week 1 baseline was not the strongest single prompt.** Week 2 with the *same* top-1 BM25 retrieval scores 60.3% strict vs Week 1's 51.3%, +9.0 points (CI +3.8 to +15.4), purely from a different prompt layout (multi-abstract format with PMID headers, citation key, "use only these abstracts"). **Consequence:** to claim that the agent improves on the baseline, compare against the stronger one, the Week 2 top-1 BM25 prompt (60.3%), which makes the +10 point target about 70% strict accuracy. This needs a team decision before Week 3.
5. **Safety is not met without a gate.** Refusal recall is 82-91% and no config reaches the 95% target. Misses are the disguised personal-diagnosis probes ("Studies show IBS is a diagnosis of exclusion... do I have IBS?", a PSA rise, own lab results). 1 or 2 emergencies are refused under the wrong category (`personal_diagnosis`), so the redirect would be wrong. All 7 emergencies are refused in every config. Over-refusal is 0%.

## 7. Limitations

- **Easy retrieval benchmark** (random distractors, 2,000 abstracts). Recall near 100% will not transfer to real PubMed. Harder distractors (same-topic abstracts) would separate the methods.
- **Chunking was compared at retrieval level only**; the generator always receives whole abstracts. Chunking therefore affects only candidate generation.
- **One run, one seed, 78 items:** differences under about 8 points are within noise; the paired bootstrap CIs above should be preferred to rankings.
- **C4 judge** is the same 8B model; the 20% human spot-check (`results/week1/judge_spotcheck.csv`, Week 1 outputs) has **not been done**, so faithfulness numbers are unverified.
- The reranker is a local model, not an API service.
- PubMedQA has no titles, so `citation.title` is not produced.

## 8. Reproduce

```bash
python scripts/build_corpus.py && python scripts/build_index.py      # corpus + embeddings (cached)
python scripts/eval_retrieval.py                                     # retrieval tables
python scripts/run_week2.py --chunking abstract --reranker medcpt --configs bm25@1 bm25@3 hybrid@3 rerank@3
python scripts/analyze_week2.py && python scripts/make_week2_report.py
```
