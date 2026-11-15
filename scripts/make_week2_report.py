"""Generate docs/week2_report.md. Tables are read from results/week2/*.json (no hand-copied numbers)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
RT = json.load(open(ROOT / "results/week2/retrieval_results.json"))
S = json.load(open(ROOT / "results/week2/summary.json"))
A = json.load(open(ROOT / "results/week2/analysis.json"))
W1 = json.load(open(ROOT / "results/week1/summary.json"))["bm25__fewshot"]
p = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
ret = {(r["chunking"], r["method"], r["slice"]): r for r in RT}
methods = ["bm25", "dense_plain", "dense_prefixed", "hybrid", "hybrid+rerank(minilm)", "hybrid+rerank(medcpt)"]
rows = []
for ch in ("abstract", "section", "window"):
    for m in methods:
        v, q, a = ret[(ch, m, "verbatim")], ret[(ch, m, "paraphrase")], ret[(ch, m, "all")]
        rows.append(f"| {ch} | {m} | {p(v['R@1'])} | {p(v['R@5'])} | {p(q['R@1'])} | {p(q['R@5'])} | {a['R@10']:.3f} | {a['MRR']:.3f} |")
rtable = "\n".join(rows)
CFG = [("bm25_1", "bm25", 1, "BM25, top-1 context"), ("bm25_3", "bm25", 3, "BM25, top-3"), ("hybrid_3", "hybrid", 3, "BM25 + dense (RRF), top-3"),
       ("rerank_3", "hybrid+rerank(medcpt)", 3, "BM25 + dense (RRF) + MedCPT rerank, top-3")]
def e2e(n):
    s = S[f"rag__abstract__{n}"]
    return (f"| {dict((c[0], c[3]) for c in CFG)[n]} | {p(s['gold_in_context'])} | {p(s['citation_acc'])} | {p(s['verdict_acc_all'])} | {s['macro_f1']:.3f} | {p(s['strict_all'])} | "
            f"{p(s['grounded_acc'])} | {p(s['faithful_acc'])} | {s['hallucinated_pmids']} | {p(s['over_refusal_rate'])} | {p(s['refusal_recall'])} | {p(s['refusal_correct'])} | {s['avg_tokens']:.0f} | {s['median_latency_s']:.1f}s |")
etable = "\n".join(e2e(c[0]) for c in CFG)
sep = []
for n, m, k, lab in CFG:
    r = ret[("abstract", m, "all")]; s = S[f"rag__abstract__{n}"]
    sep.append(f"| {lab} | {p(r['R@1'] if k == 1 else r['R@3'])} (R@{k}) | {p(s['gold_in_context'])} | {p(s['citation_acc'])} | {p(s['strict_all'])} |")
septable = "\n".join(sep)
tx = A["taxonomy"]
tax_keys = ["pass", "retrieval miss (gold not in context)", "wrong verdict (gold cited)", "insufficient_evidence despite gold in context", "quote not verbatim", "unfaithful explanation (judge)"]
taxtable = "\n".join(f"| {k} | " + " | ".join(str(tx[n].get(k, 0)) for n, *_ in CFG) + " |" for k in tax_keys)
vacc = " | ".join(p(tx[n]["_verdict_acc_when_gold_in_context"]) for n, *_ in CFG)
bs = "\n".join(f"| {k} | {v['diff']*100:+.1f} pts | [{v['ci95'][0]*100:+.1f}, {v['ci95'][1]*100:+.1f}] |" for k, v in A["bootstrap"].items()
               if k.startswith(("c2_citation", "strict: rerank_3 - bm25_1", "strict: rerank_3 - hybrid", "strict: week2")))
best = S["rag__abstract__rerank_3"]; b1 = S["rag__abstract__bm25_1"]

nch = {n: len(json.load(open(ROOT / f"data/index/chunks_{n}.json"))) for n in ("abstract", "section", "window")}
vlo = min(tx[n]["_verdict_acc_when_gold_in_context"] for n, *_ in CFG); vhi = max(tx[n]["_verdict_acc_when_gold_in_context"] for n, *_ in CFG)
md = f"""# Week 2 report: RAG over a real corpus

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
- **Chunking:** `abstract` (1 chunk per abstract), `section` (1 per labelled section, {nch['section']:,} chunks), `window` (100 words, stride 50, {nch['window']:,} chunks); `abstract` has {nch['abstract']:,}. Chunk scores are aggregated to PMID by max. **Decision: whole-abstract chunks**, because the three strategies are indistinguishable here and it is the simplest (details below).
- **Fusion:** BM25 and dense lists fused with RRF (k=60); no weight tuning.
- **Reranker:** local cross-encoder over the whole abstract of the top-20 fused candidates. **Decision: MedCPT** (trained on PubMed search logs). Note: this model runs locally, not through the hosted API; the brief's API-only constraint applies to the generator.
- **Generator:** `Qwen/Qwen3-8B`, structured JSON + 3 answered and 3 refusal few-shot demos, T=0. It sees the top-k full abstracts, each headed `[PMID n]`, and must cite one PMID and copy a verbatim quote.
- **Leakage:** the authors' conclusion is not indexed; demos come from the development split only.

## 3. Retrieval experiments (recall@k, PMID level, 78 answerable items)

Targets (spec 6.2): R@5 at least 0.95 verbatim and 0.85 paraphrase.

| Chunking | Method | verbatim R@1 | verbatim R@5 | paraphrase R@1 | paraphrase R@5 | R@10 (all) | MRR (all) |
|---|---|---|---|---|---|---|---|
{rtable}

**Reading.**
- Almost every configuration meets both targets, so the corpus is **easy**: distractors are random abstracts, not topical neighbours. On the 25-item paraphrase slice one item is 4 points, so most differences between chunking strategies are noise and no chunking strategy should be called better.
- **Dense beats BM25 on paraphrases** (R@1 about 96-100% vs 76-80%). **Plain RRF fusion is worse than dense alone on paraphrases** (R@1 88%) because BM25 drags the fused list down when the wording differs; the reranker then recovers that loss.
- **The MedCPT reranker** reaches 100% R@1 on both slices with whole-abstract chunks. With section or window chunks it gets 96% on paraphrases, where dense alone gets 96-100%, so it is not uniformly better than dense retrieval. It does beat the general MiniLM reranker in every chunking (paraphrase R@1 100% vs 96% for abstract, 96% vs 92% for the others). The query prefix made little difference.

## 4. End-to-end answer quality (all 100 items, frozen eval set)

| Retrieval config | Gold in context | Citation (C2) | Verdict acc | Macro-F1 | Strict acc | Grounded (C3) | Faithful (C4) | Halluc. PMIDs | Over-refusal | Refusal recall | Refusal fully correct | Avg tokens | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{etable}

Week 1 baseline (b), for reference: strict {p(W1['strict_all'])}, verdict {p(W1['verdict_acc_all'])}, citation {p(W1['citation_acc'])}.

## 5. Retrieval quality vs answer quality (measured separately)

| Config | Retrieval recall (abstract chunking) | Gold in generator context | Cited PMID correct | Strict acc |
|---|---|---|---|---|
{septable}

Failure taxonomy (answerable items, strict accuracy; counts out of 78):

| Cause | BM25 top-1 | BM25 top-3 | BM25 + dense top-3 | BM25 + dense + rerank top-3 |
|---|---|---|---|---|
{taxtable}
| verdict accuracy when the gold abstract is in context | {vacc} |

Paired bootstrap (10,000 resamples):

| Comparison | Difference | 95% CI |
|---|---|---|
{bs}

## 6. Findings

1. **Better retrieval raises citation accuracy but not strict accuracy.** The reranker lifts the share of questions whose gold abstract reaches the generator from {p(b1['gold_in_context'])} to {p(best['gold_in_context'])} and citation accuracy by 9.0 points (95% CI +2.6 to +16.7, significant), yet strict accuracy stays flat at about 58-60% (differences between retrieval configs are within 3 points, CIs contain 0).
2. **The verdict step is the bottleneck, as in Week 1.** With the gold abstract in context, the 8B model still gets only {vlo*100:.0f} to {vhi*100:.0f}% of verdicts right; wrong verdicts with the right abstract cited are the largest failure class in every config (20 to 26 of 78). Better retrieval cannot fix that; the Week 3 critic and planner must target it.
3. **Grounding is solid.** No hallucinated PMIDs and no cited PMID outside the retrieved set in any config. The evidence quote is verbatim for 85-89% of the 78 answerable items.
4. **The Week 1 baseline was not the strongest single prompt.** Week 2 with the *same* top-1 BM25 retrieval scores {p(b1['strict_all'])} strict vs Week 1's {p(W1['strict_all'])}, +9.0 points (CI +3.8 to +15.4), purely from a different prompt layout (multi-abstract format with PMID headers, citation key, "use only these abstracts"). **Consequence:** to claim that the agent improves on the baseline, compare against the stronger one, the Week 2 top-1 BM25 prompt ({p(b1['strict_all'])}), which makes the +10 point target about 70% strict accuracy. This needs a team decision before Week 3.
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
"""
(ROOT / "docs/week2_report.md").write_text(md)
print("written", len(md.split()), "words")
