# Week 1 report: spec, frozen eval set and baselines

Track B (API-side), MedEvidence Copilot. Model: `Qwen/Qwen3-8B` behind the shared endpoint. Status: **Week 1 gate met** (eval set frozen, baseline numbers recorded; the agent code has not been written).

## 1. Week 1 deliverables

| Deliverable (course brief) | Where |
|---|---|
| Written task specification: what the copilot does, must refuse, and what a correct answer looks like | `Project Spec.md` (v1.0 draft, team) |
| Evaluation set of 60-100 realistic queries with gold answers, written before any code | `eval/eval_set.jsonl` (100 items), frozen, sha256 `1a5befac61c86ce450dfc84e32f0940789d171f1b622f0a2a89863eb23ba6e56` |
| Prompting baseline: single agent, at least 3 strategies, and how the small model breaks | sections 3 to 5 below; `scripts/run_week1.py`, `results/week1/` |

## 2. Eval set (frozen)

100 items: 45 `verbatim`, 25 `paraphrase`, 8 `boundary` (78 answerable, gold label from the expert-annotated PQA-L **test** split: 40 yes, 27 no, 11 maybe) and 22 `refusal`
(8 personal diagnosis, 7 drug dosing, 7 emergency; 6 disguised). Checks run on import: every gold PMID is in the official test split; labels and rationales match PubMedQA;
no duplicate PMIDs; the build script reproduces the file byte for byte. Few-shot examples are drawn only from the 500-item development split.

## 3. Setup

- **Settings.** (a) closed-book: the model answers from its own knowledge. (b) top-1 BM25: the single best-matching abstract from the 2000-abstract corpus is supplied (citation = that PMID).
  Baseline (b) is the bar the agent must beat (spec 6.2).
- **Strategies (all single-agent).** structured JSON + 3 answered and 3 refusal few-shot demos; zero-shot chain-of-thought (reasoning key first); CoT with self-consistency (5 samples at T=0.7, majority vote); two-step decomposition (analyse and extract quotes, then decide).
- **Decoding.** T=0 (self-consistency: 0.7), JSON mode, thinking off, max 500-700 output tokens. One run, one seed.
- **Scoring.** Spec criteria C1 to C6 (`src/medevidence/metrics.py`). C4 is an LLM judge (the same 8B model, T=0). Strict accuracy = C1 to C6 together, over the 78 answerable items. Closed-book has no citation, so strict accuracy, C2 and C3 do not apply.

## 4. Results

| Setting | Strategy | Verdict acc | Macro-F1 | Strict acc | Citation (C2) | Grounded (C3) | Faithful (C4) | Refusal recall | Refusal fully correct | Over-refusal | Avg tokens | Median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (a) closed-book | structured + few-shot | 44.9% | 0.377 | n/a | n/a | n/a | n/a | 100.0% | 100.0% | 0.0% | 1252 | 2.0s |
| (a) closed-book | chain-of-thought | 32.1% | 0.326 | n/a | n/a | n/a | n/a | 100.0% | 63.6% | 0.0% | 783 | 4.3s |
| (a) closed-book | CoT + self-consistency (n=5) | 29.5% | 0.301 | n/a | n/a | n/a | n/a | 100.0% | 68.2% | 0.0% | 1493 | 5.1s |
| (a) closed-book | decomposition (2 calls) | 50.0% | 0.365 | n/a | n/a | n/a | n/a | 100.0% | 95.5% | 0.0% | 647 | 5.0s |
| (b) BM25 top-1 | structured + few-shot | 57.7% | 0.531 | 51.3% | 88.5% | 82.1% | 88.5% | 90.9% | 86.4% | 0.0% | 1991 | 2.8s |
| (b) BM25 top-1 | chain-of-thought | 52.6% | 0.519 | 46.2% | 88.5% | 75.6% | 82.1% | 100.0% | 18.2% | 1.3% | 1161 | 5.2s |
| (b) BM25 top-1 | CoT + self-consistency (n=5) | 52.6% | 0.519 | 48.7% | 88.5% | 75.6% | 83.3% | 100.0% | 22.7% | 1.3% | 2063 | 6.3s |
| (b) BM25 top-1 | decomposition (2 calls) | 55.1% | 0.511 | 50.0% | 88.5% | 87.2% | 78.2% | 95.5% | 63.6% | 1.3% | 1066 | 6.1s |

**Baseline to beat.** The best baseline (b) by strict accuracy is structured + few-shot at **51.3%** (40/78). Spec 6.2 asks the agent for at least +10 points, so the target is
**about 61% strict accuracy**. Safety gate for release (spec 6.3): all 7 emergencies refused **with an emergency redirect**, and zero hallucinated PMIDs. Baseline (b) refuses 7/7 emergencies in every strategy, but not always with the emergency category (see section 5, point 6), so the gate is only fully met by closed-book few-shot. Hallucinated PMIDs are 0 by construction, because the citation comes from retrieval.

Verdict accuracy by slice:

| Config | verbatim | paraphrase | boundary |
|---|---|---|---|
| closed / fewshot | 55.6% | 36.0% | 12.5% |
| closed / cot | 42.2% | 24.0% | 0.0% |
| closed / cot_sc | 33.3% | 32.0% | 0.0% |
| closed / decompose | 53.3% | 52.0% | 25.0% |
| bm25 / fewshot | 66.7% | 44.0% | 50.0% |
| bm25 / cot | 57.8% | 48.0% | 37.5% |
| bm25 / cot_sc | 60.0% | 44.0% | 37.5% |
| bm25 / decompose | 62.2% | 44.0% | 50.0% |

Paired bootstrap (10,000 resamples, 78 items) of the difference in verdict accuracy, plus strict accuracy for the BM25 strategies:

| Comparison | Difference | 95% CI |
|---|---|---|
| bm25__fewshot - closed__fewshot | +12.8 pts | [+0.0, +25.6] |
| bm25__fewshot - bm25__cot | +5.1 pts | [-5.1, +15.4] |
| bm25__fewshot - bm25__decompose | +2.6 pts | [-6.4, +11.5] |
| bm25__decompose - closed__decompose | +5.1 pts | [-7.7, +17.9] |
| closed__decompose - closed__fewshot | +5.1 pts | [-6.4, +16.7] |
| STRICT bm25__fewshot - bm25__cot | +5.1 pts | [-3.8, +14.1] |
| STRICT bm25__fewshot - bm25__decompose | +1.3 pts | [-9.0, +11.5] |

**Reading.** Supplying a retrieved abstract helps (+12.8 points, but the CI just touches 0). **No difference between prompting strategies is statistically distinguishable at n=78**: every CI contains 0.
With 78 items one item is 1.3 points, so strategy rankings should not be over-read. Self-consistency doubled the token cost and bought nothing.

## 5. How the 8B model breaks (failure characterisation)

1. **The verdict is the main failure, not retrieval.** For the best baseline, 38 of 78 items fail strict accuracy: **25 wrong verdicts with the correct abstract in context**, 9 retrieval misses, 4 quotes not verbatim.
   Even when the gold abstract is retrieved, verdict accuracy is only 63.8% (69 items).
2. **It cannot say "no".** Gold `no` items are recovered only 10 of 27 times (closed-book few-shot: 4 of 27). The typical error is `no` -> `maybe`: the model finds the study's own hedging ("but one patient developed a CLABSI, suggesting...", Q015, Q032) and answers `maybe`, while the authors concluded `no`. Gold `maybe` is mixed up with `yes` in 5 of 11 cases. Confusion matrices for all configs are in `results/week1/analysis.json`.
3. **Closed-book is poorly calibrated.** Few-shot closed-book predicts 40 yes / 4 no / 34 maybe on a gold mix of 40 / 27 / 11. Decomposition closed-book predicts `yes` 56 times. CoT closed-book declines 9 items as `insufficient_evidence`, because no abstract is supplied.
4. **BM25 top-1 misses mostly paraphrases.** 9 of 78 items (6 paraphrase, 3 verbatim). The model usually notices: it answers `insufficient_evidence` ("the abstract discusses QTcD and left ventricular hypertrophy, not PSA cut-offs", Q021) in 7 of those 9. That is the right behaviour, but it costs the item. Retrieval recall@1 of the baseline is 88.5% overall.
5. **Grounding.** The evidence quote is not verbatim in the abstract in 2 to 5 of 78 cases depending on the strategy (paraphrased or merged sentences). The 8B model's explanations are judged faithful for 78 to 89% of the 78 answerable items (judge is the same model, see limitations).
6. **Refusal decisions are good, refusal contracts are not.** Refusal recall is 100% for most configs. Failures: (i) in the BM25 setting, the **emergency items are labelled `personal_diagnosis` instead of `emergency`** (6 of 7 for decomposition, 1 of 7 for few-shot), so the redirect would say "see your clinician" instead of "call emergency services": still refused, wrong advice; (ii) two disguised diagnosis probes were answered by few-shot (Q085 "Studies show IBS is a diagnosis of exclusion... do I have IBS?", Q086 a PSA rise); (iii) the CoT prompt makes the model copy the schema placeholder ("refusal text: personal diagnosis") into the message field, so **16 of 22 refusals lack the redirect** the spec requires. The decision is right; the message is not.
7. **Over-refusal is rare.** At most 1 of 78 answerable items (CoT, self-consistency and decomposition with BM25); no boundary item was ever refused, in any config.

## 6. Findings that change the Week 3 design

- A **code-level safety gate** (spec: runs before retrieval and the planner) is justified: it should output **templated refusal messages chosen by code** and use a dedicated emergency check, rather than trust a free-text message from the model.
- The agent's gain will come from the **verdict step** (`no` vs `maybe`): the critic should check the verdict against the abstract's conclusion sentence, and the planner should prefer abstracts whose conclusion is explicit. Retrieval is secondary: dense/hybrid retrieval already reaches R@5 of 96 to 100% (Week 2).
- Use few-shot structured output as the executor prompt (cheapest, no worse than the others). Do not use self-consistency.

## 7. Limitations and corrections

- **Judge.** C4 uses the same 8B model, so it can share the generator's biases. A 20% sample (16 items) is exported to `results/week1/judge_spotcheck.csv` for a human to fill in; **this has not been done yet**, so C4 numbers are unverified.
- **Sample size.** 78 answerable items, one run, one seed: differences under about 8 points are within noise (see CIs).
- **Corrected pipeline bug (documented, original kept).** The first decomposition run only refused when the model set `in_scope=false`, but the model often set `in_scope=true` together with `refusal_category="emergency"`. That run refused only 2 of 7 emergencies and had refusal recall 22.7% (closed-book). Routing on `refusal_category` fixed it (recall 100.0%, 7/7 emergencies). The original rows are kept in `results/week1/v1_inscope_gate/`. The fix was made after seeing the results, so decomposition is the only strategy tuned against the eval set.
- **Refusal probes** are hand-written (22) and were the only refusal data. The few-shot refusal demos are fresh examples, not eval items.
- The closed-book setting has no citation; strict accuracy is not defined for it.
- Titles are not in PubMedQA, so the answer contract's `citation.title` is not produced.

## 8. Reproduce

```bash
python scripts/build_eval_set.py          # eval/eval_set.jsonl (deterministic)
python scripts/build_corpus.py            # data/corpus.jsonl
python scripts/run_week1.py               # 8 configs, writes results/week1/
python scripts/analyze_week1.py           # confusion, taxonomy, bootstrap, spot-check sample
python scripts/make_week1_report.py       # this file
```
