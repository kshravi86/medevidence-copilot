"""Generate docs/week1_report.md: tables come straight from results/week1/*.json (no hand-copied numbers)."""
import json, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
S = json.load(open(ROOT / "results/week1/summary.json"))
A = json.load(open(ROOT / "results/week1/analysis.json"))
V1 = json.load(open(ROOT / "results/week1/v1_inscope_gate/summary.json"))
sha = hashlib.sha256(open(ROOT / "eval/eval_set.jsonl", "rb").read()).hexdigest()
p = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"
order = ["closed__fewshot", "closed__cot", "closed__cot_sc", "closed__decompose", "bm25__fewshot", "bm25__cot", "bm25__cot_sc", "bm25__decompose"]
label = {"fewshot": "structured + few-shot", "cot": "chain-of-thought", "cot_sc": "CoT + self-consistency (n=5)", "decompose": "decomposition (2 calls)"}

def row(n):
    s = S[n]; st, strat = n.split("__")
    return (f"| {'(a) closed-book' if st == 'closed' else '(b) BM25 top-1'} | {label[strat]} | {p(s['verdict_acc_all'])} | {s['macro_f1']:.3f} | {p(s['strict_all'])} | "
            f"{p(s['citation_acc'])} | {p(s['grounded_acc'])} | {p(s['faithful_acc'])} | {p(s['refusal_recall'])} | {p(s['refusal_correct'])} | {p(s['over_refusal_rate'])} | "
            f"{s['avg_tokens']:.0f} | {s['median_latency_s']:.1f}s |")

main = "\n".join(row(n) for n in order)
sl = "\n".join(f"| {n.replace('__', ' / ')} | {p(S[n]['verdict_acc_verbatim'])} | {p(S[n]['verdict_acc_paraphrase'])} | {p(S[n]['verdict_acc_boundary'])} |" for n in order)
bs = "\n".join(f"| {k} | {v['diff']*100:+.1f} pts | [{v['ci95'][0]*100:+.1f}, {v['ci95'][1]*100:+.1f}] |" for k, v in A["bootstrap"].items())
tax = A["taxonomy"]["bm25__fewshot"]
best = S["bm25__fewshot"]
target = best["strict_all"] + 0.10

md = f"""# Week 1 report: spec, frozen eval set and baselines

Track B (API-side), MedEvidence Copilot. Model: `Qwen/Qwen3-8B` behind the shared endpoint. Status: **Week 1 gate met** (eval set frozen, baseline numbers recorded; the agent code has not been written).

## 1. Week 1 deliverables

| Deliverable (course brief) | Where |
|---|---|
| Written task specification: what the copilot does, must refuse, and what a correct answer looks like | `Project Spec.md` (v1.0 draft, team) |
| Evaluation set of 60-100 realistic queries with gold answers, written before any code | `eval/eval_set.jsonl` (100 items), frozen, sha256 `{sha}` |
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
{main}

**Baseline to beat.** The best baseline (b) by strict accuracy is structured + few-shot at **{p(best['strict_all'])}** (40/78). Spec 6.2 asks the agent for at least +10 points, so the target is
**about {target*100:.0f}% strict accuracy**. Safety gate for release (spec 6.3): all 7 emergencies refused **with an emergency redirect**, and zero hallucinated PMIDs. Baseline (b) refuses 7/7 emergencies in every strategy, but not always with the emergency category (see section 5, point 6), so the gate is only fully met by closed-book few-shot. Hallucinated PMIDs are 0 by construction, because the citation comes from retrieval.

Verdict accuracy by slice:

| Config | verbatim | paraphrase | boundary |
|---|---|---|---|
{sl}

Paired bootstrap (10,000 resamples, 78 items) of the difference in verdict accuracy, plus strict accuracy for the BM25 strategies:

| Comparison | Difference | 95% CI |
|---|---|---|
{bs}

**Reading.** Supplying a retrieved abstract helps (+12.8 points, but the CI just touches 0). **No difference between prompting strategies is statistically distinguishable at n=78**: every CI contains 0.
With 78 items one item is 1.3 points, so strategy rankings should not be over-read. Self-consistency doubled the token cost and bought nothing.

## 5. How the 8B model breaks (failure characterisation)

1. **The verdict is the main failure, not retrieval.** For the best baseline, 38 of 78 items fail strict accuracy: **{tax['wrong verdict (gold abstract retrieved)']} wrong verdicts with the correct abstract in context**, {tax['retrieval miss (top-1 not gold)']} retrieval misses, {tax['quote not verbatim in abstract']} quotes not verbatim.
   Even when the gold abstract is retrieved, verdict accuracy is only {p(tax['_verdict_acc_when_top1_hit'])} ({tax['_n_hit_miss'][0]} items).
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
- **Corrected pipeline bug (documented, original kept).** The first decomposition run only refused when the model set `in_scope=false`, but the model often set `in_scope=true` together with `refusal_category="emergency"`. That run refused only 2 of 7 emergencies and had refusal recall {p(V1['closed__decompose']['refusal_recall'])} (closed-book). Routing on `refusal_category` fixed it (recall {p(S['closed__decompose']['refusal_recall'])}, 7/7 emergencies). The original rows are kept in `results/week1/v1_inscope_gate/`. The fix was made after seeing the results, so decomposition is the only strategy tuned against the eval set.
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
"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/week1_report.md").write_text(md)
print("written", len(md.split()), "words")
