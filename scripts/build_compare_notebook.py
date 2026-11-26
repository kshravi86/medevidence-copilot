"""Builds notebooks/MedEvidence_Copilot_Week2_vs_Week3.ipynb (analysis only, no model calls)."""
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# MedEvidence Copilot: Week 2 vs Week 3

This notebook compares the **Week 2 RAG pipelines** with the **Week 3 agent** on the same frozen 100-question evaluation set. It only reads saved result files, so it makes **no model calls and uses no API quota**. It runs on a CPU runtime in about a minute.

| Section | What it shows |
|---|---|
| **1. Headline table** | Every metric for every configuration, side by side |
| **2. Is the gain real?** | Paired bootstrap 95% CI and an exact McNemar test, Week 3 vs each Week 2 run |
| **3. By question type** | Verbatim, paraphrase and boundary questions; refusals by category |
| **4. Which questions changed** | Questions Week 3 fixed and questions it broke |
| **5. Confusion matrices** | Gold label vs prediction, Week 2 vs Week 3 |
| **6. Failure taxonomy** | One taxonomy applied to both weeks, so the counts are comparable |
| **7. Cost and speed** | Tokens, latency, model and tool calls |
| **8. Save** | Tables, charts and a short write-up to a dated folder in Drive |

All metrics come from the project's own `metrics.summarize`, the same function the Week 2 and Week 3 scripts used, so the numbers match the reports.
""")

md("## Settings")
code(r'''
# "committed": the result files in the code bundle (what the reports use).
# "latest_drive_run": the newest Week 3 run saved by the Week 3-4 notebook in Drive (runs/week3-4_*), Week 2 from the bundle.
RESULTS_SOURCE = "committed"

# The Week 2 run used for the question-by-question comparison (sections 4 and 5).
# "W2 BM25 top-1" had the best Week 2 strict score, so it is the hardest bar to beat.
MAIN_WEEK2 = "W2 BM25 top-1"

BOOTSTRAP_N = 10000          # resamples for the paired bootstrap
SAVE_TO_DRIVE = True         # write tables, charts and the write-up to Drive (section 8)

BRANCH = "aakash_r_changes"
DRIVE_FOLDER = "medevidence-copilot-colab"
BUNDLE_NAME = "medevidence-copilot-week3-4.bundle"
''')

md(r"""
## Setup
### Connect Google Drive and get the code
The code bundle in Drive holds the frozen evaluation set and the saved results of every week. No keys or certificates are needed.
""")
code(r'''
import json, math, os, subprocess, sys, time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE = Path("/content/drive/MyDrive") / DRIVE_FOLDER
    WORK = Path("/content")
else:  # running outside Colab, for example to test the notebook
    DRIVE = Path(os.environ["MEDEV_DRIVE"])
    WORK = Path(os.environ["MEDEV_WORK"])

REPO = WORK / "medevidence-copilot"
BUNDLE = DRIVE / "code" / BUNDLE_NAME
if not REPO.exists():
    if not BUNDLE.exists():
        raise RuntimeError(f"Missing {BUNDLE}")
    subprocess.run(["git", "clone", "-q", "--branch", BRANCH, str(BUNDLE), str(REPO)], check=True)
print("Code version:", subprocess.check_output(["git", "-C", str(REPO), "log", "-1", "--format=%h  %s"], text=True).strip())

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-dotenv"], check=True)  # needed by the project config
sys.path.insert(0, str(REPO / "src"))
from medevidence import data, metrics

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"
''')

md(r"""
### Load the results
Six runs, all on the same 100 questions (78 to answer, 22 to refuse):

| Label | What it is | File |
|---|---|---|
| W1 baseline b | Week 1 spec baseline: BM25 top-1 + few-shot prompt | `results/week1/bm25__fewshot.jsonl` |
| W2 BM25 top-1 | Week 2 RAG, BM25, 1 abstract in context | `results/week2/rag__abstract__bm25_1.jsonl` |
| W2 BM25 top-3 | Week 2 RAG, BM25, 3 abstracts | `results/week2/rag__abstract__bm25_3.jsonl` |
| W2 hybrid top-3 | Week 2 RAG, BM25 + dense, 3 abstracts | `results/week2/rag__abstract__hybrid_3.jsonl` |
| W2 rerank top-3 | Week 2 RAG, hybrid + MedCPT rerank, 3 abstracts | `results/week2/rag__abstract__rerank_3.jsonl` |
| W3 agent | Week 3 agent: safety gate, planner, executor, critic | `results/week3/agent.jsonl` |
""")
code(r'''
FILES = {
    "W1 baseline b": REPO / "results/week1/bm25__fewshot.jsonl",
    "W2 BM25 top-1": REPO / "results/week2/rag__abstract__bm25_1.jsonl",
    "W2 BM25 top-3": REPO / "results/week2/rag__abstract__bm25_3.jsonl",
    "W2 hybrid top-3": REPO / "results/week2/rag__abstract__hybrid_3.jsonl",
    "W2 rerank top-3": REPO / "results/week2/rag__abstract__rerank_3.jsonl",
    "W3 agent": REPO / "results/week3/agent.jsonl",
}
if RESULTS_SOURCE == "latest_drive_run":
    runs = sorted((DRIVE / "runs").glob("week3-4_*/results/week3/agent.jsonl"))
    if not runs:
        raise RuntimeError("No Week 3 run in Drive yet: run the Week 3-4 notebook with RUN_FULL_WEEK3 = True, "
                           "or set RESULTS_SOURCE = 'committed'")
    FILES["W3 agent"] = runs[-1]
elif RESULTS_SOURCE != "committed":
    raise ValueError("RESULTS_SOURCE must be 'committed' or 'latest_drive_run'")

EVAL = data.load_eval()
ANS = [i for i in EVAL if i["expected_behavior"] == "answer"]
REF = [i for i in EVAL if i["expected_behavior"] == "refuse"]
ITEM = {i["id"]: i for i in EVAL}

ROWS = {}
for label, path in FILES.items():
    rows = [json.loads(line) for line in open(path)]
    ids = {r["id"] for r in rows}
    if ids != set(ITEM):
        raise RuntimeError(f"{label}: {path.name} does not cover the 100 evaluation questions")
    ROWS[label] = {r["id"]: r for r in rows}
    print(f"{label:16s} {len(rows)} rows  <- {path}")
if MAIN_WEEK2 not in ROWS or not MAIN_WEEK2.startswith("W2"):
    raise ValueError(f"MAIN_WEEK2 must be one of {[k for k in ROWS if k.startswith('W2')]}")
print(f"\n{len(ANS)} questions to answer, {len(REF)} to refuse")
''')

md(r"""
### Check the numbers against the saved summaries
Recomputes every metric with `metrics.summarize` and checks the strict score against `summary.json`, so this notebook and the reports agree.
""")
code(r'''
SUMMARY = {label: metrics.summarize(list(rows.values()), EVAL) for label, rows in ROWS.items()}

saved = {}
for week in ("week1", "week2", "week3"):
    p = REPO / "results" / week / "summary.json"
    if p.exists():
        saved.update({f"{week}:{k}": v for k, v in json.load(open(p)).items()})
KEYS = {"W1 baseline b": "week1:bm25__fewshot", "W2 BM25 top-1": "week2:rag__abstract__bm25_1",
        "W2 BM25 top-3": "week2:rag__abstract__bm25_3", "W2 hybrid top-3": "week2:rag__abstract__hybrid_3",
        "W2 rerank top-3": "week2:rag__abstract__rerank_3", "W3 agent": "week3:agent"}
for label, key in KEYS.items():
    ours, theirs = SUMMARY[label]["strict_all"], (saved.get(key) or {}).get("strict_all")
    if theirs is None:
        print(f"{label:16s} strict {pct(ours)}   (no saved summary to compare)")
    else:
        status = "matches" if abs(ours - theirs) < 1e-9 else "DIFFERS from"
        print(f"{label:16s} strict {pct(ours)}   {status} summary.json ({pct(theirs)})")
''')

md(r"""
---
## 1. Headline table
**Strict** is the main score: the verdict is right **and** the cited PMID is right **and** the quote is word for word **and** the explanation is faithful **and** the format is valid **and** the question was not wrongly refused. Answer metrics are over the 78 answerable questions; refusal metrics over the 22 that must be refused.
""")
code(r'''
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def tokens(r):
    if r.get("tokens") is not None:
        return r["tokens"]
    if r.get("prompt_tokens") is not None:
        return r["prompt_tokens"] + (r.get("completion_tokens") or 0)
    return None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def headline(label):
    s, rows = SUMMARY[label], ROWS[label]
    gic = [rows[i["id"]].get("gold_in_context") for i in ANS]
    return {
        "Strict (main score)": s["strict_all"],
        "Verdict accuracy": s["verdict_acc_all"],
        "Macro F1 (yes/no/maybe)": s["macro_f1"],
        "Citation correct": s["citation_acc"],
        "Quote word for word": s["grounded_acc"],
        "Explanation faithful": s["faithful_acc"],
        "Gold abstract in context": mean(gic) if any(g is not None for g in gic) else None,
        "Refusal recall": s["refusal_recall"],
        "Refusal, right category": s["refusal_correct"],
        "Over-refusal rate": s["over_refusal_rate"],
        "Hallucinated PMIDs": s["hallucinated_pmids"],
        "Emergencies refused": s["emergency_refused"],
        "Tokens per question (avg)": mean(tokens(r) for r in rows.values()),
        "Latency per question (median s)": median(r.get("latency") for r in rows.values()),
    }


HEAD = pd.DataFrame({label: headline(label) for label in ROWS})
COUNT_ROWS = {"Hallucinated PMIDs", "Emergencies refused", "Tokens per question (avg)", "Latency per question (median s)"}


def fmt(row_name, v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if row_name == "Tokens per question (avg)":
        return f"{v:,.0f}"
    if row_name == "Latency per question (median s)":
        return f"{v:.1f}"
    if row_name in COUNT_ROWS:
        return str(v)
    return pct(v)


display(HEAD.apply(lambda col: [fmt(name, v) for name, v in col.items()]).set_axis(HEAD.index))
''')

code(r'''
fig, ax = plt.subplots(figsize=(11, 4.2))
labels = list(ROWS)
x = range(len(labels))
for k, (metric, color) in enumerate((("Strict (main score)", "#1f6feb"), ("Verdict accuracy", "#8fb8ff"),
                                     ("Citation correct", "#f5b971"))):
    vals = [HEAD.loc[metric, l] * 100 for l in labels]
    bars = ax.bar([i + (k - 1) * 0.26 for i in x], vals, width=0.26, label=metric, color=color)
    ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
ax.set_xticks(list(x), labels, rotation=0, fontsize=9)
ax.set_ylabel("% of 78 answerable questions")
ax.set_ylim(0, 105)
ax.set_title("Week 1 baseline, Week 2 RAG runs and the Week 3 agent")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
FIG_HEADLINE = fig
plt.show()
''')

md(r"""
---
## 2. Is the gain real?
Each comparison is **paired**: the same 78 questions, Week 3 vs one earlier run.

- **Paired bootstrap**: resamples the 78 questions `BOOTSTRAP_N` times. If the 95% CI of the difference is entirely above 0, the gain is unlikely to be luck.
- **Exact McNemar test**: looks only at the questions where the two runs disagree (Week 3 right and the other wrong, or the reverse).
- **Wins / losses**: how many questions Week 3 gets right that the other run got wrong, and the reverse.
""")
code(r'''
def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a_label, b_label, key):
    ids = [i["id"] for i in ANS]
    a = [int(bool(ROWS[a_label][q].get(key))) for q in ids]
    b = [int(bool(ROWS[b_label][q].get(key))) for q in ids]
    diff, (lo, hi) = metrics.paired_bootstrap(a, b, n=BOOTSTRAP_N)
    wins = sum(x and not y for x, y in zip(a, b))
    losses = sum(y and not x for x, y in zip(a, b))
    return {"vs": b_label, "metric": key, "W3": sum(a) / len(a), "other": sum(b) / len(b), "diff": diff,
            "ci_low": lo, "ci_high": hi, "wins": wins, "losses": losses, "mcnemar_p": mcnemar_exact(wins, losses),
            "CI above 0": lo > 0}


t0 = time.time()
SIG = pd.DataFrame([compare("W3 agent", other, key)
                    for other in ROWS if other != "W3 agent"
                    for key in ("strict", "c1_verdict", "c2_citation")])
print(f"{BOOTSTRAP_N} resamples per comparison, {time.time() - t0:.0f} s")
names = {"strict": "Strict", "c1_verdict": "Verdict", "c2_citation": "Citation"}
display(pd.DataFrame({
    "Week 3 vs": SIG["vs"], "Metric": SIG["metric"].map(names),
    "Week 3": SIG["W3"].map(pct), "Other": SIG["other"].map(pct),
    "Difference": SIG["diff"].map(lambda d: f"{d * 100:+.1f} pts"),
    "95% CI": [f"[{lo * 100:+.1f}, {hi * 100:+.1f}]" for lo, hi in zip(SIG["ci_low"], SIG["ci_high"])],
    "Wins / losses": [f"{w} / {l}" for w, l in zip(SIG["wins"], SIG["losses"])],
    "McNemar p": SIG["mcnemar_p"].map(lambda p: f"{p:.3f}"),
    "Clear gain?": SIG["CI above 0"].map({True: "yes", False: "not proven"}),
}))
''')

md(r"""
---
## 3. By question type
- **Verbatim**: the question is close to the PubMedQA wording.
- **Paraphrase**: reworded in lay language.
- **Boundary**: research questions that mention dosing, symptoms or emergencies and must **not** be refused.
""")
code(r'''
SLICES = pd.DataFrame({label: {
    f"{name}: strict": SUMMARY[label][f"strict_{name}"] for name in ("verbatim", "paraphrase", "boundary")
} | {
    f"{name}: verdict": SUMMARY[label][f"verdict_acc_{name}"] for name in ("verbatim", "paraphrase", "boundary")
} for label in ROWS})
n = Counter(i["query_type"] for i in ANS)
print("Questions per type:", dict(n))
display(SLICES.sort_index().apply(lambda col: col.map(pct)))

cats = sorted({i["refusal_category"] for i in REF})
REFUSALS = pd.DataFrame({label: {
    f"{c} ({sum(1 for i in REF if i['refusal_category'] == c)})":
        f"{sum(ROWS[label][i['id']]['refused'] for i in REF if i['refusal_category'] == c)} refused, "
        f"{sum(ROWS[label][i['id']]['correct'] for i in REF if i['refusal_category'] == c)} right category"
    for c in cats} for label in ROWS})
print("\nMust-refuse questions by category (count in brackets):")
display(REFUSALS)
''')

md(r"""
---
## 4. Which questions changed
Question by question, Week 3 vs `MAIN_WEEK2` on the strict score. **Fixed**: Week 2 failed, Week 3 passes. **Broken**: Week 2 passed, Week 3 fails. Refusal questions are compared on "refused with the right category".
""")
code(r'''
def outcome(r):
    if r["status"] == "answered":
        return f"{r.get('verdict')} (PMID {r.get('cited_pmid')})"
    return r["status"]


def passed(label, q):
    r = ROWS[label][q]
    return bool(r.get("correct")) if ITEM[q]["expected_behavior"] == "refuse" else bool(r.get("strict"))


def changes(kind):
    out = []
    for q in ITEM:
        w2, w3 = passed(MAIN_WEEK2, q), passed("W3 agent", q)
        if (kind == "fixed" and not w2 and w3) or (kind == "broken" and w2 and not w3):
            it, a, b = ITEM[q], ROWS[MAIN_WEEK2][q], ROWS["W3 agent"][q]
            out.append({"id": q, "type": it["query_type"],
                        "gold": it["gold_label"] or f"refuse: {it['refusal_category']}",
                        "gold PMID": it.get("gold_pmid"), MAIN_WEEK2: outcome(a), "W3 agent": outcome(b),
                        "question": it["query"]})
    return pd.DataFrame(out)


FIXED, BROKEN = changes("fixed"), changes("broken")
both = sum(passed(MAIN_WEEK2, q) and passed("W3 agent", q) for q in ITEM)
neither = sum(not passed(MAIN_WEEK2, q) and not passed("W3 agent", q) for q in ITEM)
print(f"Week 3 vs {MAIN_WEEK2}, all 100 questions: fixed {len(FIXED)}, broken {len(BROKEN)}, "
      f"both pass {both}, both fail {neither}")
display(Markdown(f"**Fixed by Week 3 ({len(FIXED)})**"))
display(FIXED)
display(Markdown(f"**Broken by Week 3 ({len(BROKEN)})**"))
display(BROKEN)
''')

md(r"""
---
## 5. Confusion matrices
Rows are the gold answer, columns what the system returned, over the 78 answerable questions. `refused` and `insufficient` mean it gave no verdict.
""")
code(r'''
COLS = ["yes", "no", "maybe", "insufficient", "refused"]


def predicted(r):
    if r["status"] == "answered":
        return r.get("verdict") or "insufficient"
    return {"insufficient_evidence": "insufficient"}.get(r["status"], r["status"])


def confusion(label):
    m = pd.DataFrame(0, index=["yes", "no", "maybe"], columns=COLS)
    for i in ANS:
        p = predicted(ROWS[label][i["id"]])
        m.loc[i["gold_label"], p if p in COLS else "insufficient"] += 1
    m.index = [f"gold {g} ({int(m.loc[g].sum())})" for g in m.index]
    return m


fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, label in zip(axes, (MAIN_WEEK2, "W3 agent")):
    m = confusion(label)
    display(Markdown(f"**{label}**"))
    display(m)
    ax.imshow(m.values, cmap="Blues")
    ax.set_xticks(range(len(COLS)), COLS, fontsize=9)
    ax.set_yticks(range(3), m.index, fontsize=9)
    for (r, c), v in pd.DataFrame(m.values).stack().items():
        ax.text(c, r, v, ha="center", va="center", fontsize=10, color="white" if v > m.values.max() / 2 else "black")
    ax.set_title(label)
plt.tight_layout()
FIG_CONFUSION = fig
plt.show()
''')

md(r"""
---
## 6. Failure taxonomy
Week 2 and Week 3 saved their failure reasons under different names, so this applies **one** set of rules to every run. Each failed answerable question gets the **first** reason that applies, in this order:

1. **Wrongly refused**: an answerable question was refused (over-refusal).
2. **Retrieval miss**: the gold abstract never reached the model.
3. **No answer despite evidence**: the gold abstract was in context but the system said "insufficient evidence".
4. **Wrong study cited**: it answered but cited another PMID (a hallucinated PMID is counted here too).
5. **Wrong verdict**: right study, wrong yes/no/maybe. This is a reasoning error.
6. **Quote not word for word** and 7. **Unfaithful explanation**: right verdict and study, but the grounding checks failed.
8. **Format**: invalid output.

Failed must-refuse questions are split into **not refused** and **refused, wrong category**.
""")
code(r'''
def reason(q, r):
    it = ITEM[q]
    if it["expected_behavior"] == "refuse":
        if r.get("correct"):
            return None
        return "Refusal: not refused" if not r.get("refused") else "Refusal: wrong category"
    if r.get("strict"):
        return None
    if not r.get("c6_no_overrefusal", True):
        return "1 Wrongly refused"
    if r.get("gold_in_context") is False:
        return "2 Retrieval miss"
    if r["status"] != "answered":
        return "3 No answer despite evidence"
    if not r.get("c2_citation"):
        return "4 Wrong study cited"
    if not r.get("c1_verdict"):
        return "5 Wrong verdict"
    if not r.get("c3_grounded"):
        return "6 Quote not word for word"
    if r.get("c4_faithful") is False:
        return "7 Unfaithful explanation"
    return "8 Format"


TAX = pd.DataFrame({label: Counter(x for x in (reason(q, r) for q, r in rows.items()) if x is not None)
                    for label, rows in ROWS.items()}).fillna(0).astype(int).sort_index()
TAX.loc["Total failures"] = TAX.sum()
TAX.loc["Passed (of 100)"] = [sum(passed(l, q) for q in ITEM) for l in TAX.columns]
display(TAX)
print("W1 baseline b had no retriever log, so its misses with the gold abstract absent are counted under later reasons.")

fail_rows = [i for i in TAX.index if i[0].isdigit() or i.startswith("Refusal")]
fig, ax = plt.subplots(figsize=(11, 4))
left = [0] * len(TAX.columns)
palette = plt.cm.tab10.colors
for k, reason_name in enumerate(fail_rows):
    vals = TAX.loc[reason_name].tolist()
    ax.barh(list(TAX.columns), vals, left=left, label=reason_name, color=palette[k % 10])
    left = [a + b for a, b in zip(left, vals)]
ax.invert_yaxis()
ax.set_xlabel("failed questions (of 100)")
ax.set_title("Why questions fail, same rules for every run")
ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
FIG_TAXONOMY = fig
plt.show()
''')

md(r"""
---
## 7. Cost and speed
Week 3 makes more model calls per question (safety check, planner, executor, critic, sometimes revisions and a re-plan), so it uses more tokens and takes longer. Latency is from the original runs on the course endpoint, and saved answers from the cache make some calls faster, so read it as a rough guide.
""")
code(r'''
def p90(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[int(0.9 * (len(xs) - 1))] if xs else None


COST = pd.DataFrame({label: {
    "Tokens per question (avg)": mean(tokens(r) for r in rows.values()),
    "Latency median (s)": median(r.get("latency") for r in rows.values()),
    "Latency p90 (s)": p90(r.get("latency") for r in rows.values()),
    "Model calls per question (avg)": mean(r.get("llm_calls") for r in rows.values()),
    "Tool calls per question (avg)": mean(r.get("tool_calls") for r in rows.values()),
} for label, rows in ROWS.items()})
display(COST.apply(lambda col: col.map(lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.1f}")))
print("Model and tool calls are only logged for the Week 3 agent; Week 1 and 2 make one model call per question.")

w3 = ROWS["W3 agent"]
crit = [r.get("critic") or {} for r in w3.values() if r["status"] == "answered"]
display(Markdown(
    f"**Week 3 critic**: approved {sum(1 for c in crit if c.get('approved') and c.get('revisions') == 0)} of "
    f"{len(crit)} first drafts, sent back {sum(1 for c in crit if (c.get('revisions') or 0) > 0)}, "
    f"downgraded {sum(1 for c in crit if c.get('downgraded'))} to maybe. "
    f"**Re-plans**: {sum(1 for r in w3.values() if (r.get('plan') or {}).get('replanned_queries') is not None)}. "
    f"**JSON repairs**: {sum(r.get('json_repairs') or 0 for r in w3.values())}. "
    f"**Tool errors**: {sum(len(r.get('tool_errors') or []) for r in w3.values())}."))
''')

md(r"""
---
## 8. Save
Writes the tables (CSV), the three charts (PNG) and a short write-up (`summary.md`) to `runs/week2_vs_week3_<date>` in Drive.
""")
code(r'''
best_w2 = max((l for l in ROWS if l.startswith("W2")), key=lambda l: SUMMARY[l]["strict_all"])
row = SIG[(SIG["vs"] == best_w2) & (SIG["metric"] == "strict")].iloc[0]
row_w1 = SIG[(SIG["vs"] == "W1 baseline b") & (SIG["metric"] == "strict")].iloc[0]
w3s, w2s = SUMMARY["W3 agent"], SUMMARY[best_w2]
lines = [
    "# Week 2 vs Week 3: results",
    "",
    f"- Strict score: Week 3 agent {pct(w3s['strict_all'])} vs best Week 2 run ({best_w2}) {pct(w2s['strict_all'])}: "
    f"{row['diff'] * 100:+.1f} points, 95% CI [{row['ci_low'] * 100:+.1f}, {row['ci_high'] * 100:+.1f}], "
    f"{int(row['wins'])} wins / {int(row['losses'])} losses, McNemar p = {row['mcnemar_p']:.3f}"
    + (" (clear gain)." if row["CI above 0"] else " (gain not proven at 95%)."),
    f"- Against the Week 1 spec baseline b: {row_w1['diff'] * 100:+.1f} points, 95% CI "
    f"[{row_w1['ci_low'] * 100:+.1f}, {row_w1['ci_high'] * 100:+.1f}]"
    + (" (clear gain)." if row_w1["CI above 0"] else " (not proven)."),
    f"- Refusal recall: {pct(w2s['refusal_recall'])} -> {pct(w3s['refusal_recall'])}; "
    f"right category {pct(w2s['refusal_correct'])} -> {pct(w3s['refusal_correct'])}; "
    f"emergencies refused {w3s['emergency_refused']}.",
    f"- Citation correct: {pct(w2s['citation_acc'])} -> {pct(w3s['citation_acc'])}; "
    f"gold abstract in context: {pct(HEAD.loc['Gold abstract in context', best_w2])} -> "
    f"{pct(HEAD.loc['Gold abstract in context', 'W3 agent'])}.",
    f"- Hallucinated PMIDs: {w2s['hallucinated_pmids']} -> {w3s['hallucinated_pmids']}.",
    f"- Biggest remaining failure in Week 3: {TAX.loc[[i for i in TAX.index if i[0].isdigit()], 'W3 agent'].idxmax()[2:]} "
    f"({TAX.loc[[i for i in TAX.index if i[0].isdigit()], 'W3 agent'].max()} questions).",
    f"- Cost: about {HEAD.loc['Tokens per question (avg)', 'W3 agent']:,.0f} tokens and a median "
    f"{HEAD.loc['Latency per question (median s)', 'W3 agent']:.1f} s per question, vs "
    f"{HEAD.loc['Tokens per question (avg)', best_w2]:,.0f} tokens and {HEAD.loc['Latency per question (median s)', best_w2]:.1f} s.",
    "",
    f"Week 3 results: {FILES['W3 agent']}",
]
SUMMARY_MD = "\n".join(lines)
display(Markdown(SUMMARY_MD))

if SAVE_TO_DRIVE:
    out = DRIVE / "runs" / f"week2_vs_week3_{time.strftime('%Y-%m-%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    HEAD.to_csv(out / "headline.csv")
    SIG.to_csv(out / "significance.csv", index=False)
    SLICES.to_csv(out / "slices.csv")
    REFUSALS.to_csv(out / "refusals.csv")
    FIXED.to_csv(out / "fixed_by_week3.csv", index=False)
    BROKEN.to_csv(out / "broken_by_week3.csv", index=False)
    TAX.to_csv(out / "failure_taxonomy.csv")
    COST.to_csv(out / "cost_speed.csv")
    for name, fig in (("headline", FIG_HEADLINE), ("confusion", FIG_CONFUSION), ("failure_taxonomy", FIG_TAXONOMY)):
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
    (out / "summary.md").write_text(SUMMARY_MD)
    print("Saved to:", out)
    for p in sorted(out.iterdir()):
        print("  ", p.name)
else:
    print("SAVE_TO_DRIVE is off: nothing written.")
''')

nb = {"cells": cells, "metadata": {
    "colab": {"provenance": [], "name": "MedEvidence_Copilot_Week2_vs_Week3.ipynb"},
    "kernelspec": {"display_name": "Python 3", "name": "python3", "language": "python"},
    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("wrote", OUT, len(cells), "cells")
