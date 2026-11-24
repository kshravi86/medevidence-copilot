"""Week 3 and 4 walkthrough as a plain script: the same steps as notebooks/MedEvidence_Copilot_Week3_and_4.ipynb.

Runs on your machine (reads .env, like every other script) and in Colab after the notebook's Setup cells
(they put the credentials in the environment), so you can set breakpoints anywhere in src/ and step through.

Steps (run in this order; pick any with --steps):
  check     the eval set and corpus match the frozen versions; the index and FAISS store exist
  tests     pytest on tests/
  safety    the safety gate's rules on seven example questions
  demo      one question (Q003) through the whole agent, with its trace and OpenTelemetry span tree
  followup  a follow-up question in the same session
  samples   the agent on SAMPLE_IDS, scored like the evaluation
  full      the full Week 3 run on all 100 questions (uses quota; also --no-critic for the ablation)
  results   Week 3 summary tables, critic, failure taxonomy, one example per failure class
  rouge     ROUGE-L table and the FAISS check
  api       start the Week 4 API on a free port and test caching, sessions, rate limiting, feedback
  ui        start the Streamlit UI against that API (keeps running until you press Ctrl+C)

Examples:
  python scripts/walkthrough_week3_4.py                       # default: check safety demo followup
  python scripts/walkthrough_week3_4.py --steps demo           # just the Q003 demo
  python scripts/walkthrough_week3_4.py --steps samples --ids Q003,Q100
  python scripts/walkthrough_week3_4.py --steps api,ui
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import pandas as pd  # noqa: E402

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.width", 200)

DEFAULT_STEPS = ["check", "safety", "demo", "followup"]
ALL_STEPS = ["check", "tests", "safety", "demo", "followup", "samples", "full", "results", "rouge", "api", "ui"]
SAMPLE_IDS = ["Q001", "Q003", "Q071", "Q075", "Q026", "Q062", "Q080", "Q087", "Q094", "Q100"]
EVAL_SHA = "1a5befac61c86ce450dfc84e32f0940789d171f1b622f0a2a89863eb23ba6e56"
CORPUS_SHA = "da23af8479e15b7f3462aa40a65b91a53597ad8e7aeb11e1ad44b7a85e9e99cb"
IN_COLAB = Path("/content").exists() and "COLAB_RELEASE_TAG" in os.environ


# ---------------------------------------------------------------- helpers

def heading(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100, flush=True)


def table(rows, title=None, columns=None):
    if title:
        print(f"\n{title}")
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    print(df.to_string(index=False))


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def wrap(s, indent="    "):
    return textwrap.fill(str(s), width=100, initial_indent=indent, subsequent_indent=indent)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*args):
    """Run a project script (or `-m module`) from the repo root with this Python; stop if it fails."""
    print("$ python", *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def free_port(preferred):
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def show(r):
    """Print one agent result: outcome, plan, abstracts read, answer, critic and usage."""
    print(f"Status: {r['status']}   Verdict: {r.get('verdict')}   PMID: {r.get('citation_pmid')}   "
          f"quote verified: {r.get('quote_verified')}   {r['latency_s']:.1f}s")
    if r.get("message"):
        print("Message:"); print(wrap(r["message"]))
    if r.get("plan"):
        print("Planner's research question:"); print(wrap(r["plan"].get("research_question")))
        print("Search queries:", r["plan"].get("queries"))
    if r.get("retrieved"):
        print("Abstracts read:", [d["pmid"] for d in r["retrieved"]])
    if r.get("explanation"):
        print("Explanation:"); print(wrap(r["explanation"]))
        print("Quote:"); print(wrap(r.get("evidence_quote")))
    if r.get("critic"):
        c = r["critic"]
        print(f"Critic: approved {c['approved']}, revisions {c['revisions']}, downgraded {c['downgraded']}, "
              f"draft verdicts {c.get('draft_verdicts')}")
    u = r["usage"]
    print(f"Usage: {u['llm_calls']} model calls, {u['tool_calls']} tool calls, {u['tokens']} tokens, {u['steps']} steps")


def trace_tree(trace_id):
    """Print the OpenTelemetry spans of one request as an indented tree."""
    path = Path(os.environ.get("MEDEVIDENCE_TRACE_FILE", ROOT / "results/traces/spans.jsonl"))
    spans = [json.loads(l) for l in open(path)] if path.exists() else []
    spans = [s for s in spans if s["trace_id"] == trace_id]
    if not spans:
        print("No spans found (is opentelemetry installed?)")
        return
    kids, ids = {}, {s["span_id"] for s in spans}
    for s in spans:
        kids.setdefault(s["parent_id"], []).append(s)

    def walk(s, depth):
        a = s["attributes"]
        extra = ", ".join(f"{k}={a[k]}" for k in ("prompt_tokens", "completion_tokens", "cached", "category", "n") if k in a)
        print(f"{'  ' * depth}{s['name']:<28} {s['duration_ms']:>8.1f} ms  {extra}")
        for c in sorted(kids.get(s["span_id"], []), key=lambda x: x["start_ns"]):
            walk(c, depth + 1)

    for root in sorted([s for s in spans if s["parent_id"] not in ids], key=lambda x: x["start_ns"]):
        walk(root, 0)


# ---------------------------------------------------------------- shared state, loaded only when a step needs it

class Ctx:
    def __init__(self, args):
        self.args = args
        self._agent = self._llm = None
        self.api = None
        self.procs = []

    @property
    def by_id(self):
        from medevidence import data
        return {item["id"]: item for item in data.load_eval()}

    @property
    def llm(self):
        if self._llm is None:
            from medevidence.llm import LLM
            self._llm = LLM()
        return self._llm

    @property
    def agent(self):
        if self._agent is None:
            from medevidence.agent import build_agent
            print("Loading the agent (index, FAISS store and reranker)...", flush=True)
            t = time.time()
            self._agent = build_agent()            # with sessions, as the API uses it
            print(f"Agent ready in {time.time() - t:.0f}s: {len(self._agent.tools.abs)} abstracts, "
                  f"FAISS {self._agent.tools.index.faiss is not None}, "
                  f"reranker on {os.environ.get('RERANK_DEVICE', 'mps')}", flush=True)
        return self._agent


# ---------------------------------------------------------------- steps

def step_check(ctx):
    heading("Data: frozen eval set and corpus, index and FAISS store")
    eval_ok = sha256(ROOT / "eval/eval_set.jsonl") == EVAL_SHA
    corpus_path = ROOT / "data/corpus.jsonl"
    corpus_ok = corpus_path.exists() and sha256(corpus_path) == CORPUS_SHA
    print("eval/eval_set.jsonl matches the frozen version:", eval_ok)
    print("data/corpus.jsonl matches the committed results:", corpus_ok)
    if not eval_ok or not corpus_ok:
        raise SystemExit("Rebuild with scripts/build_eval_set.py and scripts/build_corpus.py (see the notebook's step 5)")
    index = ROOT / "data/index"
    if not (index / "q_prefixed.npy").exists():
        run("scripts/build_index.py")
    if not (index / "faiss_abstract_flat.index").exists():
        run("scripts/build_faiss.py")
    table([{"chunking": n, "chunks": len(json.load(open(index / f"chunks_{n}.json")))}
           for n in ("abstract", "section", "window")])


def step_tests(ctx):
    heading("Unit tests")
    run("-m", "pytest", "-q", "tests")


def step_safety(ctx):
    heading("The safety gate on its own")
    from medevidence.agent import safety
    examples = [
        "Does CBT reduce suicidal ideation in adolescents?",
        "Do higher loading doses of vancomycin improve target attainment?",
        "Can we predict outcome in chest pain patients?",
        "My dad just collapsed and is not breathing",
        "I have crushing chest pain right now",
        "How many mg of ibuprofen should I give my son?",
        "My PSA went from 4 to 7, does that mean I have cancer?",
    ]
    rows = []
    for q in examples:
        cat, rule = safety.rule_check(q)
        rows.append({"question": q, "rule category": cat or "(passes)", "rule": rule or ""})
    table(rows)
    print("\nEmergency refusal text:\n" + wrap(safety.refusal("emergency")))


def step_demo(ctx):
    heading("Follow one question through the agent (Q003)")
    item = ctx.by_id["Q003"]
    print("Question:", item["query"])
    print("Correct answer:", item["gold_label"], "| correct PMID:", item["gold_pmid"], "\n")
    demo = ctx.agent.ask(item["query"], session_id=ctx.args.session)
    show(demo)
    df = pd.DataFrame(demo["trace"])
    cols = [c for c in ("step", "ms", "prompt_tokens", "completion_tokens", "cached", "query", "pmid", "ok") if c in df]
    df = df[cols].astype(object).where(df[cols].notna(), "")
    if "query" in df:
        df["query"] = df["query"].map(lambda q: q if len(q) <= 60 else q[:57] + "...")
    table(df, "\nTrace (main columns):")
    print("\nOpenTelemetry span tree:")
    trace_tree(demo["trace_id"])
    ctx.demo = demo


def step_followup(ctx):
    heading("A follow-up in the same session")
    if not getattr(ctx, "demo", None):
        print(f"(No demo in this run: the follow-up uses whatever session '{ctx.args.session}' already holds.)")
    follow = ctx.agent.ask("Is that also true after hip replacement surgery?", session_id=ctx.args.session)
    show(follow)


def score(ctx, item, r):
    import run_week3
    from medevidence import judge, metrics
    from medevidence.rouge import rouge_l
    corpus = ctx.agent.tools.abs
    o, err = run_week3.pred_obj(r)
    s = metrics.score_item(item, o, err, corpus, citation_pmid=r.get("citation_pmid"), has_citation=True)
    if item["expected_behavior"] == "answer":
        if r["status"] == "answered" and r.get("citation_pmid") in corpus:
            j = judge.judge(ctx.llm, item, corpus[r["citation_pmid"]], r.get("explanation"))
            s["c4_faithful"], s["c4_unsupported"] = judge.faithful(j), j["unsupported"]
        else:
            s["c4_faithful"] = False
        s["rougeL"] = round(rouge_l(r.get("explanation") if r["status"] == "answered" else None, item["gold_rationale"])["f"], 3)
    metrics.finish_strict(s)
    s["failure"] = run_week3.taxonomy(item, r, s, corpus)
    return s


def step_samples(ctx):
    heading("The agent on the sample, scored like the evaluation")
    from medevidence.agent import Agent
    eval_agent = Agent(ctx.llm, ctx.agent.tools)    # no session history, as in the evaluation
    rows = []
    for qid in ctx.args.ids:
        item = ctx.by_id[qid]
        r = eval_agent.ask(item["query"])
        s = score(ctx, item, r)
        said = (f"refused: {r['refusal_category']}" if r["status"] == "refused"
                else r.get("verdict") if r["status"] == "answered" else r["status"])
        rows.append({"id": qid, "type": item["query_type"],
                     "correct": item["gold_label"] or f"refuse: {item['refusal_category']}",
                     "agent": said, "pass": bool(s.get("strict") if item["expected_behavior"] == "answer" else s.get("correct")),
                     "failure": s["failure"] or "", "critic revisions": (r.get("critic") or {}).get("revisions"),
                     "calls (model/tool)": f"{r['usage']['llm_calls']}/{r['usage']['tool_calls']}",
                     "seconds": round(r["latency_s"], 1), "ROUGE-L": s.get("rougeL")})
        print("finished", qid, flush=True)
    table(rows)


def step_full(ctx):
    heading("Full Week 3 run on all 100 questions")
    run("scripts/run_week3.py", "--name", "agent")
    if ctx.args.no_critic:
        run("scripts/run_week3.py", "--name", "agent_nocritic", "--no-critic")


def step_results(ctx):
    heading("Week 3 results (results/week3/summary.json)")
    now = json.load(open(ROOT / "results/week3/summary.json"))
    metrics_ = [("strict_all", "strict (C1-C6)"), ("verdict_acc_all", "verdict (C1)"), ("macro_f1", "macro-F1"),
                ("citation_acc", "citation (C2)"), ("grounded_acc", "quote exact (C3)"), ("faithful_acc", "faithful (C4)"),
                ("format_all", "format (C5)"), ("hallucinated_pmids", "hallucinated PMIDs"),
                ("refusal_recall", "refused"), ("refusal_correct", "refusal fully right"),
                ("emergency_refused", "emergencies refused"), ("over_refusal_rate", "wrongly refused"),
                ("gold_in_context", "right abstract in context")]
    rows = []
    for key, label in metrics_:
        row = {"metric": label}
        for name, s in now.items():
            v = s.get(key)
            row[name] = round(v, 3) if key == "macro_f1" and v is not None else pct(v) if isinstance(v, float) else v
        rows.append(row)
    table(rows)

    base = now["agent"].get("vs_baselines", {})
    table([{"comparison": k, "agent": pct(v["agent"]), "baseline": pct(v["baseline"]),
            "difference": f"{v['diff'] * 100:+.1f} pts",
            "95% CI": f"[{v['ci95'][0] * 100:+.1f}, {v['ci95'][1] * 100:+.1f}]",
            "proven improvement": v["improvement"]} for k, v in base.items()],
          "\nPaired bootstrap, 10k resamples: agent minus baseline")

    s = now["agent"]
    table([
        {"measure": "task completion rate", "value": pct(s["task_completion_rate"])},
        {"measure": "model calls per question", "value": round(s["avg_llm_calls"], 2)},
        {"measure": "tool calls per question", "value": round(s["avg_tool_calls"], 2)},
        {"measure": "tokens per question", "value": round(s["avg_tokens"])},
        {"measure": "estimated cost per question (USD)", "value": f"{s['avg_cost_usd_est']:.5f}"},
        {"measure": "latency median / p90 (s)", "value": f"{s['latency_median_s']:.1f} / {s['latency_p90_s']:.1f}"},
        {"measure": "JSON repairs / tool errors", "value": f"{s['json_repairs']} / {s['tool_errors']}"},
    ], "\nObservability")
    table([s["critic"]], "\nWhat the critic did (answerable questions)")
    table(sorted(([k, v] for k, v in s["failure_taxonomy"].items()), key=lambda x: -x[1]), "\nWhy questions fail", columns=["failure", "questions"])

    rows = [json.loads(l) for l in open(ROOT / "results/week3/agent.jsonl")]
    errs = [r for r in rows if r.get("failure") == "reasoning_error"]
    print("\nReasoning errors: correct verdict down the side, the agent's across the top")
    print(pd.crosstab(pd.Series([r["gold"] for r in errs], name="correct"),
                      pd.Series([r["verdict"] for r in errs], name="agent")).to_string())

    by_id, seen = ctx.by_id, set()
    print("\nOne example of each failure class")
    for r in rows:
        f = r.get("failure")
        if not f or f in seen:
            continue
        seen.add(f)
        item, pred = by_id[r["id"]], r.get("pred") or {}
        print("-" * 100 + f"\n{f}  ·  {r['id']} ({item['query_type']})")
        print(wrap(item["query"]))
        print("Correct:", item["gold_label"] or f"REFUSE as {item['refusal_category']}")
        print("Agent:  ", r["verdict"] if r["status"] == "answered" else f"{r['status']} {pred.get('refusal_category') or ''}")
        if pred.get("explanation"):
            print("Explanation:"); print(wrap(pred["explanation"]))
        feedback = (r.get("critic_notes") or [{}])[-1].get("feedback")
        if feedback:
            print("Critic's last note:"); print(wrap("; ".join(feedback) if isinstance(feedback, list) else feedback))


def step_rouge(ctx):
    heading("ROUGE-L and the FAISS check")
    run("scripts/eval_rouge.py")
    rouge = json.load(open(ROOT / "results/rouge_l.json"))["summary"]
    keep = ["week1/closed__fewshot", "week1/bm25__fewshot", "week2/rag__abstract__bm25_1", "week2/rag__abstract__rerank_3",
            *[k for k in rouge if k.startswith("week3/")]]
    table([{"run": k, "all 78": v["rougeL_all"], "answered": v["rougeL_answered"],
            "right verdict": v["verdict_right"], "wrong verdict": v["verdict_wrong"]}
           for k, v in rouge.items() if k in keep])
    fc = json.load(open(ROOT / "results/faiss_check.json"))
    table([{"store": k, "top-10 identical": pct(v["top10_identical_to_numpy"]), "R@1": pct(v["R@1"]),
            "R@5": pct(v["R@5"]), "ms / query": v["ms_per_query"]} for k, v in fc.items()],
          "\nFAISS against the numpy scan, 100 eval queries")


def start_api(ctx):
    import requests
    port = free_port(ctx.args.api_port)
    ctx.api = f"http://127.0.0.1:{port}"
    env = {**os.environ, "HOST": "127.0.0.1", "PORT": str(port), "MEDEVIDENCE_MODE": "agent",
           "RATE_LIMIT_PER_MIN": "10", "RATE_LIMIT_BURST": "5", "CACHE_TTL_S": "3600"}
    log = open(ROOT / "app/logs_run.txt", "w")
    proc = subprocess.Popen([sys.executable, "app/server.py"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    ctx.procs.append(proc)
    print(f"Starting the API on {ctx.api} (log: app/logs_run.txt)...", flush=True)
    for _ in range(120):
        try:
            return requests.get(ctx.api + "/api/health", timeout=2).json()
        except requests.RequestException:
            if proc.poll() is not None:
                raise RuntimeError("The API stopped:\n" + (ROOT / "app/logs_run.txt").read_text()[-2000:])
            time.sleep(3)
    raise RuntimeError("The API did not start in 6 minutes")


def step_api(ctx):
    heading("Week 4 API: caching, sessions, rate limiting and feedback")
    import requests
    print(json.dumps(start_api(ctx), indent=1))

    def ask(question, session_id=None):
        t0 = time.time()
        resp = requests.post(ctx.api + "/api/ask", json={"question": question, "session_id": session_id}, timeout=300)
        return resp.status_code, resp.json(), time.time() - t0

    q = ctx.by_id["Q001"]["query"]
    print("Question:", q)
    rows = []
    for label, sid, question in (("first ask", None, q), ("same question again", None, q),
                                 ("new session (cached answer, saved to its history)", "api-demo", q),
                                 ("follow-up in that session", "api-demo", "Does the same hold for amateur athletes?")):
        code, body, secs = ask(question, sid)
        rows.append({"request": label, "HTTP": code, "status": body.get("status"), "verdict": body.get("verdict"),
                     "cached": body.get("cached"), "seconds": round(secs, 2)})
    table(rows)

    burst = []
    for i in range(8):
        resp = requests.post(ctx.api + "/api/ask", json={"question": q}, timeout=60)
        burst.append({"request": i + 1, "HTTP": resp.status_code, "Retry-After": resp.headers.get("Retry-After"),
                      "cached": resp.json().get("cached")})
    table(burst, "\n8 requests in a row from one client")
    print("\nFeedback endpoint:", requests.post(ctx.api + "/api/feedback",
                                                json={"question": q, "rating": "up", "note": "walkthrough test"}).json())
    print("Health:", requests.get(ctx.api + "/api/health").json().get("cache"))


def step_ui(ctx):
    heading("Streamlit UI")
    import requests
    if not ctx.api:
        start_api(ctx)
    port = free_port(ctx.args.ui_port)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", f"--server.port={port}",
                             "--server.headless=true", "--browser.gatherUsageStats=false"],
                            cwd=ROOT, env={**os.environ, "API_URL": ctx.api})
    ctx.procs.append(proc)
    for _ in range(60):
        try:
            requests.get(f"http://127.0.0.1:{port}/_stcore/health", timeout=2)
            break
        except requests.RequestException:
            time.sleep(2)
    print(f"UI running at http://127.0.0.1:{port}  (API {ctx.api}). Press Ctrl+C (or stop the debugger) to quit.")
    if IN_COLAB:
        print("In Colab, open it with: from google.colab import output; output.serve_kernel_port_as_iframe(%d)" % port)
    try:
        proc.wait()
    except KeyboardInterrupt:
        pass


STEPS = {name: globals()[f"step_{name}"] for name in ALL_STEPS}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", default=",".join(DEFAULT_STEPS),
                   help=f"comma-separated, from: {', '.join(ALL_STEPS)}, or 'all' (all except full)")
    p.add_argument("--ids", default=",".join(SAMPLE_IDS), help="question IDs for the samples step")
    p.add_argument("--session", default=None,
                   help="session ID for the demo and follow-up (default: a new one per run, so earlier runs "
                        "don't add history and the saved model answers can be reused)")
    p.add_argument("--no-critic", action="store_true", help="with 'full': also run the critic ablation")
    p.add_argument("--api-port", type=int, default=8100, help="preferred API port (a free one is used if taken)")
    p.add_argument("--ui-port", type=int, default=8501)
    args = p.parse_args(argv)
    args.ids = [i.strip() for i in args.ids.split(",") if i.strip()]
    args.session = args.session or f"walkthrough-{time.strftime('%Y%m%d-%H%M%S')}"
    steps = [s for s in ALL_STEPS if s != "full"] if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        p.error(f"unknown step(s): {unknown}")

    ctx = Ctx(args)
    try:
        for name in steps:
            STEPS[name](ctx)
    finally:
        for proc in ctx.procs:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
