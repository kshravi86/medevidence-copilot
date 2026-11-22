#!/usr/bin/env python3
"""MedEvidence Copilot HTTP API (Week 4: serving, rate limiting, caching).

Endpoints
  GET  /                 the plain HTML test UI (app/static/index.html)
  GET  /api/health       {"ok", "docs", "mode", "cache": {...}}
  GET  /api/examples     example questions
  POST /api/ask          {"question": str, "session_id"?: str} -> answer JSON (+ "cached", "session_id")
  POST /api/feedback     appends one JSON line to app/feedback.jsonl

Settings (environment)
  MEDEVIDENCE_MODE   agent (default: safety gate -> planner -> executor -> critic) | rag (Week 2 single pass)
  HOST / PORT        bind address (default 127.0.0.1:8100; use HOST=0.0.0.0 in Docker)
  RATE_LIMIT_PER_MIN per-client token bucket for /api/ask (default 10/min, burst RATE_LIMIT_BURST=5)
  CACHE_TTL_S / CACHE_SIZE   answer cache for stateless questions (default 3600 s, 256 entries)

The shared LLM key allows one active request, so /api/ask is serialised with a lock.
Follow-up questions in a session depend on history, so they bypass the cache.
"""

import json
import mimetypes
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
FEEDBACK_PATH = APP_DIR / "feedback.jsonl"
for p in (SRC, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MODE = os.environ.get("MEDEVIDENCE_MODE", "agent")
RATE_PER_MIN = float(os.environ.get("RATE_LIMIT_PER_MIN", "10"))
RATE_BURST = float(os.environ.get("RATE_LIMIT_BURST", "5"))
CACHE_TTL = float(os.environ.get("CACHE_TTL_S", "3600"))
CACHE_SIZE = int(os.environ.get("CACHE_SIZE", "256"))
MAX_BODY = 16_384

_backend = None
_backend_lock = threading.Lock()
_ask_lock = threading.Lock()
_feedback_lock = threading.Lock()


# ----------------------------------------------------------------------------- backend
def get_backend():
    """Lazily build the agent (or the Week 2 pipeline) once per process."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                if MODE == "rag":
                    from pipeline import Pipeline
                    _backend = Pipeline()
                else:
                    from medevidence.agent import build_agent
                    _backend = build_agent()
    return _backend


def doc_count(b):
    for attr in ("absmap",):
        if hasattr(b, attr):
            return len(getattr(b, attr))
    if hasattr(b, "tools"):
        return len(b.tools.abs)
    return 0


def answer(question, session_id):
    b = get_backend()
    if MODE == "rag":
        return b.ask(question)
    return b.ask(question, session_id=session_id)


# ----------------------------------------------------------------------------- rate limit + cache
class RateLimiter:
    """Token bucket per client key: RATE_PER_MIN refill, RATE_BURST capacity."""

    def __init__(self, per_min, burst):
        self.rate, self.cap = per_min / 60.0, burst
        self.buckets, self.lock = {}, threading.Lock()

    def check(self, key):
        """Returns (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self.lock:
            tokens, last = self.buckets.get(key, (self.cap, now))
            tokens = min(self.cap, tokens + (now - last) * self.rate)
            if tokens >= 1:
                self.buckets[key] = (tokens - 1, now)
                return True, 0
            self.buckets[key] = (tokens, now)
            return False, int((1 - tokens) / self.rate) + 1


class TTLCache:
    def __init__(self, size, ttl):
        self.size, self.ttl = size, ttl
        self.data, self.lock = OrderedDict(), threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def key(q):
        return re.sub(r"\s+", " ", q.strip().lower())

    def get(self, q):
        k = self.key(q)
        with self.lock:
            v = self.data.get(k)
            if v and time.time() - v[0] < self.ttl:
                self.data.move_to_end(k)
                self.hits += 1
                return v[1]
            self.data.pop(k, None)
            self.misses += 1
            return None

    def put(self, q, value):
        with self.lock:
            self.data[self.key(q)] = (time.time(), value)
            self.data.move_to_end(self.key(q))
            while len(self.data) > self.size:
                self.data.popitem(last=False)

    def stats(self):
        return {"entries": len(self.data), "hits": self.hits, "misses": self.misses, "ttl_s": self.ttl}


LIMITER = RateLimiter(RATE_PER_MIN, RATE_BURST)
CACHE = TTLCache(CACHE_SIZE, CACHE_TTL)

EXAMPLES = [
    {"label": "GLP-1 in Heart Failure",
     "question": "Do GLP-1 receptor agonists reduce cardiovascular mortality in patients with heart failure?"},
    {"label": "Steroids in Septic Shock",
     "question": "Does low-dose corticosteroid therapy improve 28-day survival in septic shock?"},
    {"label": "Stenting vs CABG in Left Main",
     "question": "Is percutaneous coronary intervention non-inferior to coronary artery bypass grafting for left main coronary artery disease?"},
    {"label": "Magnesium for Migraine",
     "question": "Does intravenous magnesium sulfate provide effective acute pain relief in emergency department migraine patients?"},
    {"label": "Personal Diagnosis (Refusal)",
     "question": "I have had a throbbing pain in my right calf for two days and it is red and warm to the touch. Do I have deep vein thrombosis?"},
    {"label": "Drug Dosing (Refusal)",
     "question": "What starting dose of metoprolol succinate should I take for high blood pressure, and how should I titrate it?"},
    {"label": "Emergency (Refusal)",
     "question": "I suddenly cannot speak clearly and my right arm feels completely numb and weak. What should I do right now?"},
]


def error_result(question, msg):
    return {"question": question, "status": "error", "error": msg, "refusal_category": None, "verdict": None,
            "explanation": None, "evidence_quote": None, "citation_pmid": None, "quote_verified": False,
            "citation_valid": False, "message": None, "retrieved": [], "latency_s": 0.0}


# ----------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "MedEvidence/1.0"

    def _json(self, code, data, headers=None):
        payload = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(payload)

    def _file(self, path, ctype):
        if not path.is_file():
            self.send_error(404, "File not found")
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _client(self):
        # Behind a proxy (Streamlit container) the caller can pass its own end-user id.
        return self.headers.get("X-Client-Id") or self.client_address[0]

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/api/health":
            try:
                return self._json(200, {"ok": True, "docs": doc_count(get_backend()), "mode": MODE,
                                        "cache": CACHE.stats(), "rate_limit_per_min": RATE_PER_MIN})
            except Exception as e:
                return self._json(500, {"ok": False, "docs": 0, "error": str(e)})
        if path == "/api/examples":
            return self._json(200, EXAMPLES)
        if path.startswith("/static/"):
            target = (STATIC_DIR / path[len("/static/"):].lstrip("/")).resolve()
            if STATIC_DIR in target.parents and target.is_file():
                ctype, _ = mimetypes.guess_type(str(target))
                return self._file(target, ctype or "application/octet-stream")
        self.send_error(404, "Not Found")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            n = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            n = 0
        if n <= 0 or n > MAX_BODY:
            return self._json(400 if n <= 0 else 413, {"status": "error", "error": "Missing or too large request body"})
        try:
            body = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._json(400, {"status": "error", "error": f"Invalid JSON payload: {e}"})
        if not isinstance(body, dict):
            return self._json(400, {"status": "error", "error": "Body must be a JSON object"})

        if path == "/api/ask":
            if "question" not in body:
                return self._json(400, {"status": "error", "error": "Request body must contain 'question'"})
            question = str(body["question"])
            sid = body.get("session_id")
            sid = str(sid)[:64] if sid else None
            ok, retry = LIMITER.check(self._client())
            if not ok:
                return self._json(429, {"status": "error", "error": f"Rate limit exceeded, retry in {retry}s"},
                                  {"Retry-After": retry})
            stateless = not sid or not (MODE == "agent" and get_backend().sessions
                                       and get_backend().sessions.history(sid))
            if stateless:
                hit = CACHE.get(question)
                if hit is not None:
                    res = dict(hit, cached=True, session_id=sid)
                    if sid and MODE == "agent" and get_backend().sessions:
                        get_backend().sessions.append(sid, res)  # keep the session history complete
                    return self._json(200, res)
            with _ask_lock:
                try:
                    res = answer(question, sid)
                except Exception as e:
                    res = error_result(question, str(e))
            res["cached"] = False
            if stateless and res.get("status") in ("answered", "refused", "insufficient_evidence"):
                CACHE.put(question, {k: v for k, v in res.items() if k != "session_id"})
            return self._json(200, res)

        if path == "/api/feedback":
            with _feedback_lock:
                try:
                    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(dict(body, ts=time.time()), ensure_ascii=False) + "\n")
                except OSError as e:
                    return self._json(500, {"ok": False, "error": f"Failed to record feedback: {e}"})
            return self._json(200, {"ok": True})
        self.send_error(404, "Not Found")

    def log_message(self, fmt, *args):  # no request bodies in logs
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} - {fmt % args}\n")


def run_server(host=None, port=None):
    host = host or os.environ.get("HOST", "127.0.0.1")
    port = int(port or os.environ.get("PORT", "8100"))
    print(f"Starting MedEvidence API ({MODE}) at http://{host}:{port} ...", flush=True)
    try:
        print(f"Backend ready. Indexed documents: {doc_count(get_backend())}", flush=True)
    except Exception as e:
        print(f"Backend init failed (will retry on first request): {e}", flush=True)
    srv = ThreadingHTTPServer((host, port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    run_server()
