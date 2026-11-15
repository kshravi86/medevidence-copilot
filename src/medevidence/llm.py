"""OpenAI-compatible chat client for the shared Qwen3-8B endpoint.

Constraints from the Learner Guide: 1 active request per group key, 60 req/min,
8192 tokens in+out, max 1024 output tokens. We therefore serialise requests with a
lock, retry 429/503 with 2/4/8 s backoff, and cache every response on disk so that
re-running an experiment is free and deterministic.
"""
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field

import httpx

from .config import CACHE_DIR, abs_path, env

_LOCK = threading.Lock()


@dataclass
class ChatResult:
    texts: list
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    cached: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def text(self):
        return self.texts[0]


class LLM:
    def __init__(self, cache_name="llm.sqlite"):
        self.base = env("LLM_BASE_URL").rstrip("/")
        self.model = env("LLM_MODEL")
        self.key = env("LLM_API_KEY")
        self.client = httpx.Client(verify=abs_path(env("CA_CERT_PATH")), timeout=120)
        CACHE_DIR.mkdir(exist_ok=True)
        self.db = sqlite3.connect(CACHE_DIR / cache_name, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.last_call = 0.0

    def chat(self, messages, *, temperature=0.0, max_tokens=512, n=1, seed=0,
             json_mode=False, retries=4) -> ChatResult:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "n": n, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        row = self.db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
        if row:
            d = json.loads(row[0])
            return ChatResult(d["texts"], d["pt"], d["ct"], d["lat"], cached=True)
        with _LOCK:
            delay = 2
            for attempt in range(retries + 1):
                gap = 1.05 - (time.time() - self.last_call)  # stay under 60 req/min
                if gap > 0:
                    time.sleep(gap)
                t0 = time.time()
                try:
                    r = self.client.post(f"{self.base}/chat/completions", json=body,
                                         headers={"Authorization": f"Bearer {self.key}"})
                except httpx.HTTPError:
                    r = None
                self.last_call = time.time()
                if r is not None and r.status_code == 200:
                    d = r.json()
                    texts = [c["message"]["content"] or "" for c in d["choices"]]
                    u = d.get("usage", {})
                    res = ChatResult(texts, u.get("prompt_tokens", 0),
                                     u.get("completion_tokens", 0), time.time() - t0)
                    self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (key, json.dumps(
                        {"texts": texts, "pt": res.prompt_tokens, "ct": res.completion_tokens,
                         "lat": res.latency})))
                    self.db.commit()
                    return res
                status = None if r is None else r.status_code
                if status in (None, 429, 503) and attempt < retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise RuntimeError(f"LLM call failed: HTTP {status} {'' if r is None else r.text[:300]}")
