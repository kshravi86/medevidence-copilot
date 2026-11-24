"""Client for the shared bge-small-en-v1.5 embedding endpoint (384-d, <=16 texts/request)."""
import hashlib
import sqlite3
import time

import httpx
import numpy as np

from .config import CACHE_DIR, abs_path, env


class TooLong(Exception):
    pass


class Embedder:
    def __init__(self):
        self.base = env("EMBEDDING_BASE_URL").rstrip("/")
        self.model = env("EMBEDDING_MODEL")
        self.key = env("EMBEDDING_API_KEY")
        self.client = httpx.Client(verify=abs_path(env("EMBEDDING_CA_CERT_PATH")), timeout=120)
        CACHE_DIR.mkdir(exist_ok=True)
        self.db = sqlite3.connect(CACHE_DIR / "emb.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS e (k TEXT PRIMARY KEY, v BLOB)")
        self.truncated = set()  # texts that exceeded the 512-token limit and were cut
        self._tok = None

    def _key(self, t):
        return hashlib.sha256((self.model + "\x00" + t).encode()).hexdigest()

    def _call(self, batch, retries=5):
        delay = 1
        for attempt in range(retries + 1):
            try:
                r = self.client.post(f"{self.base}/embeddings", json={"model": self.model, "input": batch},
                                     headers={"Authorization": f"Bearer {self.key}"})
            except httpx.HTTPError:
                r = None
            if r is not None and r.status_code == 200:
                d = sorted(r.json()["data"], key=lambda x: x["index"])
                return [np.asarray(x["embedding"], dtype=np.float32) for x in d]
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            if r is not None and r.status_code == 422:
                raise TooLong(r.text)
            raise RuntimeError(f"embedding call failed: {None if r is None else (r.status_code, r.text[:200])}")

    def _prep(self, text, max_tokens=500):
        """Cut text to the endpoint's 512-token limit using the model's own tokenizer (exact, one pass)."""
        if self._tok is None:
            from transformers import AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        enc = self._tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= max_tokens:
            return text
        self.truncated.add(text)
        return text[: enc["offset_mapping"][max_tokens - 1][1]]

    def _one_truncating(self, text):
        """Server rejects >512 tokens: shrink the word list until it fits (and remember we did)."""
        words = text.split()
        while True:
            try:
                return self._call([" ".join(words)])[0]
            except TooLong:
                self.truncated.add(text)
                words = words[: int(len(words) * 0.9)]

    def embed(self, texts, progress=False):
        """Return L2-normalised (len(texts), 384) float32 matrix; cached on disk."""
        from .llm import cache_off
        fresh = cache_off()
        out = [None] * len(texts)
        todo = []
        for i, t in enumerate(texts):
            row = None if fresh else self.db.execute("SELECT v FROM e WHERE k=?", (self._key(t),)).fetchone()
            if row:
                out[i] = np.frombuffer(row[0], dtype=np.float32)
            else:
                todo.append(i)
        for s in range(0, len(todo), 16):
            idx = todo[s:s + 16]
            try:
                vecs = self._call([self._prep(texts[i]) for i in idx])
            except TooLong:
                vecs = [self._one_truncating(texts[i]) for i in idx]
            for i, v in zip(idx, vecs):
                out[i] = v
                if not fresh:
                    self.db.execute("INSERT OR REPLACE INTO e VALUES (?,?)", (self._key(texts[i]), v.tobytes()))
            if not fresh:
                self.db.commit()
            if progress and (s // 16) % 25 == 0:
                print(f"  embedded {s + len(idx)}/{len(todo)}", flush=True)
        m = np.vstack(out)
        return m / np.linalg.norm(m, axis=1, keepdims=True)
