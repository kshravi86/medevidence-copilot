"""Agent state: per-request working state, and a session store that persists across turns.

Sessions live in SQLite (.cache/sessions.sqlite) so a follow-up such as "and in children?" can be
resolved by the planner after a restart. Only the question, status, verdict and cited PMID of each
turn are stored, never abstracts or model reasoning.
"""
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field

from ..config import CACHE_DIR


@dataclass
class Budget:
    max_steps: int = 30          # LLM calls + tool calls per request (infinite-loop guard)
    max_tokens: int = 30000      # prompt + completion tokens per request (token budget cap)
    max_revisions: int = 2       # critic -> executor send-backs (spec section 2)
    max_replans: int = 1         # planner re-searches after insufficient evidence
    steps: int = 0
    tokens: int = 0

    def charge(self, steps=1, tokens=0):
        self.steps += steps
        self.tokens += tokens

    def exhausted(self):
        if self.steps >= self.max_steps:
            return "step_limit"
        if self.tokens >= self.max_tokens:
            return "token_budget"
        return None


@dataclass
class AgentState:
    question: str
    session_id: str | None = None
    history: list = field(default_factory=list)       # prior turns of this session
    plan: dict | None = None
    queries_tried: list = field(default_factory=list)
    candidates: list = field(default_factory=list)    # [{"pmid", "score", "query"}]
    drafts: list = field(default_factory=list)        # every executor draft
    critiques: list = field(default_factory=list)     # every critic verdict
    tool_calls: int = 0
    tool_errors: list = field(default_factory=list)
    llm_calls: int = 0
    json_repairs: int = 0
    budget: Budget = field(default_factory=Budget)


class SessionStore:
    def __init__(self, path=None, max_turns=6):
        CACHE_DIR.mkdir(exist_ok=True)
        self.db = sqlite3.connect(path or CACHE_DIR / "sessions.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS turns (sid TEXT, ts REAL, turn TEXT)")
        self.lock = threading.Lock()
        self.max_turns = max_turns

    @staticmethod
    def new_id():
        return uuid.uuid4().hex[:16]

    def history(self, sid):
        if not sid:
            return []
        with self.lock:
            rows = self.db.execute("SELECT turn FROM turns WHERE sid=? ORDER BY ts DESC LIMIT ?",
                                   (sid, self.max_turns)).fetchall()
        return [json.loads(r[0]) for r in reversed(rows)]

    def append(self, sid, result):
        if not sid:
            return
        turn = {k: result.get(k) for k in ("question", "status", "verdict", "citation_pmid", "refusal_category")}
        with self.lock:
            self.db.execute("INSERT INTO turns VALUES (?,?,?)", (sid, time.time(), json.dumps(turn)))
            self.db.commit()
