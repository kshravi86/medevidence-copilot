"""Rate limiter and answer cache used by the API (Week 4)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from server import RateLimiter, TTLCache  # noqa: E402


def test_rate_limiter_burst_then_block_then_refill():
    rl = RateLimiter(per_min=600, burst=3)  # 10 tokens/s
    assert [rl.check("a")[0] for _ in range(3)] == [True, True, True]
    ok, retry = rl.check("a")
    assert not ok and retry >= 1
    assert rl.check("b")[0]  # other clients are independent
    time.sleep(0.15)
    assert rl.check("a")[0]


def test_cache_normalises_question_and_expires():
    c = TTLCache(size=2, ttl=0.2)
    c.put("Does  X help Y?", {"verdict": "yes"})
    assert c.get("does x help y?")["verdict"] == "yes"
    time.sleep(0.25)
    assert c.get("does x help y?") is None


def test_cache_evicts_least_recently_used():
    c = TTLCache(size=2, ttl=60)
    c.put("a", 1); c.put("b", 2); c.get("a"); c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
