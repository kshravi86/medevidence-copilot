"""Paths and environment settings. Secrets come from .env (git-ignored)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

EVAL_PATH = ROOT / "eval" / "eval_set.jsonl"
RAW_DIR = ROOT / "data" / "raw"
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
CACHE_DIR = ROOT / ".cache"
RESULTS = ROOT / "results"
HF_DIR = Path(os.environ.get("PUBMEDQA_HF_DIR", "/Users/aakashr/pubmedqa/hf_data"))


def env(name: str, required: bool = True) -> str:
    v = os.environ.get(name, "")
    if required and not v:
        raise RuntimeError(f"{name} is not set (see .env.example)")
    return v


def abs_path(p: str) -> str:
    q = Path(p)
    return str(q if q.is_absolute() else ROOT / q)
