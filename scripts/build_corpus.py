"""Build data/corpus.jsonl: 1000 PQA-L abstracts + 500 PQA-A + 500 PQA-U distractors (seed 42)."""
import collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from medevidence import data

docs = data.build_corpus()
print(len(docs), collections.Counter(d["source"] for d in docs))
ev = data.load_eval()
ids = {d["pmid"] for d in docs}
miss = [r["id"] for r in ev if r["gold_pmid"] and r["gold_pmid"] not in ids]
print("eval gold PMIDs missing from corpus:", miss)
import statistics
print("median words/abstract:", statistics.median(len(d["text"].split()) for d in docs))
print("sections/abstract mean:", sum(len(d["sections"]) for d in docs) / len(docs))
assert not any("long_answer" in d for d in docs)
