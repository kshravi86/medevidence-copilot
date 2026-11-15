"""Week 2: chunk the corpus three ways and embed every chunk + every eval query (bge-small-en-v1.5).
Embeddings are cached in .cache/emb.sqlite so this is resumable. Writes data/index/*.npy|json."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from medevidence import data
from medevidence.chunking import CHUNKERS
from medevidence.embed import Embedder

QPREFIX = "Represent this sentence for searching relevant passages: "
IDX = ROOT / "data" / "index"
IDX.mkdir(parents=True, exist_ok=True)
docs, ev, emb = data.load_corpus(), data.load_eval(), Embedder()
t0 = time.time()
for name, fn in CHUNKERS.items():
    ch = fn(docs)
    print(f"{name}: {len(ch)} chunks (median {sorted(len(t.split()) for _, t in ch)[len(ch)//2]} words)", flush=True)
    json.dump([{"pmid": p, "text": t} for p, t in ch], open(IDX / f"chunks_{name}.json", "w"))
    np.save(IDX / f"emb_{name}.npy", emb.embed([t for _, t in ch], progress=True))
    print(f"  done {time.time()-t0:.0f}s", flush=True)
qs = [i["query"] for i in ev]
np.save(IDX / "q_plain.npy", emb.embed(qs))
np.save(IDX / "q_prefixed.npy", emb.embed([QPREFIX + q for q in qs]))
print("queries embedded; total", f"{time.time()-t0:.0f}s")
json.dump(len(emb.truncated), open(IDX / "n_truncated_last_run.json", "w"))
