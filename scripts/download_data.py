"""Download the public PubMedQA files this project is built from.

  data/raw/ori_pqal.json, data/raw/test_ground_truth.json   <- github.com/pubmedqa/pubmedqa (PQA-L, 1,000 labelled)
  data/parquet/pqa_artificial.parquet, pqa_unlabeled.parquet <- huggingface.co/datasets/qiaojin/PubMedQA (distractors)

Prints the SHA-256 of each file so a fresh copy can be checked against the one the results were produced with.
Usage: python scripts/download_data.py   (then build_eval_set.py, build_corpus.py, build_index.py, build_faiss.py)
"""
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PARQUET = ROOT / "data" / "parquet"
GITHUB = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PARQUET.mkdir(parents=True, exist_ok=True)
    out = []
    for name in ("ori_pqal.json", "test_ground_truth.json"):
        dst = RAW / name
        if not dst.exists():
            urllib.request.urlretrieve(GITHUB + name, dst)
        out.append(dst)
    for config in ("pqa_artificial", "pqa_unlabeled"):
        dst = PARQUET / f"{config}.parquet"
        if not dst.exists():
            src = hf_hub_download("qiaojin/PubMedQA", f"{config}/train-00000-of-00001.parquet", repo_type="dataset")
            shutil.copyfile(src, dst)
        out.append(dst)
    for p in out:
        print(f"{sha256(p)}  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
