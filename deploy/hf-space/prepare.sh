#!/bin/bash
# Stage the Hugging Face Space folder: code, eval set and the abstract-level index only (~15 MB).
# Usage: deploy/hf-space/prepare.sh <out_dir>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="${1:?usage: prepare.sh <out_dir>}"
rm -rf "$OUT"; mkdir -p "$OUT/app" "$OUT/data/index" "$OUT/data/raw"
cp "$ROOT/deploy/hf-space/Dockerfile" "$ROOT/deploy/hf-space/entrypoint.sh" "$ROOT/deploy/hf-space/README.md" "$OUT/"
cp "$ROOT/requirements.txt" "$OUT/"
cp -R "$ROOT/src" "$ROOT/eval" "$OUT/"
cp "$ROOT/app/server.py" "$ROOT/app/pipeline.py" "$OUT/app/"
cp -R "$ROOT/app/static" "$OUT/app/"
cp "$ROOT/data/corpus.jsonl" "$OUT/data/"
cp "$ROOT/data/raw/ori_pqal.json" "$ROOT/data/raw/test_ground_truth.json" "$OUT/data/raw/"  # few-shot demos (dev split)
cp "$ROOT/data/index/chunks_abstract.json" "$ROOT/data/index/emb_abstract.npy" \
   "$ROOT/data/index/faiss_abstract_flat.index" "$OUT/data/index/"
find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$OUT" -name "*.bak*" -delete
# never ship secrets
test ! -e "$OUT/.env" && ! grep -rIl "LLM_API_KEY=" "$OUT" >/dev/null
du -sh "$OUT"
