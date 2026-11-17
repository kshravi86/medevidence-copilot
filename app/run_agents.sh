#!/usr/bin/env bash
# Launch three agy agents in parallel; each builds one file from app/SPEC.md.
set -u
cd "$(dirname "$0")/.."
mkdir -p app/logs
run() { # name, owned file, extra instruction
  agy --model gemini-3.8-flash-medium --dangerously-skip-permissions --add-dir "$PWD" \
    --print="You are agent '$1'. Read app/SPEC.md and build ONLY the file $2 exactly per its contract. $3 Do not edit any other file. When done, reply with one line: the file path and what you verified." \
    > "app/logs/$1.log" 2>&1 &
}
run pipeline app/pipeline.py "Verify by running: .venv/bin/python app/pipeline.py (the hosted LLM allows 1 request at a time; run it once)."
run server   app/server.py   "Server also owns POST /api/feedback. Verify with python -m py_compile only; the pipeline file is being written by another agent."
run ui       app/static/index.html "Verify the JS has no syntax errors and never uses innerHTML with model text."
wait
echo "all agents finished"
