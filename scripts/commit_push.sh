#!/usr/bin/env bash
# Commit and push your work on aakash_r_changes, with safety checks.
#
#   ./scripts/commit_push.sh code      # code, eval set, config  (safe any time)
#   ./scripts/commit_push.sh results   # results/ and docs/      (after the runs finish)
#
# It refuses to commit secrets (.env, certs/, data/, API keys) and never touches main.
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH="aakash_r_changes"
MODE="${1:-code}"

if [ "$(git branch --show-current)" != "$BRANCH" ]; then
  echo "ERROR: you are on '$(git branch --show-current)', expected '$BRANCH'. Aborting." >&2
  exit 1
fi

case "$MODE" in
  code)
    PATHS=(.gitignore .env.example eval src scripts)
    MSG=$'Add Week 1/2 pipeline: eval set, baselines, hybrid RAG retrieval\n\n- import frozen 100-item eval set (from capstone.zip) and build script\n- LLM/embedding clients with caching and retry, BM25 + dense + RRF + reranker\n- Week 1 baseline runner, Week 2 retrieval eval and end-to-end RAG runner'
    ;;
  results)
    if pgrep -f "run_week1.py|run_week2.py" >/dev/null; then
      echo "WARNING: a run is still writing to results/. Wait for it to finish, then re-run." >&2
      exit 1
    fi
    PATHS=(results docs)
    MSG=$'Add Week 1 baseline and Week 2 RAG results'
    ;;
  *)
    echo "Usage: $0 [code|results]" >&2
    exit 1
    ;;
esac
# Only add paths that exist
EXISTING=()
for p in "${PATHS[@]}"; do [ -e "$p" ] && EXISTING+=("$p"); done
git add -- "${EXISTING[@]}"

# --- safety checks on what is staged ---
BAD=$(git diff --cached --name-only | grep -E '(^|/)\.env$|(^|/)(certs|data|\.cache|\.venv)/|\.crt$|\.pem$' || true)
if [ -n "$BAD" ]; then
  echo "ERROR: sensitive/ignored paths are staged:" >&2
  echo "$BAD" >&2
  git reset -q
  exit 1
fi
if git diff --cached | grep -E '^\+' | grep -qE 'sk-[A-Za-z0-9_-]{20,}'; then
  echo "ERROR: something that looks like an API key is staged. Unstaging everything." >&2
  git reset -q
  exit 1
fi

echo "=== Files to be committed ==="
git diff --cached --stat
if git diff --cached --quiet; then
  echo "Nothing new to commit for '$MODE'."
else
  git commit -q -m "$MSG"
  echo "Committed: $(git log -1 --oneline)"
fi

echo "=== Pushing $BRANCH ==="
git push -u origin "$BRANCH"
echo "Done. Open a PR into main with:  gh pr create --base main"
