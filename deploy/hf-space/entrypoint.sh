#!/bin/bash
# Writes the endpoint CA certificates from Space secrets, checks the course model server is
# reachable from here (printed in the Space logs), then starts the API + chat page on :7860.
set -e
mkdir -p certs
if [ -n "$CA_CERT_PEM" ]; then printf '%s\n' "$CA_CERT_PEM" > certs/llm-ca.crt; export CA_CERT_PATH=certs/llm-ca.crt; fi
if [ -n "$EMBEDDING_CA_CERT_PEM" ]; then printf '%s\n' "$EMBEDDING_CA_CERT_PEM" > certs/emb-ca.crt; export EMBEDDING_CA_CERT_PATH=certs/emb-ca.crt; fi

python - <<'PY' || true
import os, httpx
for name, base, cert in (("model", "LLM_BASE_URL", "CA_CERT_PATH"), ("embedding", "EMBEDDING_BASE_URL", "EMBEDDING_CA_CERT_PATH")):
    url = os.environ.get(base, "").rstrip("/")
    if not url:
        print(f"[check] {name} server: {base} is not set (add it as a Space secret)", flush=True); continue
    try:
        r = httpx.get(url + "/models", verify=os.environ.get(cert) or True, timeout=10)
        print(f"[check] {name} server reachable (HTTP {r.status_code})", flush=True)
    except Exception as e:
        print(f"[check] {name} server NOT reachable from this Space: {type(e).__name__}: {e}", flush=True)
PY

exec python app/server.py
