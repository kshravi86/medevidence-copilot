#!/usr/bin/env python3
"""HTTP Server for MedEvidence Copilot test application.

Owned by agent 'server'.
Serves static UI, health check, examples, question answering via Pipeline,
and clinician feedback recording.
"""

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from urllib.parse import urlparse

# Ensure src/ and app/ are in sys.path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
FEEDBACK_PATH = APP_DIR / "feedback.jsonl"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Thread locks
_pipeline = None
_pipeline_lock = threading.Lock()
_ask_lock = threading.Lock()
_feedback_lock = threading.Lock()


def get_pipeline():
    """Lazily import and initialize the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                try:
                    from app.pipeline import Pipeline
                except ImportError:
                    from pipeline import Pipeline
                _pipeline = Pipeline()
    return _pipeline


def get_doc_count(pipeline_obj):
    """Retrieve document count from pipeline or underlying corpus."""
    if hasattr(pipeline_obj, "corpus"):
        try:
            return len(pipeline_obj.corpus)
        except Exception:
            pass
    if hasattr(pipeline_obj, "absmap"):
        try:
            return len(pipeline_obj.absmap)
        except Exception:
            pass
    if hasattr(pipeline_obj, "docs"):
        try:
            d = pipeline_obj.docs
            return d if isinstance(d, int) else len(d)
        except Exception:
            pass
    if hasattr(pipeline_obj, "index") and hasattr(pipeline_obj.index, "pmids"):
        try:
            return len(set(pipeline_obj.index.pmids))
        except Exception:
            pass
    try:
        from medevidence.data import load_corpus
        return len(load_corpus())
    except Exception:
        return 0


EXAMPLES = [
    {
        "label": "GLP-1 in Heart Failure",
        "question": "Do GLP-1 receptor agonists reduce cardiovascular mortality in patients with heart failure?"
    },
    {
        "label": "Steroids in Septic Shock",
        "question": "Does low-dose corticosteroid therapy improve 28-day survival in septic shock?"
    },
    {
        "label": "Stenting vs CABG in Left Main",
        "question": "Is percutaneous coronary intervention non-inferior to coronary artery bypass grafting for left main coronary artery disease?"
    },
    {
        "label": "Magnesium for Migraine",
        "question": "Does intravenous magnesium sulfate provide effective acute pain relief in emergency department migraine patients?"
    },
    {
        "label": "Personal Diagnosis (Refusal)",
        "question": "I have had a throbbing pain in my right calf for two days and it is red and warm to the touch. Do I have deep vein thrombosis?"
    },
    {
        "label": "Drug Dosing (Refusal)",
        "question": "What starting dose of metoprolol succinate should I take for high blood pressure, and how should I titrate it?"
    },
    {
        "label": "Emergency (Refusal)",
        "question": "I suddenly cannot speak clearly and my right arm feels completely numb and weak. What should I do right now?"
    }
]


class MedEvidenceHandler(BaseHTTPRequestHandler):
    """Request handler implementing MedEvidence Copilot API and UI serving."""

    def _send_json(self, status_code: int, data: object) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, file_path: Path, content_type: str = "text/html; charset=utf-8") -> None:
        if not file_path.is_file():
            self.send_error(404, "File not found")
            return
        try:
            content = file_path.read_bytes()
        except OSError:
            self.send_error(500, "Error reading file")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path in ("/", "/index.html"):
            index_path = STATIC_DIR / "index.html"
            self._serve_file(index_path, "text/html; charset=utf-8")
            return

        if path == "/api/health":
            try:
                pipeline_obj = get_pipeline()
                docs = get_doc_count(pipeline_obj)
                self._send_json(200, {"ok": True, "docs": docs})
            except Exception as e:
                self._send_json(500, {"ok": False, "docs": 0, "error": str(e)})
            return

        if path == "/api/examples":
            self._send_json(200, EXAMPLES)
            return

        # Serve static assets from app/static/ if requested
        if path.startswith("/static/"):
            rel_path = path[len("/static/"):].lstrip("/")
            target = (STATIC_DIR / rel_path).resolve()
            if STATIC_DIR in target.parents and target.is_file():
                ctype, _ = mimetypes.guess_type(str(target))
                self._serve_file(target, ctype or "application/octet-stream")
                return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (ValueError, TypeError):
            content_length = 0

        if content_length <= 0:
            self._send_json(400, {"status": "error", "error": "Missing or empty request body"})
            return

        try:
            raw_body = self.rfile.read(content_length)
            body = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"status": "error", "error": f"Invalid JSON payload: {e}"})
            return

        if path == "/api/ask":
            if not isinstance(body, dict) or "question" not in body:
                self._send_json(400, {"status": "error", "error": "Request body must contain 'question' field"})
                return

            question = str(body["question"])
            with _ask_lock:
                try:
                    pipeline_obj = get_pipeline()
                    res = pipeline_obj.ask(question)
                except Exception as e:
                    res = {
                        "question": question,
                        "status": "error",
                        "error": str(e),
                        "refusal_category": None,
                        "verdict": None,
                        "explanation": None,
                        "evidence_quote": None,
                        "citation_pmid": None,
                        "quote_verified": False,
                        "citation_valid": False,
                        "message": None,
                        "retrieved": [],
                        "latency_s": 0.0,
                    }
            self._send_json(200, res)
            return

        if path == "/api/feedback":
            if not isinstance(body, dict):
                self._send_json(400, {"ok": False, "error": "Feedback payload must be a JSON object"})
                return

            with _feedback_lock:
                try:
                    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(body, ensure_ascii=False) + "\n")
                except OSError as e:
                    self._send_json(500, {"ok": False, "error": f"Failed to record feedback: {e}"})
                    return

            self._send_json(200, {"ok": True})
            return

        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: object) -> None:
        """Standard log without leaking sensitive payload data."""
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}\n")


def run_server(host: str = "127.0.0.1", port: int = 8100) -> None:
    # Lazily initialize pipeline at startup
    print(f"Starting server at http://{host}:{port} ...")
    try:
        pipeline_obj = get_pipeline()
        docs = get_doc_count(pipeline_obj)
        print(f"Pipeline ready. Indexed documents: {docs}")
    except Exception as e:
        print(f"Pipeline lazy init deferred or failed: {e}")

    server = ThreadingHTTPServer((host, port), MedEvidenceHandler)
    print(f"MedEvidence Copilot server listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
