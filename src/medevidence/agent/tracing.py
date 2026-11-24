"""Tracing for the agent (course brief V: LangSmith, Langfuse or OpenTelemetry).

Every agent step opens an OpenTelemetry span. Finished spans are written as JSON lines to
`MEDEVIDENCE_TRACE_FILE` (default results/traces/spans.jsonl), so traces can be inspected offline
or shipped to any OTLP backend later by swapping the exporter. Each request also keeps a light
in-memory step list (`Trace.steps`) that the API returns and the eval script aggregates.
If opentelemetry is not installed, spans are no-ops and only the step list is kept.
"""
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from ..config import ROOT

try:
    from opentelemetry import trace as ot
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
except ImportError:  # tracing is optional at runtime
    ot = None

_TRACER = None
_INIT = threading.Lock()


if ot is not None:
    class JsonlSpanExporter(SpanExporter):
        def __init__(self, path: Path):
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.lock = threading.Lock()

        def export(self, spans):
            with self.lock, open(self.path, "a") as f:
                for s in spans:
                    f.write(json.dumps({
                        "trace_id": format(s.context.trace_id, "032x"),
                        "span_id": format(s.context.span_id, "016x"),
                        "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
                        "name": s.name,
                        "start_ns": s.start_time, "end_ns": s.end_time,
                        "duration_ms": round((s.end_time - s.start_time) / 1e6, 1),
                        "status": s.status.status_code.name,
                        "attributes": dict(s.attributes or {}),
                    }) + "\n")
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass


def tracer():
    global _TRACER
    if ot is None:
        return None
    with _INIT:
        if _TRACER is None:
            path = Path(os.environ.get("MEDEVIDENCE_TRACE_FILE", ROOT / "results" / "traces" / "spans.jsonl"))
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(path)))
            _TRACER = provider.get_tracer("medevidence.agent")
    return _TRACER


def _clean(v):
    if isinstance(v, (str, bool, int, float)):
        return v
    return json.dumps(v, default=str)[:2000]


class Trace:
    """Per-request record of steps: what ran, how long, tokens, errors."""

    def __init__(self, on_event=None):
        self.steps = []
        self.trace_id = None
        self.on_event = on_event  # optional callback(event: dict) for live step streaming (the /eval page)

    def _emit(self, event):
        if self.on_event is not None:
            try:
                self.on_event(event)
            except Exception:  # a broken listener must never break the agent
                pass

    @contextmanager
    def span(self, name, **attrs):
        step = {"step": name, "t0": time.time(), **{k: v for k, v in attrs.items() if v is not None}}
        t = tracer()
        cm = t.start_as_current_span(name) if t else None
        sp = cm.__enter__() if cm else None
        if sp is not None and self.trace_id is None:
            self.trace_id = format(sp.get_span_context().trace_id, "032x")
        self._emit({"type": "start", "step": name, **{k: v for k, v in attrs.items() if v is not None}})
        try:
            yield step
        except Exception as e:
            step["error"] = f"{type(e).__name__}: {e}"
            if sp is not None:
                sp.set_status(ot.Status(ot.StatusCode.ERROR, str(e)))
            raise
        finally:
            step["ms"] = round((time.time() - step.pop("t0")) * 1000, 1)
            if sp is not None:
                for k, v in step.items():
                    if v is not None and not k.startswith("_"):
                        sp.set_attribute(k, _clean(v))
                cm.__exit__(None, None, None)
            # keys starting with "_" (full LLM outputs, tool results) go to the live stream only,
            # so the stored trace and the OpenTelemetry spans stay small
            public = {k: v for k, v in step.items() if not k.startswith("_")}
            self._emit({"type": "end", **step})
            step.clear(); step.update(public)
            self.steps.append(step)
