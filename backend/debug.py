"""
Persistent raw-error logger for text-to-CAD diagnosis.

`_safe_error` in main.py redacts filesystem paths to `<internal>` before
sending errors to the client. That's fine for clients but useless when
debugging — `[Errno 13] Permission denied: '<internal>'` doesn't tell
you *which* path was denied.

This module writes UNSANITIZED error records to a JSONL file on the
container's tmpfs so developers can read them via::

    docker compose exec -T backend cat /tmp/mirum_debug.jsonl

Each record carries the pipeline stage, the exception class and message,
the full traceback, and an optional context dict (part id, prompt, etc.).
Logging here is best-effort — any failure is swallowed so the observer
can never break the thing it's observing.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import traceback
from typing import Any

# Writable tmpfs mount inside the container. Outside a container this
# falls back to whatever /tmp resolves to on the host OS.
_DEBUG_DIR = pathlib.Path(os.environ.get("MIRUM_DEBUG_DIR", "/tmp"))
_DEBUG_LOG = _DEBUG_DIR / "mirum_debug.jsonl"

# Cap individual field sizes so one runaway traceback can't eat the
# tmpfs budget.
_MAX_TRACEBACK_LEN = 8_000
_MAX_MESSAGE_LEN = 2_000
_MAX_CONTEXT_LEN = 4_000


def log_error(
    stage: str,
    error: BaseException,
    context: dict[str, Any] | None = None,
) -> None:
    """Append one raw-error record. Never raises.

    Args:
        stage: Short pipeline-stage tag, e.g. "planner", "critic_loop",
            "export", "assembler".
        error: The caught exception. Its full traceback is captured via
            ``traceback.format_exception``.
        context: Optional extra fields (part id, attempt number, script
            excerpt). Serialized with ``default=str`` so non-JSON types
            don't blow up the writer.
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        tb_text = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[:_MAX_TRACEBACK_LEN]

        record = {
            "timestamp": time.time(),
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error)[:_MAX_MESSAGE_LEN],
            "traceback": tb_text,
            "context": _truncate_context(context or {}),
        }
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Observer must never break the pipeline.
        pass


def log_event(stage: str, message: str, context: dict[str, Any] | None = None) -> None:
    """Append a non-error diagnostic event. Never raises.

    Useful for marking pipeline stage boundaries when correlating with
    an error recorded later.
    """
    try:
        _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "stage": stage,
            "event": message[:_MAX_MESSAGE_LEN],
            "context": _truncate_context(context or {}),
        }
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _truncate_context(ctx: dict[str, Any]) -> dict[str, Any]:
    """Cap the serialized length of the context dict."""
    try:
        blob = json.dumps(ctx, default=str)
        if len(blob) <= _MAX_CONTEXT_LEN:
            return ctx
        return {"_truncated": True, "preview": blob[:_MAX_CONTEXT_LEN]}
    except Exception:
        return {"_serialization_error": True}
