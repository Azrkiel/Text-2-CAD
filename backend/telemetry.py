"""
Telemetry data capture for the Text-to-CAD Critic Loop.

Logs every Machinist attempt as a structured JSONL record for future
fine-tuning and failure-mode analysis. Each record captures the full
context of a code generation attempt: the part being built, the domain
classification, which attempt number it was, the generated code, any
error traceback, and whether the attempt succeeded.

Log location
------------
Defaults to ``/tmp/mirum/telemetry`` because the previous location
(``backend/telemetry_logs`` — a Docker named volume) was created by the
engine as ``root:root`` while the app runs as ``caduser``, causing every
write to raise ``PermissionError``. The /tmp path is the container's
tmpfs and is guaranteed writable. Override with ``MIRUM_TELEMETRY_DIR``
if a persistent location is available.

Failures are swallowed
----------------------
Telemetry is analytics-only; a broken logger must never break the
pipeline. If the handler can't be created or the write fails, the
attempt is silently dropped and the bug is mirrored into
``debug.log_error`` so it stays diagnosable.
"""

import json
import logging
import os
import pathlib
import time
from logging.handlers import RotatingFileHandler
from typing import Optional

from debug import log_error as _debug_log_error

_LOG_DIR = pathlib.Path(
    os.environ.get("MIRUM_TELEMETRY_DIR", "/tmp/mirum/telemetry")
)
_LOG_FILE = _LOG_DIR / "critic_loop.jsonl"

_MAX_BYTES = 50 * 1024 * 1024   # 50 MB per file
_BACKUP_COUNT = 5                # keep 5 rotated files
_MAX_CODE_LEN = 10_000           # truncate generated_code per record
_MAX_ERROR_LEN = 2_000           # truncate traceback per record

_logger = logging.getLogger("mirum.telemetry")
_handler_initialized = False


def _ensure_handler() -> None:
    global _handler_initialized
    if _handler_initialized:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        str(_LOG_FILE),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    _handler_initialized = True


def log_attempt(
    part_id: str,
    domain: str,
    attempt: int,
    code: str,
    error: Optional[str],
    success: bool,
    failure_operation: Optional[str] = None,
    solver_success: Optional[bool] = None,
    interpenetration_pairs: Optional[list] = None,
) -> None:
    """Append a single critic-loop attempt record to the JSONL log.

    Best-effort: all failures are caught and forwarded to the debug log
    so broken telemetry can never break the pipeline.
    """
    try:
        _ensure_handler()
        record = {
            "timestamp": time.time(),
            "part_id": part_id,
            "domain_classification": domain,
            "attempt_number": attempt,
            "generated_code": code[:_MAX_CODE_LEN],
            "traceback_error": error[:_MAX_ERROR_LEN] if error else None,
            "success_status": success,
        }
        if failure_operation is not None:
            record["failure_operation"] = failure_operation
        if solver_success is not None:
            record["solver_success"] = solver_success
        if interpenetration_pairs is not None:
            record["interpenetration_pairs"] = interpenetration_pairs
        _logger.info(json.dumps(record))
    except Exception as exc:
        _debug_log_error(
            "telemetry",
            exc,
            context={"part_id": part_id, "attempt": attempt, "log_dir": str(_LOG_DIR)},
        )


def log_assembly_event(
    assembly_name: str,
    solver_success: bool,
    constraint_mode: bool,
    interpenetration_pairs: list,
    part_count: int,
) -> None:
    """Log an assembly-level event (solver result, interpenetration)."""
    try:
        _ensure_handler()
        record = {
            "timestamp": time.time(),
            "event_type": "assembly",
            "assembly_name": assembly_name,
            "solver_success": solver_success,
            "constraint_mode": constraint_mode,
            "interpenetration_pairs": [list(p) for p in interpenetration_pairs],
            "part_count": part_count,
        }
        _logger.info(json.dumps(record))
    except Exception as exc:
        _debug_log_error("telemetry", exc, context={"assembly_name": assembly_name})


def log_token_usage(
    agent: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Log LLM token consumption for cost analysis."""
    # Gemini 2.5 Flash pricing (approximate, as of 2026-04):
    # Input: ~$0.075/1M tokens, Output: ~$0.30/1M tokens
    INPUT_PRICE_PER_M = 0.075
    OUTPUT_PRICE_PER_M = 0.30
    estimated_cost_usd = (
        prompt_tokens * INPUT_PRICE_PER_M / 1_000_000
        + completion_tokens * OUTPUT_PRICE_PER_M / 1_000_000
    )
    try:
        _ensure_handler()
        record = {
            "timestamp": time.time(),
            "event_type": "token_usage",
            "agent": agent,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        }
        _logger.info(json.dumps(record))
    except Exception as exc:
        _debug_log_error("telemetry", exc, context={"agent": agent})
