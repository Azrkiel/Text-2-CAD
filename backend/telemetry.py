"""
Telemetry data capture for the Text-to-CAD Critic Loop.

Logs every Machinist attempt as a structured JSONL record for future
fine-tuning and failure-mode analysis. Each record captures the full
context of a code generation attempt: the part being built, the domain
classification, which attempt number it was, the generated code, any
error traceback, and whether the attempt succeeded.
"""

import json
import pathlib
import time
from typing import Optional

_LOG_DIR = pathlib.Path(__file__).parent / "telemetry_logs"
_LOG_FILE = _LOG_DIR / "critic_loop.jsonl"


def log_attempt(
    part_id: str,
    domain: str,
    attempt: int,
    code: str,
    error: Optional[str],
    success: bool,
) -> None:
    """Append a single critic-loop attempt record to the JSONL log."""
    _LOG_DIR.mkdir(exist_ok=True)

    record = {
        "timestamp": time.time(),
        "part_id": part_id,
        "domain_classification": domain,
        "attempt_number": attempt,
        "generated_code": code,
        "traceback_error": error,
        "success_status": success,
    }

    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
