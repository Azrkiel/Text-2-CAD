"""
Failure analysis and error attribution for the Critic Loop.

Classifies Critic Loop failures by CadQuery operation type, enabling
targeted training data collection and strategy prompt improvements.
"""

import json
import os
import pathlib
import re
from collections import Counter, defaultdict
from typing import Optional

# Operation classification patterns
# Each entry: (operation_name, [regex_patterns_in_traceback_or_code])
OPERATION_PATTERNS: list[tuple[str, list[str]]] = [
    ("extrude",    [r"\.extrude\(", r"BRep_API: command not done.*extrude"]),
    ("revolve",    [r"\.revolve\(", r"RevolutionAlgo"]),
    ("fillet",     [r"\.fillet\(", r"StdFail_NotDone", r"BRepFilletAPI"]),
    ("chamfer",    [r"\.chamfer\(", r"BRepChamfer"]),
    ("loft",       [r"\.loft\(", r"makeLoft", r"LoftAlgo", r"ValueError.*loft"]),
    ("boolean",    [r"BRepAlgoAPI", r"\.cut\(", r"\.union\(", r"\.intersect\(", r"BooleanAlgo"]),
    ("shell",      [r"\.shell\(", r"OffsetAlgo", r"BRepOffsetAPI"]),
    ("spline",     [r"\.spline\(", r"makeSpline", r"Geom_BSplineCurve"]),
    ("selector",   [r"CadQuery selector", r"No object found", r"Empty wire", r"NullShape"]),
    ("constraint", [r"assembly\.constrain", r"assembly\.solve", r"SOLVER_FAILURE"]),
    ("thread",     [r"makeThread", r"helix", r"Helix"]),
    ("text",       [r"\.text\(", r"fontsize", r"Font"]),
    ("import",     [r"importStep", r"importBrep", r"STEP"]),
    ("polarArray", [r"polarArray", r"PolarArray"]),
    ("syntax",     [r"SyntaxError", r"IndentationError", r"NameError"]),
    ("security",   [r"Security violation", r"not allowed"]),
    ("timeout",    [r"TIMEOUT", r"exceeded 30-second"]),
]


def classify_failure(traceback: str, code: str = "") -> str:
    """Classify a Critic Loop failure by CadQuery operation type.

    Checks the traceback and generated code against known error patterns.
    Returns the operation name (e.g., 'fillet', 'loft', 'selector') or
    'unknown' if no pattern matches.
    """
    combined = (traceback + "\n" + code).lower()

    for op_name, patterns in OPERATION_PATTERNS:
        if any(re.search(p, combined, re.IGNORECASE) for p in patterns):
            return op_name

    return "unknown"


def generate_failure_report(log_path: Optional[str] = None) -> dict:
    """Read telemetry JSONL and generate a failure rate report by operation.

    Returns a dict with:
    - operation_counts: {op_name: {"attempts": N, "failures": N, "rate": float}}
    - domain_counts: {domain: {"attempts": N, "failures": N}}
    - total_attempts: int
    - total_failures: int
    """
    if log_path is None:
        log_path = str(
            pathlib.Path(os.environ.get("MIRUM_TELEMETRY_DIR", "/tmp/mirum/telemetry"))
            / "critic_loop.jsonl"
        )

    op_stats: dict[str, dict] = defaultdict(lambda: {"attempts": 0, "failures": 0})
    domain_stats: dict[str, dict] = defaultdict(lambda: {"attempts": 0, "failures": 0})
    total_attempts = 0
    total_failures = 0

    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Skip assembly-level and LVM events
                if r.get("event_type") in ("assembly", "lvm_score"):
                    continue

                domain = r.get("domain_classification", "unknown")
                success = r.get("success_status", False)
                op = r.get("failure_operation", "unknown")

                total_attempts += 1
                domain_stats[domain]["attempts"] += 1

                if not success:
                    total_failures += 1
                    domain_stats[domain]["failures"] += 1
                    op_stats[op]["attempts"] += 1
                    op_stats[op]["failures"] += 1
                else:
                    # Track successes per operation too (from classified retries)
                    if op and op != "unknown":
                        op_stats[op]["attempts"] += 1

    except FileNotFoundError:
        pass

    # Compute rates
    for op, stats in op_stats.items():
        attempts = max(stats["attempts"], 1)
        stats["rate"] = round(stats["failures"] / attempts, 3)

    for domain, stats in domain_stats.items():
        attempts = max(stats["attempts"], 1)
        stats["rate"] = round(stats["failures"] / attempts, 3)

    return {
        "operation_counts": dict(op_stats),
        "domain_counts": dict(domain_stats),
        "total_attempts": total_attempts,
        "total_failures": total_failures,
        "overall_failure_rate": (
            round(total_failures / max(total_attempts, 1), 3)
        ),
    }


def print_failure_report() -> None:
    """Print a formatted failure report to stdout."""
    report = generate_failure_report()

    print(f"\n{'='*60}")
    print(f"Mirum Failure Analysis Report")
    print(f"{'='*60}")
    print(f"Total attempts: {report['total_attempts']}")
    print(f"Total failures: {report['total_failures']}")
    print(f"Overall failure rate: {report['overall_failure_rate']:.1%}")

    print(f"\nFailure by Operation Type:")
    print(f"{'Operation':<16} {'Failures':>8} {'Rate':>8}")
    print("-" * 36)
    op_sorted = sorted(
        report["operation_counts"].items(),
        key=lambda x: x[1]["failures"],
        reverse=True,
    )
    for op, stats in op_sorted:
        print(f"{op:<16} {stats['failures']:>8} {stats['rate']:>7.1%}")

    print(f"\nFailure by Domain:")
    print(f"{'Domain':<10} {'Attempts':>10} {'Failures':>10} {'Rate':>8}")
    print("-" * 44)
    for domain, stats in sorted(report["domain_counts"].items()):
        print(
            f"{domain:<10} {stats['attempts']:>10} "
            f"{stats['failures']:>10} {stats['rate']:>7.1%}"
        )

    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_failure_report()
