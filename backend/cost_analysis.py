"""
Cost Analysis — token consumption and per-assembly economics.

Run this script directly to get a cost report from telemetry:
    python cost_analysis.py
"""

import json
import os
import pathlib
from collections import defaultdict

_TELEMETRY_DIR = pathlib.Path(
    os.environ.get("MIRUM_TELEMETRY_DIR", "/tmp/mirum/telemetry")
)
_LOG_FILE = _TELEMETRY_DIR / "critic_loop.jsonl"

# Gemini 2.5 Flash pricing
INPUT_PRICE_PER_M = 0.075   # USD per 1M input tokens
OUTPUT_PRICE_PER_M = 0.30   # USD per 1M output tokens


def compute_cost_report() -> dict:
    """Compute cost statistics from telemetry token usage records."""
    agent_stats = defaultdict(lambda: {
        "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0
    })
    assembly_costs = []  # Per-generation total cost
    current_gen_cost = 0.0

    try:
        with open(_LOG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if r.get("event_type") != "token_usage":
                    continue

                agent = r.get("agent", "unknown")
                pt = r.get("prompt_tokens", 0)
                ct = r.get("completion_tokens", 0)
                cost = r.get("estimated_cost_usd", 0.0)

                agent_stats[agent]["calls"] += 1
                agent_stats[agent]["prompt_tokens"] += pt
                agent_stats[agent]["completion_tokens"] += ct
                agent_stats[agent]["cost_usd"] += cost

    except FileNotFoundError:
        pass

    total_cost = sum(s["cost_usd"] for s in agent_stats.values())
    total_calls = sum(s["calls"] for s in agent_stats.values())

    return {
        "agent_stats": dict(agent_stats),
        "total_cost_usd": round(total_cost, 4),
        "total_llm_calls": total_calls,
        "avg_cost_per_call": round(total_cost / max(total_calls, 1), 6),
        "pricing_note": (
            f"Input: ${INPUT_PRICE_PER_M}/1M tokens, "
            f"Output: ${OUTPUT_PRICE_PER_M}/1M tokens (Gemini 2.5 Flash)"
        ),
    }


def print_cost_report() -> None:
    report = compute_cost_report()
    print(f"\n{'='*60}")
    print("Mirum Cost Analysis Report")
    print(f"{'='*60}")
    print(f"Total LLM API calls: {report['total_llm_calls']}")
    print(f"Total estimated cost: ${report['total_cost_usd']:.4f} USD")
    print(f"Avg cost per call:   ${report['avg_cost_per_call']:.6f} USD")
    print(f"Pricing: {report['pricing_note']}")
    print(f"\nBreakdown by Agent:")
    print(f"{'Agent':<24} {'Calls':>6} {'Input Tok':>10} {'Out Tok':>9} {'Cost ($)':>10}")
    print("-" * 64)
    for agent, stats in sorted(report["agent_stats"].items()):
        print(
            f"{agent:<24} {stats['calls']:>6} "
            f"{stats['prompt_tokens']:>10,} "
            f"{stats['completion_tokens']:>9,} "
            f"{stats['cost_usd']:>10.4f}"
        )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_cost_report()
