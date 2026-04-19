"""
T2-05: Multi-Level Prompt Stratification for Machinist SFT Dataset.

Takes existing (prompt, CadQuery code) training pairs and generates four
complexity variants of each prompt using Gemini:
  - Abstract   : 1 sentence, no dimensions, just object type
  - Beginner   : key geometric relationships, no dimensions
  - Intermediate: the original (with key dimensions)
  - Expert     : adds material, tolerances, fit specs, surface finish

This quadruples dataset size at minimal cost and teaches the fine-tuned
Machinist to handle the full spectrum of real-world prompt styles.

Usage:
    python data_pipeline/prompt_stratification.py \
        --input  data/seed_pairs.jsonl \
        --output data/machinist_sft_stratified.jsonl \
        --max-workers 4

Input JSONL format  : {"description": "...", "code": "...", "domain": "A"}
Output JSONL format : {"prompt": "...", "code": "...", "domain": "A",
                       "complexity_level": "abstract|beginner|intermediate|expert",
                       "source": "stratified"}

Dependencies:
    pip install google-generativeai tqdm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("mirum.stratification")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

STRATIFICATION_SYSTEM = (
    "You are a mechanical engineering documentation specialist. "
    "Your job is to rephrase a 3D CAD part description at four different "
    "specificity levels so a fine-tuned model can learn to handle any style "
    "of user input. All four descriptions must accurately describe the same part."
)

STRATIFICATION_TEMPLATE = """\
Given a CadQuery script that generates a 3D mechanical part, and an \
intermediate-level description of that part, generate FOUR descriptions \
at different specificity levels.

RULES:
  - Abstract: 1 sentence, object type only, no dimensions, no features
  - Beginner: 1-2 sentences, key shape and main features, no numeric dimensions
  - Intermediate: the original description (lightly cleaned if needed)
  - Expert: adds material specification, tolerances (e.g. h6/H7 fits), \
surface finish callouts (Ra values), and engineering-specific feature names

Part description:
{description}

CadQuery code (for context only — do not quote it in the descriptions):
{code_excerpt}

Return JSON: {{"abstract": "...", "beginner": "...", "intermediate": "...", \
"expert": "..."}}
"""


def stratify_pair(
    description: str,
    code: str,
    model_name: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> dict[str, str] | None:
    """Call Gemini to generate four complexity variants of a prompt.

    Returns a dict with keys abstract/beginner/intermediate/expert,
    or None on failure.
    """
    try:
        import google.generativeai as genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=key)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=STRATIFICATION_SYSTEM,
        )

        # Truncate code to save tokens — first 60 lines is enough for context
        code_excerpt = "\n".join(code.splitlines()[:60])

        prompt = STRATIFICATION_TEMPLATE.format(
            description=description[:800],
            code_excerpt=code_excerpt,
        )

        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "abstract":     {"type": "string"},
                        "beginner":     {"type": "string"},
                        "intermediate": {"type": "string"},
                        "expert":       {"type": "string"},
                    },
                    "required": ["abstract", "beginner", "intermediate", "expert"],
                },
            },
        )
        return json.loads(response.text)
    except Exception as exc:
        logger.warning("Stratification failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_seed_pairs(path: str) -> list[dict]:
    """Load seed (description, code, domain) pairs from a JSONL file."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "description" not in rec or "code" not in rec:
                    logger.warning("Line %d missing required fields — skipped", line_no)
                    continue
                pairs.append(rec)
            except json.JSONDecodeError as exc:
                logger.warning("JSON parse error on line %d: %s", line_no, exc)
    return pairs


def stratify_dataset(
    input_path: str,
    output_path: str,
    max_workers: int = 4,
    api_key: str | None = None,
    delay_between_calls: float = 0.5,
) -> dict:
    """Run stratification over the full input dataset.

    For each input pair, generates 4 output records (one per complexity level).
    Appends results incrementally to output_path so progress is not lost on crash.

    Returns a summary dict with counts.
    """
    pairs = load_seed_pairs(input_path)
    logger.info("Loaded %d seed pairs from %s", len(pairs), input_path)

    # Track already-processed descriptions (for resume)
    processed: set[str] = set()
    out_path = Path(output_path)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed.add(rec.get("_source_description_hash", ""))
                except json.JSONDecodeError:
                    pass
        logger.info("Resuming — %d pairs already processed", len(processed))

    pending = [
        p for p in pairs
        if str(hash(p["description"])) not in processed
    ]
    logger.info("%d pairs to process", len(pending))

    total_written = 0
    total_failed = 0

    with open(out_path, "a", encoding="utf-8") as out_f:
        def _process(pair: dict):
            variants = stratify_pair(
                pair["description"], pair["code"], api_key=api_key
            )
            return pair, variants

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, pair): pair for pair in pending}
            try:
                from tqdm import tqdm
                it = tqdm(as_completed(futures), total=len(futures), desc="Stratifying")
            except ImportError:
                it = as_completed(futures)

            for future in it:
                pair, variants = future.result()
                if variants is None:
                    total_failed += 1
                    continue

                src_hash = str(hash(pair["description"]))
                domain = pair.get("domain", "A")
                code = pair["code"]
                levels = [
                    ("abstract", variants["abstract"]),
                    ("beginner", variants["beginner"]),
                    ("intermediate", variants["intermediate"]),
                    ("expert", variants["expert"]),
                ]
                for level_name, prompt_text in levels:
                    record = {
                        "prompt": prompt_text,
                        "code": code,
                        "domain": domain,
                        "complexity_level": level_name,
                        "source": "stratified",
                        "_source_description_hash": src_hash,
                    }
                    out_f.write(json.dumps(record) + "\n")
                    total_written += 1
                out_f.flush()
                time.sleep(delay_between_calls)

    summary = {
        "input_pairs": len(pairs),
        "processed": len(pending) - total_failed,
        "failed": total_failed,
        "output_records": total_written,
        "output_path": str(out_path.resolve()),
    }
    logger.info("Done: %s", summary)
    return summary


# ---------------------------------------------------------------------------
# Telemetry mining (builds seed pairs from existing successful critic runs)
# ---------------------------------------------------------------------------

def mine_telemetry_pairs(
    telemetry_jsonl: str = "/tmp/mirum/telemetry/critic_loop.jsonl",
    min_attempts_filter: int = 1,
) -> list[dict]:
    """Extract successful (description, code, domain) pairs from telemetry.

    Only includes records where success=True and attempt==1 (first-try
    successes are highest quality; retry successes are acceptable but noisier).
    """
    pairs = []
    seen: set[str] = set()
    try:
        with open(telemetry_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not rec.get("success"):
                    continue
                if rec.get("attempt", 99) > min_attempts_filter:
                    continue
                code = rec.get("code", "")
                domain = rec.get("domain", "A")
                part_id = rec.get("part_id", "")
                if not code or not part_id:
                    continue
                # Use part_id as description proxy (real descriptions not logged yet)
                desc = part_id.replace("_", " ")
                key = f"{domain}::{desc[:80]}"
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"description": desc, "code": code, "domain": domain})
    except FileNotFoundError:
        logger.warning("Telemetry file not found: %s", telemetry_jsonl)
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate multi-level prompt variants for Machinist SFT dataset"
    )
    p.add_argument("--input", required=True, help="Input JSONL with seed pairs")
    p.add_argument("--output", required=True, help="Output JSONL path")
    p.add_argument("--max-workers", type=int, default=4, help="Gemini concurrency")
    p.add_argument("--api-key", default=None, help="Gemini API key (default: env var)")
    p.add_argument("--delay", type=float, default=0.3,
                   help="Seconds between API calls per worker")
    p.add_argument("--mine-telemetry", action="store_true",
                   help="Mine telemetry JSONL first; merge with --input")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = stratify_dataset(
        input_path=args.input,
        output_path=args.output,
        max_workers=args.max_workers,
        api_key=args.api_key,
        delay_between_calls=args.delay,
    )
    print(json.dumps(summary, indent=2))
