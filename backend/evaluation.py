"""
LVM Score — Language-Vision Match Evaluation.

Evaluates the semantic accuracy of generated 3D models by asking Gemini
to compare multi-view renders against the original text prompt.

Since generating renders requires OCCT visualization infrastructure not
present in all deployments, this module provides two paths:
1. Full: render the model and send images to Gemini Vision
2. Lite: send the generated CadQuery code to Gemini for code-level evaluation

The Lite path is always available and provides a useful proxy score.
"""

import asyncio
import json
import logging
import os

import google.generativeai as genai

logger = logging.getLogger("mirum.evaluation")

MODEL = "gemini-2.5-flash"

_LVM_SCORE_PROMPT = """You are evaluating a CadQuery 3D CAD script that was generated from a text prompt.

Original prompt: {prompt}

Generated CadQuery code:
{code}

Score the code on three dimensions (1–5 each):
1. Semantic accuracy: Does the code attempt to generate the described object type?
2. Feature completeness: Are the key features from the prompt likely to be present?
3. Geometric plausibility: Does the code use appropriate CadQuery patterns for this geometry?

Return JSON ONLY:
{{"semantic_accuracy": N, "feature_completeness": N, "geometric_plausibility": N,
  "lvm_score": avg, "missing_features": ["feature1", "feature2"],
  "notes": "one sentence summary"}}

Where lvm_score = (semantic_accuracy + feature_completeness + geometric_plausibility) / 3.
"""


async def compute_lvm_score(
    prompt: str,
    code: str,
    part_id: str = "",
) -> dict:
    """Compute LVM Score for a generated CadQuery script.

    Uses Gemini to evaluate semantic match between the prompt and the
    generated code. This is the 'Lite' path (code-level evaluation).

    Returns a dict with keys: semantic_accuracy, feature_completeness,
    geometric_plausibility, lvm_score, missing_features, notes.
    Returns {"lvm_score": None} on failure (non-blocking).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"lvm_score": None, "error": "No API key"}

    try:
        model = genai.GenerativeModel(model_name=MODEL)
        prompt_filled = _LVM_SCORE_PROMPT.format(
            prompt=prompt[:500],
            code=code[:3000],
        )
        response = await model.generate_content_async(
            prompt_filled,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "semantic_accuracy": {"type": "number"},
                        "feature_completeness": {"type": "number"},
                        "geometric_plausibility": {"type": "number"},
                        "lvm_score": {"type": "number"},
                        "missing_features": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "notes": {"type": "string"},
                    },
                    "required": [
                        "semantic_accuracy", "feature_completeness",
                        "geometric_plausibility", "lvm_score", "notes",
                    ],
                },
            },
        )
        result = json.loads(response.text)
        result["part_id"] = part_id
        result["evaluation_method"] = "code_lite"
        return result
    except Exception as e:
        logger.debug("LVM Score failed for %s: %s", part_id, e)
        return {"lvm_score": None, "error": str(e)}


def log_lvm_score(result: dict) -> None:
    """Write LVM score result to telemetry log."""
    try:
        from telemetry import _ensure_handler, _logger
        import time
        _ensure_handler()
        record = {
            "timestamp": time.time(),
            "event_type": "lvm_score",
            **result,
        }
        _logger.info(json.dumps(record))
    except Exception:
        pass  # Non-blocking
