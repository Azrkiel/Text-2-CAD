"""
Hierarchical Subagent Orchestrator for Text-to-CAD Pipeline.

This module implements the "Chief Engineer" router — a custom Python
orchestrator that makes isolated calls to Gemini 2.5 Flash.
Each function represents a distinct subagent with its own system prompt
and context boundary, preventing spatial hallucinations through strict
context isolation.

Subagent Flow:
  User Prompt → Planner → [Machinist + Critic Loop] × N (concurrent) → Assembler → .glb
"""

import asyncio
import json
import re
from typing import Any

import google.generativeai as genai

from classifier import classify_part
from compiler import execute_cad_script
from schemas import AssemblyManifest, PartDefinition
from strategies import get_strategy

MODEL = "gemini-2.5-flash"


class ScriptError(RuntimeError):
    """RuntimeError subclass that carries the last attempted CadQuery script."""

    def __init__(self, message: str, script: str = ""):
        super().__init__(message)
        self.script = script

# Max concurrent Machinist calls (respects Gemini rate limits)
_MACHINIST_SEMAPHORE = asyncio.Semaphore(3)

# Fields that Gemini's structured output schema does NOT support.
_UNSUPPORTED_SCHEMA_KEYS = {"title", "default", "minItems", "maxItems", "minimum", "maximum", "$defs"}


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip JSON Schema fields unsupported by the Gemini API."""
    cleaned = {}
    for key, value in schema.items():
        if key in _UNSUPPORTED_SCHEMA_KEYS:
            continue
        if isinstance(value, dict):
            cleaned[key] = _sanitize_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _sanitize_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _get_gemini_schema() -> dict[str, Any]:
    """Generate a Gemini-compatible JSON schema from AssemblyManifest."""
    raw = AssemblyManifest.model_json_schema()
    defs = raw.pop("$defs", {})
    raw_json = json.dumps(raw)
    for def_name, def_schema in defs.items():
        ref_str = f'{{"$ref": "#/$defs/{def_name}"}}'
        raw_json = raw_json.replace(ref_str, json.dumps(def_schema))
    resolved = json.loads(raw_json)
    return _sanitize_schema(resolved)


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


async def run_planner(user_prompt: str) -> AssemblyManifest:
    """Subagent 1: The Draftsman.

    Converts a natural-language mechanical request into a structured
    AssemblyManifest using Gemini's structured output mode.
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=(
            "You are the Chief Draftsman for a mechanical CAD system. "
            "Convert the user's mechanical request into a strict AssemblyManifest JSON. "
            "Follow ALL anchor_tag and mating_rule constraints defined in the schema. "
            "RULES: "
            "(1) Every part must be self-contained and modeled at the origin. "
            "(2) anchor_tags MUST be standard CadQuery directional selectors ONLY: "
            "'>Z' (top), '<Z' (bottom), '>X' (right), '<X' (left), '>Y' (front), '<Y' (back). "
            "NEVER use custom named anchors like 'face_top_horizontal' or 'hole_center'. "
            "NEVER use coordinate tuples. "
            "(3) mating_rules must reference only existing part_ids and anchor_tags. "
            "(4) The assembly graph must be fully connected — no orphan parts. "
            "(5) List parts in logical build order: base/foundation first, fasteners last. "
            "(6) All dimensions must be in millimeters. "
            "\n"
            "SINGLE-PART vs MULTI-PART DETECTION: "
            "If the user describes a SINGLE object (e.g., 'a gear', 'a bolt', 'a bracket'), "
            "create EXACTLY ONE PartDefinition. For a single-part assembly, create one "
            "mating_rule where source and target are the SAME part_id (self-reference), "
            "with translation '0, 0, 0' and clearance 0.0. This signals the pipeline "
            "to skip the assembly step and export the part directly. "
            "\n"
            "MULTI-PART PLANNING RULES: "
            "(A) You MUST break down every multi-part assembly into DISTINCT, INDIVIDUALLY "
            "NAMED parts. Do NOT group identical parts into a single entry. For example, a "
            "table must have 'tabletop', 'leg_1', 'leg_2', 'leg_3', 'leg_4' — five "
            "separate PartDefinition entries, NOT 'tabletop' and 'legs'. Even if parts "
            "are geometrically identical, each instance must be its own entry with a "
            "unique part_id. "
            "(B) Each MatingRule's 'translation' field MUST contain the precise absolute "
            "3D coordinate (X, Y, Z) in millimeters for where to place the target part. "
            "The base/first part is always at '0, 0, 0'."
        ),
    )

    response = await model.generate_content_async(
        user_prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": _get_gemini_schema(),
        },
    )

    return AssemblyManifest.model_validate_json(response.text)


def is_single_part(manifest: AssemblyManifest) -> bool:
    """Check if the manifest describes a single part (no real assembly needed)."""
    if len(manifest.parts) == 1:
        return True
    return False


async def run_machinist(part_def: PartDefinition, error_context: str = "") -> str:
    """Subagent 2: The Machinist.

    Generates isolated CadQuery Python code for a single part.
    Classifies the part into a geometric domain first, then applies the
    domain-specific strategy prompt.
    """
    classification = await classify_part(part_def.description)
    domain = classification.get("domain", "A")
    system_prompt = get_strategy(domain)

    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=system_prompt,
    )

    prompt = (
        f"Generate CadQuery code for this part:\n"
        f"Part ID: {part_def.part_id}\n"
        f"Description: {part_def.description}\n"
        f"Anchor Tags: {part_def.anchor_tags}\n"
    )

    if error_context:
        prompt += (
            f"\nYour previous attempt FAILED with this error:\n"
            f"---\n{error_context}\n---\n"
            f"Fix the CadQuery code to resolve this error. "
            f"Return the complete corrected script."
        )

    response = await model.generate_content_async(prompt)
    return _strip_markdown_fences(response.text)


async def run_critic_loop(part_def: PartDefinition, max_retries: int = 3) -> str:
    """Subagent 3: The QA Inspector (Critic Loop).

    Runs the Machinist's code via subprocess with concurrency control.
    On failure, feeds the traceback back for self-correction.
    """
    async with _MACHINIST_SEMAPHORE:
        error_context = ""

        for attempt in range(1, max_retries + 1):
            code = await run_machinist(part_def, error_context=error_context)
            result = await execute_cad_script(code)

            if result["status"] == "success":
                return code

            error_context = result["traceback"]
            if attempt == max_retries:
                raise ScriptError(
                    f"Part '{part_def.part_id}' failed after {max_retries} attempts. "
                    f"Last error:\n{error_context}",
                    script=code,
                )

        raise ScriptError(f"Part '{part_def.part_id}' failed unexpectedly.")


async def run_machinist_batch(parts: list[PartDefinition]) -> dict[str, str]:
    """Run all Machinist critic loops concurrently with rate limiting.

    Deduplicates parts with identical descriptions: calls the Machinist
    only once per unique description and reuses the cached code for all
    identical parts.
    """
    # Group parts by description for deduplication
    desc_to_parts: dict[str, list[PartDefinition]] = {}
    for part in parts:
        desc_to_parts.setdefault(part.description, []).append(part)

    # Build one representative per unique description
    unique_parts = [group[0] for group in desc_to_parts.values()]

    tasks = [run_critic_loop(part) for part in unique_parts]
    results = await asyncio.gather(*tasks)

    # Map representative description -> generated code
    desc_to_code = {
        part.description: code for part, code in zip(unique_parts, results)
    }

    # Fan out cached code to all parts sharing the same description
    return {part.part_id: desc_to_code[part.description] for part in parts}


async def run_single_part_export(part_script: str, output_filename: str) -> str:
    """Generate a simple export script for single-part models (no assembly)."""
    return (
        f"{part_script}\n\n"
        f"import cadquery as cq\n"
        f"assembly = cq.Assembly()\n"
        f"assembly.add(result, name='part')\n"
        f"assembly.save('{output_filename}')\n"
    )


async def run_assembler(
    manifest: AssemblyManifest, part_scripts: dict[str, str],
    error_context: str = "",
) -> str:
    """Subagent 4: The Assembler.

    Takes all validated part scripts and the mating rules, then writes a
    single final CadQuery script that assembles everything and exports to .glb.
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=(
            "You are the Assembler for a mechanical CAD system. Your job is to "
            "write a single, final CadQuery Python script that combines multiple "
            "individual parts into one assembly using cq.Assembly(). "
            "\n"
            "RULES: "
            "(1) Import cadquery as cq at the top. "
            "(2) Paste each part's code inline (do NOT use file imports). Each "
            "part script defines a variable named 'result'. Rename each to a "
            "unique variable (e.g., base_plate, leg_1) immediately after pasting. "
            "(3) Create the assembly: assembly = cq.Assembly() "
            "(4) Add each part using the EXACT translation coordinates from the "
            "mating rules. The 'translation' field contains 'X, Y, Z' values. "
            "Parse them and use: "
            "assembly.add(part_var, name='part_id', loc=cq.Location(cq.Vector(X, Y, Z))) "
            "The base/first part is always at (0, 0, 0). "
            "\n"
            "CRITICAL — DO NOT use assembly.constrain() or assembly.solve(). "
            "DO NOT try to compute positions yourself. Use ONLY the translation "
            "coordinates provided in the mating rules. They are pre-computed. "
            "\n"
            "(5) End the script with: assembly.save('output.glb') "
            "(6) Return ONLY raw executable Python. No markdown, no code fences. "
            "(7) If there is only ONE part, simply create the assembly, add it at "
            "the origin, and save."
        ),
    )

    parts_section = ""
    for part_id, code in part_scripts.items():
        parts_section += f"\n--- Part: {part_id} ---\n{code}\n"

    rules_section = ""
    for rule in manifest.mating_rules:
        rules_section += (
            f"  - Place '{rule.target_part_id}' at translation ({rule.translation}) "
            f"[mating {rule.source_part_id}@{rule.source_anchor} "
            f"to {rule.target_part_id}@{rule.target_anchor}, "
            f"clearance: {rule.clearance}mm]\n"
        )

    prompt = (
        f"Assembly Name: {manifest.assembly_name}\n\n"
        f"== VALIDATED PART SCRIPTS ==\n{parts_section}\n"
        f"== MATING RULES ==\n{rules_section}\n"
        f"Write the final assembly script."
    )

    if error_context:
        prompt += (
            f"\nYour previous assembly script FAILED with this error:\n"
            f"---\n{error_context}\n---\n"
            f"Fix the script to resolve this error. "
            f"Return the complete corrected script."
        )

    response = await model.generate_content_async(prompt)
    return _strip_markdown_fences(response.text)


async def run_assembly_critic_loop(
    manifest: AssemblyManifest, part_scripts: dict[str, str], max_retries: int = 3,
) -> str:
    """Critic loop for the Assembler — retries on execution failure."""
    error_context = ""

    for attempt in range(1, max_retries + 1):
        code = await run_assembler(manifest, part_scripts, error_context=error_context)
        result = await execute_cad_script(code)

        if result["status"] == "success":
            return code

        error_context = result["traceback"]
        if attempt == max_retries:
            raise ScriptError(
                f"Assembly failed after {max_retries} attempts. "
                f"Last error:\n{error_context}",
                script=code,
            )

    raise ScriptError("Assembly failed unexpectedly.")
