"""
FastAPI Orchestration Layer for Text-to-CAD Pipeline.

Exposes the hierarchical subagent pipeline as a streaming SSE endpoint.
Each pipeline step emits a JSON progress event so the frontend can render
a live checklist.
"""

import base64
import json
import os
import uuid
from collections.abc import AsyncGenerator

import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents import (
    ScriptError,
    is_single_part,
    run_assembly_critic_loop,
    run_critic_loop,
    run_planner,
    run_single_part_export,
)
from compiler import execute_cad_script

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI(title="Text-to-CAD API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str


class RunRequest(BaseModel):
    script: str


def _event(step: str, status: str, detail: str = "") -> str:
    """Format a server-sent event line."""
    payload = {"step": step, "status": status}
    if detail:
        payload["detail"] = detail
    return f"data: {json.dumps(payload)}\n\n"


async def _pipeline(prompt: str) -> AsyncGenerator[str, None]:
    """Run the full pipeline, yielding SSE progress events."""

    # --- Step 1: Planner ---
    yield _event("planner", "running")
    try:
        manifest = await run_planner(prompt)
    except Exception as e:
        yield _event("planner", "error", str(e))
        return
    part_names = [p.part_id for p in manifest.parts]
    single = is_single_part(manifest)
    mode = "single-part" if single else f"{len(manifest.parts)}-part assembly"
    yield _event("planner", "done", f"Planned {mode}: {', '.join(part_names)}")

    # --- Step 2: Machinist + Critic Loop ---
    output_filename = uuid.uuid4().hex + ".glb"

    if single:
        # Single-part fast path: one machinist call, direct export, no assembler
        part = manifest.parts[0]
        label = f"machinist:{part.part_id}"
        yield _event(label, "running", "Manufacturing single part")
        try:
            code, _ = await run_critic_loop(part)
        except ScriptError as e:
            yield _event(label, "error", str(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": str(e),
                "script": e.script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        except RuntimeError as e:
            yield _event(label, "error", str(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": str(e),
                "script": getattr(e, "script", ""),
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        yield _event(label, "done", f"Part '{part.part_id}' validated")

        # Direct export — no assembler needed
        yield _event("export", "running", "Exporting to .glb")
        final_script = await run_single_part_export(code, output_filename)
        yield _event("script", "done", final_script)
        try:
            result = await execute_cad_script(final_script)
        except Exception as e:
            result = {"status": "error", "traceback": str(e)}
        if result["status"] == "error":
            yield _event("export", "error", result["traceback"])
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": result["traceback"],
                "script": final_script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
    else:
        # Multi-part path: concurrent manufacturing + assembler
        import asyncio

        # Launch all machinist jobs concurrently
        parts = manifest.parts
        labels = {p.part_id: f"machinist:{p.part_id}" for p in parts}

        # Emit "running" for all parts
        for i, part in enumerate(parts, 1):
            yield _event(labels[part.part_id], "running", f"Manufacturing part {i}/{len(parts)}")

        # Run concurrently with semaphore-based rate limiting
        async def _build_part(part):
            code, step_path = await run_critic_loop(part)
            return part.part_id, code, step_path

        tasks = [_build_part(part) for part in parts]
        part_scripts: dict[str, str] = {}
        step_files: dict[str, str] = {}
        for coro in asyncio.as_completed(tasks):
            try:
                part_id, code, step_path = await coro
                part_scripts[part_id] = code
                step_files[part_id] = step_path
                yield _event(labels[part_id], "done", f"Part '{part_id}' validated")
            except ScriptError as e:
                yield _event("machinist", "error", str(e))
                error_complete = {
                    "step": "complete",
                    "status": "error",
                    "message": str(e),
                    "script": e.script,
                }
                yield f"data: {json.dumps(error_complete)}\n\n"
                return
            except RuntimeError as e:
                yield _event("machinist", "error", str(e))
                error_complete = {
                    "step": "complete",
                    "status": "error",
                    "message": str(e),
                    "script": getattr(e, "script", ""),
                }
                yield f"data: {json.dumps(error_complete)}\n\n"
                return

        # Deterministic assembler (no LLM, no retries)
        yield _event("assembler", "running", "Assembling from pre-compiled .step files")
        try:
            final_script = await run_assembly_critic_loop(manifest, step_files)
            final_script = final_script.replace("output.glb", output_filename)
        except ScriptError as e:
            yield _event("assembler", "error", str(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": str(e),
                "script": e.script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        except RuntimeError as e:
            yield _event("assembler", "error", str(e))
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": str(e),
                "script": getattr(e, "script", ""),
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return

        yield _event("script", "done", final_script)
        try:
            result = await execute_cad_script(final_script)
        except Exception as e:
            result = {"status": "error", "traceback": str(e)}
        if result["status"] == "error":
            yield _event("assembler", "error", result["traceback"])
            error_complete = {
                "step": "complete",
                "status": "error",
                "message": result["traceback"],
                "script": final_script,
            }
            yield f"data: {json.dumps(error_complete)}\n\n"
            return
        yield _event("assembler", "done")

    # --- Final: Check file and deliver ---
    if not os.path.exists(output_filename):
        msg = "Script succeeded but .glb file was not created."
        yield _event("export", "error", msg)
        error_complete = {
            "step": "complete",
            "status": "error",
            "message": msg,
            "script": final_script,
        }
        yield f"data: {json.dumps(error_complete)}\n\n"
        return

    if single:
        yield _event("export", "done")

    with open(output_filename, "rb") as f:
        glb_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(output_filename)

    complete_payload = {
        "step": "complete",
        "status": "done",
        "glb": glb_b64,
        "script": final_script,
    }
    yield f"data: {json.dumps(complete_payload)}\n\n"


@app.post("/generate")
async def generate(request: GenerateRequest):
    return StreamingResponse(
        _pipeline(request.prompt),
        media_type="text/event-stream",
    )


@app.post("/run")
async def run_script(request: RunRequest):
    """Execute a CadQuery script directly and return the GLB model."""
    output_filename = uuid.uuid4().hex + ".glb"
    script = request.script.replace("output.glb", output_filename)

    result = await execute_cad_script(script)
    if result["status"] == "error":
        return {"status": "error", "detail": result["traceback"]}

    if not os.path.exists(output_filename):
        return {"status": "error", "detail": "Script ran but no .glb file was created."}

    with open(output_filename, "rb") as f:
        glb_b64 = base64.b64encode(f.read()).decode("utf-8")
    os.remove(output_filename)

    return {"status": "ok", "glb": glb_b64}
