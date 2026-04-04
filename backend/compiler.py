"""
Secure CadQuery Script Execution Engine.

Executes AI-generated CadQuery Python scripts in an isolated subprocess
with strict timeout enforcement. This is the runtime behind the QA Inspector
(Critic Loop) — failed executions return tracebacks that get fed back to the
Machinist subagent for correction.
"""

import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile

# Directory containing cad_utils.py and other backend modules.
# Added to PYTHONPATH so subprocess-executed scripts can import them.
_BACKEND_DIR = str(pathlib.Path(__file__).parent.resolve())


async def execute_cad_script(script_string: str) -> dict:
    """Execute a CadQuery Python script in a sandboxed subprocess.

    The script is written to a temporary file and run as a separate process
    to isolate the main application from crashes, infinite loops, or unsafe
    operations in AI-generated code.

    Args:
        script_string: Complete, self-contained Python/CadQuery source code.

    Returns:
        On success: {"status": "success", "output": <stdout>}
        On failure: {"status": "error", "traceback": <stderr>}
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as tmp:
            tmp.write(script_string)
            tmp_path = tmp.name

        env = os.environ.copy()
        env["PYTHONPATH"] = _BACKEND_DIR + os.pathsep + env.get("PYTHONPATH", "")

        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        if result.returncode != 0:
            return {"status": "error", "traceback": result.stderr}

        return {"status": "success", "output": result.stdout}

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "traceback": "TIMEOUT: Script exceeded 30-second execution limit.",
        }
    except Exception as e:
        return {
            "status": "error",
            "traceback": f"Execution error: {e}",
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
