"""
Secure CadQuery Script Execution Engine.

Executes AI-generated CadQuery Python scripts in an isolated subprocess
with strict timeout enforcement. This is the runtime behind the QA Inspector
(Critic Loop) — failed executions return tracebacks that get fed back to the
Machinist subagent for correction.
"""

import ast
import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile

try:
    import resource

    _HAS_RESOURCE = True
except ImportError:          # Windows — no resource module
    _HAS_RESOURCE = False

# Directory containing cad_utils.py and other backend modules.
# Added to PYTHONPATH so subprocess-executed scripts can import them.
_BACKEND_DIR = str(pathlib.Path(__file__).parent.resolve())

# ---------------------------------------------------------------------------
# AST Security Scanner
# ---------------------------------------------------------------------------
_BANNED_MODULES = {"os", "sys", "subprocess"}
_BANNED_CALLS = {"eval", "exec", "open"}
_MEM_LIMIT_BYTES = int(1.5 * 1024 * 1024 * 1024)  # 1.5 GB


def _check_ast_security(script_string: str) -> str | None:
    """Return an error message if the script imports or calls banned targets."""
    try:
        tree = ast.parse(script_string)
    except SyntaxError as exc:
        return f"SyntaxError in script: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BANNED_MODULES:
                    return "Security violation: Banned module or function detected."
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _BANNED_MODULES:
                return "Security violation: Banned module or function detected."
        elif isinstance(node, ast.Call):
            func = node.func
            # bare calls: eval(), exec(), open()
            if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                return "Security violation: Banned module or function detected."
            # attribute calls: os.system(), subprocess.run(), builtins.open()
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id in _BANNED_MODULES or (
                    func.value.id == "builtins" and func.attr == "open"
                ):
                    return "Security violation: Banned module or function detected."
    return None


def _limit_memory() -> None:
    """preexec_fn callback — cap subprocess virtual memory at 1.5 GB (POSIX only)."""
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


async def execute_cad_script(script_string: str, *, trusted: bool = False) -> dict:
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
    # --- AST security gate (skipped for internally-generated scripts) ---
    if not trusted:
        violation = _check_ast_security(script_string)
        if violation:
            return {"status": "error", "traceback": violation}

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

        run_kwargs: dict = dict(
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
        if _HAS_RESOURCE:
            run_kwargs["preexec_fn"] = _limit_memory

        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, tmp_path],
            **run_kwargs,
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
