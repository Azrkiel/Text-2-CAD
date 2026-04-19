"""
Secure CadQuery Script Execution Engine.

Executes AI-generated CadQuery Python scripts in an isolated subprocess
with strict timeout enforcement. This is the runtime behind the QA Inspector
(Critic Loop) — failed executions return tracebacks that get fed back to the
Machinist subagent for correction.
"""

import ast
import asyncio
import logging
import os
import pathlib
import signal
import subprocess
import sys

try:
    import resource

    _HAS_RESOURCE = True
except ImportError:          # Windows — no resource module
    _HAS_RESOURCE = False
    logging.getLogger("mirum.compiler").warning(
        "Running on Windows: subprocess memory limits are NOT enforced. "
        "Production deployments MUST use Linux/Docker."
    )

# Directory containing cad_utils.py and other backend modules.
# Added to PYTHONPATH so subprocess-executed scripts can import them.
_BACKEND_DIR = str(pathlib.Path(__file__).parent.resolve())

# Writable working directory for subprocess-executed scripts. The host
# `/app` bind mount is read-only (see docker-compose.yml), and CadQuery /
# OCCT will happily drop temp files like `.stepcode.log` or
# `SaveToFile.tmp` into cwd when saving, which would otherwise raise
# PermissionError before the script's own save call even runs. Routing
# cwd to the tmpfs sidesteps all of that.
_WORK_DIR = pathlib.Path(os.environ.get("MIRUM_WORK_DIR", "/tmp/mirum/work"))
try:
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Fall back to the parent's cwd if even /tmp is unwritable — better
    # to let the subprocess fail loudly than to crash module import.
    pass

# ---------------------------------------------------------------------------
# AST Security Scanner (Allowlist Architecture)
# ---------------------------------------------------------------------------
_ALLOWED_MODULES = {"cadquery", "cq", "math", "cad_utils"}
_BANNED_BUILTINS = {
    "eval", "exec", "open", "compile", "__import__", "getattr", "setattr",
    "delattr", "globals", "locals", "vars", "breakpoint", "input",
    "memoryview", "type",
}
_BANNED_DUNDERS = {
    "__import__", "__subclasses__", "__globals__", "__builtins__",
    "__loader__", "__spec__", "__code__",
}
_MEM_LIMIT_BYTES = int(1.5 * 1024 * 1024 * 1024)  # 1.5 GB


def _check_ast_security(script_string: str) -> str | None:
    """Return an error message if the script imports or calls banned targets.

    Uses an ALLOWLIST for imports: only modules in _ALLOWED_MODULES may be
    imported.  All other modules (os, sys, subprocess, importlib, ctypes,
    pickle, socket, shutil, pathlib, etc.) are rejected automatically.
    """
    try:
        tree = ast.parse(script_string)
    except SyntaxError as exc:
        return f"SyntaxError in script: {exc}"

    for node in ast.walk(tree):
        # --- Import allowlist ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    return (
                        f"Security violation: import of '{alias.name}' is not "
                        f"allowed. Permitted modules: {sorted(_ALLOWED_MODULES)}."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_MODULES:
                    return (
                        f"Security violation: import from '{node.module}' is "
                        f"not allowed. Permitted modules: {sorted(_ALLOWED_MODULES)}."
                    )

        # --- Banned builtin calls ---
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_BUILTINS:
                return (
                    f"Security violation: call to '{func.id}()' is banned."
                )

        # --- Dangerous dunder attribute access ---
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_DUNDERS:
            return (
                f"Security violation: access to '{node.attr}' is banned."
            )

    return None


def _preexec_sandbox() -> None:
    """preexec_fn — new process group + memory cap (POSIX only)."""
    os.setpgrp()
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))


async def execute_cad_script(script_string: str) -> dict:
    """Execute a CadQuery Python script in a sandboxed subprocess.

    The script is piped to `python -` via stdin rather than written to a
    temp file. This avoids an entire class of Windows permission errors:
    on-access AV scanners (Avast, Defender) briefly lock freshly-written
    .py files, causing `PermissionError: [Errno 13]` when either the
    parent closes the temp file or the child tries to open it. With
    stdin delivery there is no file for AV to touch.

    Args:
        script_string: Complete, self-contained Python/CadQuery source code.

    Returns:
        On success: {"status": "success", "output": <stdout>}
        On failure: {"status": "error", "traceback": <stderr>}
    """
    # --- AST security gate (always enforced) ---
    violation = _check_ast_security(script_string)
    if violation:
        return {"status": "error", "traceback": violation}

    try:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": _BACKEND_DIR,
            "HOME": os.environ.get("HOME", os.environ.get("USERPROFILE", "/tmp")),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
        }
        # Windows-specific env vars the Python runtime and its deps need.
        # ezdxf (a CadQuery dep) calls pathlib.Path("~").expanduser() at import
        # time; on Windows that checks USERPROFILE / HOMEDRIVE+HOMEPATH (NOT
        # HOME), so without these the subprocess crashes before running user
        # code. Inherit conditionally — only if set in the parent env.
        for _key in (
            "SYSTEMROOT", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
            "TEMP", "TMP", "APPDATA", "LOCALAPPDATA",
        ):
            _val = os.environ.get(_key)
            if _val:
                env[_key] = _val

        popen_kwargs: dict = dict(
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            # Run inside a writable tmpfs location so any implicit
            # relative-path writes (intermediate STEP logs, OCCT temp
            # files, scripts that still use 'output.glb' etc.) land
            # somewhere the unprivileged caduser can actually write.
            cwd=str(_WORK_DIR) if _WORK_DIR.exists() else None,
        )
        if _HAS_RESOURCE:
            popen_kwargs["preexec_fn"] = _preexec_sandbox
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        def _run() -> tuple[int, str, str]:
            # `python -` reads the program from stdin. Tracebacks show the
            # pseudo-filename "<stdin>", which main.py's _safe_error already
            # collapses via its `File "..."` regex.
            proc = subprocess.Popen([sys.executable, "-"], **popen_kwargs)
            try:
                stdout, stderr = proc.communicate(
                    input=script_string, timeout=30
                )
            except subprocess.TimeoutExpired:
                # Kill the entire process group, not just the parent
                if _HAS_RESOURCE:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except OSError:
                        proc.kill()
                else:
                    proc.kill()
                proc.wait()
                raise
            return proc.returncode, stdout, stderr

        returncode, stdout, stderr = await asyncio.to_thread(_run)

        if returncode != 0:
            return {"status": "error", "traceback": stderr}

        return {"status": "success", "output": stdout}

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
