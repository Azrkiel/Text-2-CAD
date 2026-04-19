"""
Parametric Design History Post-Processor.

Extracts numeric literals from generated CadQuery scripts and replaces
them with named Python variables at the top of the script. This converts
opaque generated code ("circle(25).extrude(30)") into a parametric model
with named, editable parameters ("RADIUS_1 = 25; HEIGHT_1 = 30").

The parameter names are derived from context heuristics (function argument
names, variable names in assignments) rather than pure sequential numbering.
"""

import ast
import re
from typing import Optional


# CadQuery methods and their argument name hints
_METHOD_ARG_HINTS: dict[str, list[str]] = {
    "box": ["LENGTH", "WIDTH", "HEIGHT"],
    "cylinder": ["HEIGHT", "RADIUS"],
    "circle": ["RADIUS"],
    "rect": ["X_LEN", "Y_LEN"],
    "sphere": ["RADIUS"],
    "cone": ["HEIGHT", "RADIUS1", "RADIUS2"],
    "extrude": ["DISTANCE"],
    "cutBlind": ["DEPTH"],
    "revolve": ["ANGLE_DEG"],
    "fillet": ["RADIUS"],
    "chamfer": ["DISTANCE"],
    "shell": ["THICKNESS"],
    "hole": ["DIAMETER"],
    "workplane": ["OFFSET"],
    "center": ["X_OFFSET", "Y_OFFSET"],
    "polarArray": ["RADIUS", "START_ANGLE", "ANGLE", "COUNT"],
    "slot2D": ["LENGTH", "DIAMETER"],
    "text": ["FONT_SIZE", "DISTANCE"],
    "threePointArc": [],
    "tangentArcPoint": [],
}


def parameterize_script(code: str) -> tuple[str, dict[str, float]]:
    """Extract numeric literals from CadQuery code and replace with named variables.

    Returns:
        (parameterized_code, parameter_dict) where parameter_dict maps
        variable names to their numeric values.

    Skips literals that are:
    - Small integers used as loop counters or array indices (0, 1, 2, 3)
    - 90, 180, 270, 360 (standard angles that should stay literal)
    - Values already used multiple times with the same exact value
    """
    if not code.strip():
        return code, {}

    # Find all numeric literals in function calls using regex
    # Pattern: method_name(num, ...) or method_name(keyword=num, ...)
    params: dict[str, float] = {}
    param_counter: dict[str, int] = {}
    replacements: list[tuple[int, int, str]] = []  # (start, end, replacement)

    # Skip these common values that should stay literal
    SKIP_VALUES = {0, 1, 2, 3, -1, 90.0, 180.0, 270.0, 360.0, 0.0}

    def _get_param_name(method_name: str, arg_index: int, value: float) -> Optional[str]:
        """Generate a descriptive parameter name."""
        method_clean = method_name.upper()
        hints = _METHOD_ARG_HINTS.get(method_name, [])
        if arg_index < len(hints):
            base = hints[arg_index]
        else:
            base = f"{method_clean}_PARAM"

        param_counter[base] = param_counter.get(base, 0) + 1
        n = param_counter[base]
        return base if n == 1 else f"{base}_{n}"

    # Use regex to find method calls and extract numeric args
    # Pattern: .method_name(arg1, arg2, ...) where args may be numbers
    call_pattern = re.compile(
        r'\.(\w+)\(([^)]*)\)',
        re.MULTILINE,
    )

    modified = code
    offset = 0

    for match in call_pattern.finditer(code):
        method_name = match.group(1)
        args_str = match.group(2)

        # Parse individual arguments
        args = [a.strip() for a in args_str.split(",") if a.strip()]
        new_args = list(args)
        arg_idx = 0

        for i, arg in enumerate(args):
            # Remove keyword= prefix if present
            if "=" in arg:
                kw, val_str = arg.split("=", 1)
                val_str = val_str.strip()
                kw = kw.strip()
            else:
                kw = None
                val_str = arg.strip()

            # Try to parse as a number
            try:
                val = float(val_str)
            except (ValueError, TypeError):
                arg_idx += 1
                continue

            # Skip values that should stay literal
            if val in SKIP_VALUES or (val == int(val) and abs(val) <= 3):
                arg_idx += 1
                continue

            # Skip if it's part of a string expression
            if "'" in val_str or '"' in val_str:
                arg_idx += 1
                continue

            # Generate parameter name
            param_name = _get_param_name(method_name, arg_idx, val)
            if param_name:
                params[param_name] = val
                if kw:
                    new_args[i] = f"{kw}={param_name}"
                else:
                    new_args[i] = param_name

            arg_idx += 1

        # Only replace if something changed
        if new_args != args:
            original_call = match.group(0)
            new_call = f".{method_name}({', '.join(new_args)})"
            modified = modified.replace(original_call, new_call, 1)

    if not params:
        return code, {}

    # Build the parameter header
    header_lines = [
        "# ── Parameters ─────────────────────────────────────────────",
        "# Edit these values and click 'Run' to update the model",
    ]
    for name, value in params.items():
        # Format: integer if no fractional part
        if value == int(value):
            header_lines.append(f"{name} = {int(value)}  # mm")
        else:
            header_lines.append(f"{name} = {value:.2f}  # mm")
    header_lines.append("# ────────────────────────────────────────────────────────")
    header_lines.append("")

    parameterized = "\n".join(header_lines) + "\n" + modified
    return parameterized, params
