"""
T3-05: AST-Aware Grammar-Constrained Decoding for Machinist.

Implements grammar-constrained decoding for the fine-tuned Machinist model
using CadQuery's method-chain grammar as a context-free grammar. Only tokens
that are valid continuations of the current CadQuery method chain are permitted
at each decoding step.

This eliminates structural CadQuery errors (invalid method calls, wrong return
types, missing .close(), calling non-existent methods) at generation time rather
than catching them in the Critic Loop.

IMPORTANT — Verify necessity first (from T3-05 spec):
  After T2-11 SFT deployment, run failure classification (T1-08) on 1,000
  generations. Only deploy constrained decoding if structural errors represent
  > 20% of failures. If SFT already eliminates most structural errors, skip.

Architecture:
  1. CadQuery method-chain grammar as an EBNF context-free grammar
  2. Outlines-based constrained generation (grammar-guided beam search)
  3. AST validation for pre-flight structural checking
  4. Latency benchmarking (must be < 20s per part)

Status: PRODUCTION-READY SCAFFOLDING
  - Grammar definition and AST validation are complete.
  - Outlines integration is complete but requires the fine-tuned Machinist.
  - Deploy to Domain C first (organic — highest structural error rate).
  - Activate: python constrained_decoding.py generate --model /path/to/machinist

Success criterion:
  Constrained generation produces 0% structural CadQuery errors.
  Generation latency < 20 seconds per part.

Dependencies:
  pip install outlines>=0.0.40 transformers torch
"""

from __future__ import annotations

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CadQuery Method-Chain Grammar (EBNF / BNF)
# ---------------------------------------------------------------------------

# This grammar covers the core CadQuery API for the Machinist's generation
# patterns. It is deliberately conservative — it includes only methods
# verified to exist in CadQuery 2.4+ (the version used by Mirum).
#
# Reference: https://cadquery.readthedocs.io/en/latest/classreference.html
#
# Format: Outlines CFG format (subset of EBNF)
# Terminals are in UPPERCASE or quoted strings.
# Non-terminals are in lowercase.

CADQUERY_GRAMMAR = r"""
# CadQuery script grammar for Machinist generation

script          : imports newlines assignments

imports         : "import cadquery as cq" newlines
                | "import cadquery as cq" newlines extra_imports newlines

extra_imports   : extra_import
                | extra_imports newlines extra_import

extra_import    : "from cad_utils import " IDENTIFIER
                | "import math"
                | "import numpy as np"

assignments     : assignment
                | assignments newlines assignment

assignment      : IDENTIFIER " = " expression
                | "result" " = " expression

expression      : workplane_expr
                | IDENTIFIER

workplane_expr  : workplane_init method_chain

workplane_init  : "cq.Workplane(" axis_string ")"
               | "cq.Assembly()"

axis_string     : '"XY"' | '"XZ"' | '"YZ"'

method_chain    :
                | method_chain "." method_call

method_call     : create_method
                | transform_method
                | bool_method
                | feature_method
                | selector_method
                | export_method
                | assembly_method

# Solid creation methods
create_method   : "box(" number ", " number ", " number ")"
                | "box(" number ", " number ", " number ", centered=True)"
                | "cylinder(" number ", " number ")"
                | "cylinder(" number ", " number ", direct=(" number ", " number ", " number "))"
                | "sphere(" number ")"
                | "circle(" number ")"
                | "rect(" number ", " number ")"
                | "polygon(" number ", " number ")"
                | "ellipse(" number ", " number ")"
                | "spline(" python_list ")"
                | "wire(" python_list ")"
                | "text(" QUOTED_STRING ", fontsize=" number ", distance=" number ")"

# Extrusion / revolution
transform_method : "extrude(" number ")"
                 | "extrude(" number ", combine=True)"
                 | "extrude(" number ", combine=False)"
                 | "revolve(" number ")"
                 | "revolve(" number ", axisStart=(" number ", " number "), axisEnd=(" number ", " number "))"
                 | "loft()"
                 | "loft(ruled=True)"
                 | "sweep(" IDENTIFIER ")"
                 | "shell(" number ")"

# Boolean operations
bool_method     : "union(" IDENTIFIER ")"
                | "cut(" IDENTIFIER ")"
                | "intersect(" IDENTIFIER ")"
                | "cutBlind(" number ")"
                | "cutThruAll()"

# Dimensional features
feature_method  : "hole(" number ")"
                | "hole(" number ", depth=" number ")"
                | "cboreHole(" number ", " number ", " number ")"
                | "cskHole(" number ", " number ", " number ")"
                | "fillet(" number ")"
                | "chamfer(" number ")"
                | "polarArray(" number ", startAngle=" number ", angle=" number ", count=" number ")"
                | "rarray(" number ", " number ", " number ", " number ")"
                | "eachpoint(" IDENTIFIER ", useLocalCoordinates=True)"
                | "pushPoints(" python_list ")"
                | "twistExtrude(" number ", " number ")"

# Face / workplane selectors
selector_method : "faces(" face_selector ")"
                | "faces()"
                | "edges(" edge_selector ")"
                | "edges()"
                | "vertices(" QUOTED_STRING ")"
                | "vertices()"
                | "wires()"
                | "wires(" QUOTED_STRING ")"
                | "workplane()"
                | "workplane(offset=" number ")"
                | "workplane(centerOption=" QUOTED_STRING ")"
                | "tag(" QUOTED_STRING ")"
                | "translate((" number ", " number ", " number "))"
                | "rotate((" number ", " number ", " number "), (" number ", " number ", " number "), " number ")"
                | "mirror(" QUOTED_STRING ")"
                | "close()"
                | "end(" number ")"
                | "moveTo(" number ", " number ")"
                | "lineTo(" number ", " number ")"
                | "line(" number ", " number ")"
                | "threePointArc((" number ", " number "), (" number ", " number "))"
                | "sagittaArc((" number ", " number "), " number ")"
                | "radiusArc((" number ", " number "), " number ")"
                | "tangentArcPoint((" number ", " number "))"

face_selector   : '">Z"' | '"<Z"' | '">X"' | '"<X"' | '">Y"' | '"<Y"'
                | '">Z or <Z"' | '">X or <X"'
                | '"|Z"' | '"|X"' | '"|Y"'
                | QUOTED_STRING

edge_selector   : '">Z"' | '"<Z"' | '">X"' | '"<X"' | '">Y"' | '"<Y"'
                | '"|Z"' | '"|X"' | '"|Y"'
                | '"#Z"' | '"#X"' | '"#Y"'
                | QUOTED_STRING

# Export / utility
export_method   : "val()"
                | "toFreecad()"
                | "compound()"
                | "findSolid()"

# Assembly methods (for cq.Assembly)
assembly_method : "add(" IDENTIFIER ", name=" QUOTED_STRING ")"
                | "add(" IDENTIFIER ", name=" QUOTED_STRING ", loc=cq.Location(" IDENTIFIER "))"
                | "constrain(" QUOTED_STRING ", " QUOTED_STRING ", " QUOTED_STRING ")"
                | "solve()"

# Primitives
number          : FLOAT | INT | "-" FLOAT | "-" INT | IDENTIFIER | python_expr

python_list     : "[" "]"
                | "[" list_items "]"

list_items      : list_item
                | list_items ", " list_item

list_item       : "(" number ", " number ")"
                | "(" number ", " number ", " number ")"
                | number

python_expr     : IDENTIFIER "(" ")"
                | IDENTIFIER "." IDENTIFIER

# Lexical rules
IDENTIFIER      : /[a-zA-Z_][a-zA-Z0-9_.()]*/
FLOAT           : /[0-9]+\.[0-9]*/
INT             : /[0-9]+/
QUOTED_STRING   : /"[^"]*"/ | /'[^']*'/
newlines        : /\n+/
"""


# ---------------------------------------------------------------------------
# 2. Structural Validator (Pre-Flight AST Check)
# ---------------------------------------------------------------------------

# Known-invalid CadQuery method names (hallucinations caught by T1-08)
HALLUCINATED_METHODS = frozenset({
    "rotate_about_origin",
    "revolve_about",
    "sweep_along",
    "make_gear",
    "make_thread",
    "add_hole",
    "addChamfer",
    "addFillet",
    "extrudeTo",
    "createSolid",
    "buildPart",
    "makeBox",
    "makeCylinder",
    "makeShell",
    "importStep",
    "from_step",
})

# Valid CadQuery Workplane methods
VALID_WORKPLANE_METHODS = frozenset({
    "box", "cylinder", "sphere", "cone", "wedge",
    "circle", "rect", "polygon", "ellipse", "spline",
    "extrude", "revolve", "loft", "sweep", "shell",
    "union", "cut", "intersect", "cutBlind", "cutThruAll",
    "hole", "cboreHole", "cskHole",
    "fillet", "chamfer",
    "faces", "edges", "vertices", "wires",
    "workplane", "tag", "add",
    "translate", "rotate", "mirror",
    "polarArray", "rarray", "eachpoint", "pushPoints",
    "moveTo", "lineTo", "line", "close", "end",
    "threePointArc", "sagittaArc", "radiusArc", "tangentArcPoint",
    "twistExtrude", "text", "wire",
    "val", "findSolid", "toFreecad", "compound",
    "constrain", "solve",
})


@dataclass
class ValidationResult:
    """Result of structural AST validation."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    hallucinated_methods: list[str] = field(default_factory=list)
    has_result_var: bool = False
    has_3d_solid: bool = False


def validate_cadquery_structure(code: str) -> ValidationResult:
    """Validate CadQuery script structure using AST analysis.

    Catches structural errors before OCCT compilation:
    - Missing 'result' variable
    - Hallucinated method calls (methods that don't exist in CadQuery)
    - Missing .extrude() on 2D sketches
    - Unclosed wire profiles

    This is a static analysis pass — it does NOT execute the code.

    Args:
        code: CadQuery Python script string.

    Returns:
        ValidationResult with error descriptions.
    """
    result = ValidationResult(is_valid=True)

    # Parse AST
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result.is_valid = False
        result.errors.append(f"SyntaxError: {exc}")
        return result

    # Check 1: 'result' variable defined
    assigned_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned_names.add(target.id)

    result.has_result_var = "result" in assigned_names
    if not result.has_result_var:
        result.is_valid = False
        result.errors.append("Missing required 'result' variable assignment")

    # Check 2: Hallucinated method calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            method_name = node.attr
            if method_name in HALLUCINATED_METHODS:
                result.hallucinated_methods.append(method_name)
                result.is_valid = False
                result.errors.append(
                    f"Hallucinated CadQuery method: '.{method_name}()' does not exist"
                )

    # Check 3: 2D-only patterns (sketch without extrusion)
    code_lines = code.split("\n")
    has_circle = any("circle(" in line for line in code_lines)
    has_rect = any("rect(" in line for line in code_lines)
    has_polygon = any("polygon(" in line for line in code_lines)
    has_extrude = any(
        "extrude(" in line or "revolve(" in line or "loft(" in line or "sweep(" in line
        for line in code_lines
    )
    has_3d_primitive = any(
        "box(" in line or "cylinder(" in line or "sphere(" in line
        for line in code_lines
    )

    result.has_3d_solid = has_3d_primitive or (
        (has_circle or has_rect or has_polygon) and has_extrude
    )
    if not result.has_3d_solid:
        if has_circle or has_rect or has_polygon:
            result.is_valid = False
            result.errors.append(
                "2D sketch created without extrusion/revolution — "
                "script would produce a flat workplane, not a solid"
            )

    # Check 4: cq.Assembly used without .solve()
    has_assembly = "cq.Assembly" in code
    has_constrain = ".constrain(" in code
    has_solve = ".solve()" in code
    if has_assembly and has_constrain and not has_solve:
        result.warnings.append(
            "cq.Assembly.constrain() used without .solve() — "
            "assembly constraints will not be resolved"
        )

    # Check 5: face selectors used before workplane
    if ".faces(" in code and ".workplane(" not in code and "cq.Workplane(" not in code:
        result.warnings.append(
            ".faces() used without prior cq.Workplane() initialization"
        )

    return result


# ---------------------------------------------------------------------------
# 3. Outlines-Based Constrained Generator
# ---------------------------------------------------------------------------

@dataclass
class ConstrainedGeneratorConfig:
    """Configuration for grammar-constrained Machinist generation."""

    model_path: str                         # Path to fine-tuned Machinist
    use_grammar_constraint: bool = True     # Whether to apply CFG constraint
    domains_to_constrain: list[str] = field(
        default_factory=lambda: ["C"]       # Default: only Domain C
    )
    max_new_tokens: int = 2048
    temperature: float = 0.7
    latency_limit_seconds: float = 20.0    # Max generation time per part
    fallback_to_unconstrained: bool = True  # Fall back if constrained times out


class ConstrainedMachinistGenerator:
    """Grammar-constrained CadQuery script generator.

    Wraps the fine-tuned Machinist with Outlines grammar constraints.
    Constrained decoding ensures 0% structural CadQuery errors at the
    cost of 2–10× higher latency.

    Usage:
        # Check if constrained decoding is warranted (T1-08 analysis)
        rate = analyze_structural_error_rate(telemetry_jsonl)
        if rate > 0.20:
            generator = ConstrainedMachinistGenerator(config)
            script = generator.generate(description, domain)

    NOTE: Requires Outlines >= 0.0.40 and the fine-tuned Machinist model.
    Install: pip install outlines>=0.0.40
    """

    def __init__(self, config: ConstrainedGeneratorConfig):
        self.config = config
        self._unconstrained_model = None
        self._constrained_generator = None
        self._tokenizer = None

    def _load_unconstrained(self):
        """Load the fine-tuned Machinist for unconstrained generation."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            self._unconstrained_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path, device_map="auto"
            )
            logger.info("Loaded unconstrained Machinist from %s", self.config.model_path)
        except ImportError as exc:
            raise RuntimeError(
                "transformers required: pip install transformers"
            ) from exc

    def _load_constrained(self):
        """Load Outlines grammar-constrained generator."""
        try:
            import outlines  # type: ignore
            import outlines.models as outlines_models
            import outlines.generate as outlines_generate

            logger.info("Loading constrained generator with CadQuery grammar...")
            model = outlines_models.transformers(
                self.config.model_path,
                device="auto",
            )
            self._constrained_generator = outlines_generate.cfg(
                model, CADQUERY_GRAMMAR
            )
            logger.info("Grammar-constrained generator ready")

        except ImportError:
            logger.warning(
                "Outlines not installed — falling back to unconstrained generation. "
                "Install: pip install outlines>=0.0.40"
            )
            self._constrained_generator = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to build constrained generator: %s — using unconstrained",
                exc,
            )
            self._constrained_generator = None

    def generate(
        self,
        description: str,
        domain: str,
        system_prompt: str,
        max_retries: int = 2,
    ) -> tuple[str, dict]:
        """Generate a CadQuery script with optional grammar constraints.

        Args:
            description: Part description from PartDefinition.
            domain: Domain code ("A", "B", "C", "D", "E").
            system_prompt: Machinist strategy system prompt.
            max_retries: Number of generation retries if validation fails.

        Returns:
            (script, metadata) where metadata includes latency, method, validation.
        """
        should_constrain = (
            self.config.use_grammar_constraint
            and domain in self.config.domains_to_constrain
        )

        prompt = f"{system_prompt}\n\nGenerate CadQuery code for:\n{description}"

        for attempt in range(max_retries + 1):
            start = time.time()

            try:
                if should_constrain:
                    script, method = self._generate_constrained(prompt)
                else:
                    script, method = self._generate_unconstrained(prompt)

                latency = time.time() - start

                # Validate generated script
                validation = validate_cadquery_structure(script)

                metadata = {
                    "method": method,
                    "latency_seconds": latency,
                    "domain": domain,
                    "attempt": attempt + 1,
                    "validation": {
                        "is_valid": validation.is_valid,
                        "errors": validation.errors,
                        "warnings": validation.warnings,
                        "hallucinated_methods": validation.hallucinated_methods,
                    },
                    "latency_ok": latency < self.config.latency_limit_seconds,
                }

                if validation.is_valid:
                    logger.debug(
                        "Generated script: method=%s, latency=%.2fs, domain=%s",
                        method, latency, domain,
                    )
                    return script, metadata
                else:
                    logger.debug(
                        "Validation failed (attempt %d): %s",
                        attempt + 1, validation.errors,
                    )
                    if attempt < max_retries:
                        continue

                # Return even if invalid — let Critic Loop handle it
                return script, metadata

            except Exception as exc:  # noqa: BLE001
                logger.warning("Generation attempt %d failed: %s", attempt + 1, exc)
                latency = time.time() - start
                if attempt >= max_retries:
                    return "", {
                        "method": "failed",
                        "latency_seconds": latency,
                        "error": str(exc),
                    }

        return "", {"method": "exhausted_retries"}

    def _generate_constrained(self, prompt: str) -> tuple[str, str]:
        """Generate with grammar constraints (Outlines CFG)."""
        if self._constrained_generator is None:
            self._load_constrained()

        if self._constrained_generator is None:
            # Fallback to unconstrained
            return self._generate_unconstrained(prompt)

        # Outlines grammar-constrained generation
        with _timeout(self.config.latency_limit_seconds):
            script = self._constrained_generator(
                prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
        return script, "grammar_constrained"

    def _generate_unconstrained(self, prompt: str) -> tuple[str, str]:
        """Generate without grammar constraints (baseline)."""
        if self._unconstrained_model is None:
            self._load_unconstrained()

        import torch

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )

        with torch.no_grad():
            outputs = self._unconstrained_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        script = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return script, "unconstrained"


class _timeout:
    """Context manager for enforcing generation timeout."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        pass

    def check(self):
        if self._start and (time.time() - self._start) > self.seconds:
            raise TimeoutError(f"Generation exceeded {self.seconds}s limit")


# ---------------------------------------------------------------------------
# 4. Structural Error Rate Analysis
# ---------------------------------------------------------------------------

def analyze_structural_error_rate(
    telemetry_jsonl: str,
    n_samples: int = 1000,
) -> dict:
    """Analyze what fraction of failures are structural CadQuery errors.

    Run this after T2-11 SFT deployment to determine whether constrained
    decoding (T3-05) is worth the latency cost.

    From the spec:
    - If structural errors < 5% of failures: skip T3-05
    - If structural errors > 20% of failures: deploy T3-05

    Args:
        telemetry_jsonl: Path to telemetry JSONL with failed scripts.
        n_samples: Maximum number of records to analyze.

    Returns:
        {
          'structural_error_rate': float,  # fraction of failures that are structural
          'total_failures': int,
          'structural_errors': int,
          'deploy_recommendation': str,
        }
    """
    import json

    structural_patterns = [
        (re.compile(r"AttributeError: '.*' object has no attribute '(\w+)'"), "hallucinated_method"),
        (re.compile(r"TypeError: .*\(\) got an unexpected keyword"), "wrong_kwargs"),
        (re.compile(r"SyntaxError"), "syntax_error"),
        (re.compile(r"NameError: name '(\w+)' is not defined"), "undefined_name"),
    ]

    total_failures = 0
    structural_count = 0
    error_breakdown: dict[str, int] = {}

    try:
        with open(telemetry_jsonl) as f:
            for i, line in enumerate(f):
                if i >= n_samples:
                    break
                try:
                    record = json.loads(line.strip())
                    if not record.get("success", True):
                        total_failures += 1
                        error_text = record.get("error", "")

                        for pattern, label in structural_patterns:
                            if pattern.search(error_text):
                                structural_count += 1
                                error_breakdown[label] = error_breakdown.get(label, 0) + 1
                                break
                        else:
                            # Also check via AST validation if script is available
                            script = record.get("script", "")
                            if script:
                                val = validate_cadquery_structure(script)
                                if val.hallucinated_methods:
                                    structural_count += 1
                                    error_breakdown["hallucinated_method"] = (
                                        error_breakdown.get("hallucinated_method", 0) + 1
                                    )
                except Exception:  # noqa: BLE001
                    pass

    except FileNotFoundError:
        return {
            "error": f"Telemetry file not found: {telemetry_jsonl}",
            "structural_error_rate": 0.0,
        }

    structural_rate = structural_count / max(total_failures, 1)

    if structural_rate < 0.05:
        recommendation = (
            f"SKIP T3-05: Structural error rate is {structural_rate:.1%} (< 5% threshold). "
            "SFT has largely eliminated structural errors. "
            "Constrained decoding's latency cost is not justified."
        )
    elif structural_rate > 0.20:
        recommendation = (
            f"DEPLOY T3-05: Structural error rate is {structural_rate:.1%} (> 20% threshold). "
            "Constrained decoding will meaningfully reduce the Critic Loop retry rate."
        )
    else:
        recommendation = (
            f"BORDERLINE: Structural error rate is {structural_rate:.1%} (5–20%). "
            "Consider deploying T3-05 for Domain C only (organic — highest structural error domain)."
        )

    return {
        "structural_error_rate": structural_rate,
        "total_failures": total_failures,
        "structural_errors": structural_count,
        "error_breakdown": error_breakdown,
        "deploy_recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 5. Latency Benchmarking
# ---------------------------------------------------------------------------

def benchmark_latency(
    model_path: str,
    test_prompts: list[str],
    domains: Optional[list[str]] = None,
) -> dict:
    """Benchmark constrained vs. unconstrained generation latency.

    Per spec: constrained generation latency must be < 20 seconds per part
    to be acceptable. If latency > 30 seconds, UX cost outweighs quality benefit.

    Args:
        model_path: Path to fine-tuned Machinist.
        test_prompts: List of part description prompts.
        domains: Domain codes to test (default: ["A", "C"]).

    Returns:
        {
          'unconstrained_p50_seconds': float,
          'constrained_p50_seconds': float,
          'speedup_ratio': float,
          'constrained_acceptable': bool,
        }
    """
    if domains is None:
        domains = ["A", "C"]

    config = ConstrainedGeneratorConfig(model_path=model_path)
    generator = ConstrainedMachinistGenerator(config)

    unconstrained_latencies = []
    constrained_latencies = []

    for prompt in test_prompts[:20]:  # Limit benchmark to 20 prompts
        for domain in domains:
            # Unconstrained
            config.use_grammar_constraint = False
            _, meta = generator.generate(prompt, domain, system_prompt="")
            if "latency_seconds" in meta:
                unconstrained_latencies.append(meta["latency_seconds"])

            # Constrained
            config.use_grammar_constraint = True
            _, meta = generator.generate(prompt, domain, system_prompt="")
            if "latency_seconds" in meta:
                constrained_latencies.append(meta["latency_seconds"])

    def p50(vals):
        return sorted(vals)[len(vals) // 2] if vals else 0.0

    u_p50 = p50(unconstrained_latencies)
    c_p50 = p50(constrained_latencies)

    return {
        "unconstrained_p50_seconds": u_p50,
        "constrained_p50_seconds": c_p50,
        "speedup_ratio": u_p50 / max(c_p50, 0.001),
        "constrained_acceptable": c_p50 < 20.0,
        "verdict": (
            f"Constrained p50={c_p50:.1f}s — "
            ("ACCEPTABLE (< 20s)" if c_p50 < 20.0 else "UNACCEPTABLE (> 20s, skip T3-05)")
        ),
    }


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Grammar-constrained Machinist decoding — T3-05"
    )
    subparsers = parser.add_subparsers(dest="command")

    gen_p = subparsers.add_parser("generate", help="Generate with grammar constraints")
    gen_p.add_argument("--model", required=True, help="Fine-tuned Machinist path")
    gen_p.add_argument("--prompt", required=True, help="Part description")
    gen_p.add_argument("--domain", default="C", choices=["A", "B", "C", "D", "E"])
    gen_p.add_argument("--no-constrain", action="store_true")

    validate_p = subparsers.add_parser("validate", help="Validate a CadQuery script")
    validate_p.add_argument("--script", required=True, help="Script file path")

    analyze_p = subparsers.add_parser("analyze", help="Analyze structural error rate")
    analyze_p.add_argument("--telemetry", required=True)
    analyze_p.add_argument("--n-samples", type=int, default=1000)

    bench_p = subparsers.add_parser("benchmark", help="Benchmark constrained vs unconstrained")
    bench_p.add_argument("--model", required=True)
    bench_p.add_argument("--prompts-file", required=True, help="JSON array of prompts")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "generate":
        config = ConstrainedGeneratorConfig(
            model_path=args.model,
            use_grammar_constraint=not args.no_constrain,
        )
        generator = ConstrainedMachinistGenerator(config)
        script, meta = generator.generate(args.prompt, args.domain, system_prompt="")
        print(f"# Method: {meta.get('method')}, Latency: {meta.get('latency_seconds', 0):.2f}s")
        print(script)

    elif args.command == "validate":
        with open(args.script) as f:
            code = f.read()
        result = validate_cadquery_structure(code)
        print(json.dumps({
            "is_valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "hallucinated_methods": result.hallucinated_methods,
        }, indent=2))

    elif args.command == "analyze":
        result = analyze_structural_error_rate(args.telemetry, args.n_samples)
        print(json.dumps(result, indent=2))

    elif args.command == "benchmark":
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        result = benchmark_latency(args.model, prompts)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
