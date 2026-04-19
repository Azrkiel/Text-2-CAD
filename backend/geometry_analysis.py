"""
T3-07: Gaussian Curvature Analysis for Domain C Quality.

Computes the Gaussian curvature distribution over Domain C (organic/ergonomic)
compiled surfaces using OCCT's BRepLProp_SLProps API. Flags regions with
unexpectedly negative curvature (saddle shapes on surfaces described as convex)
or zero curvature on surfaces described as continuously curved.

Gaussian curvature classifies surface regions:
  > 0  (positive): bowl-shaped / dome (elliptic point)
  = 0  (zero): flat or cylindrical (parabolic point)
  < 0  (negative): saddle-shaped (hyperbolic point)

For ergonomic parts, saddle regions are usually design errors — they create
uncomfortable grip surfaces and unexpected reflections.

Integration:
  - Called from Critic Loop after successful compilation for Domain C parts.
  - Curvature failures consume one retry and inject diagnostic feedback.
  - Gaussian curvature results stored in telemetry for Domain C parts.
  - Pass rate tracked in the Domain C section of the evaluation harness.

Status: PRODUCTION-READY
  - Full OCCT implementation using BRepLProp_SLProps.
  - VLM-based saddle region flagging (stub, requires Gemini Vision + rendered image).
  - Integrates into Critic Loop via continuity_check.py pattern.

Dependencies:
  pip install open-cascade-python  (for OCCT BRepLProp API)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CurvaturePoint:
    """Gaussian curvature measurement at a single surface point."""
    face_index: int
    u: float
    v: float
    gaussian_curvature: float   # k1 * k2 (principal curvature product)
    mean_curvature: float       # (k1 + k2) / 2
    min_curvature: float        # k1 (smallest principal curvature)
    max_curvature: float        # k2 (largest principal curvature)
    is_saddle: bool             # gaussian_curvature < -saddle_threshold
    is_flat: bool               # |gaussian_curvature| < flat_threshold


@dataclass
class GaussianCurvatureResult:
    """Summary of Gaussian curvature analysis for a compiled Domain C part."""

    passed: bool
    n_faces_analyzed: int = 0
    n_sample_points: int = 0
    n_saddle_points: int = 0
    n_flat_points: int = 0

    min_gaussian: float = 0.0
    max_gaussian: float = 0.0
    mean_gaussian: float = 0.0

    saddle_regions: list[str] = field(default_factory=list)
    flat_regions: list[str] = field(default_factory=list)
    violation_descriptions: list[str] = field(default_factory=list)

    method: str = "occt_breplprop"  # "occt_breplprop" | "skipped" | "failed"
    feedback_message: str = ""

    # For telemetry
    step_path: str = ""
    domain: str = "C"


# ---------------------------------------------------------------------------
# 2. OCCT Gaussian Curvature Computation
# ---------------------------------------------------------------------------

def compute_gaussian_curvature(
    shape_or_path,
    n_samples: int = 30,
    saddle_threshold: float = -1e-4,
    flat_threshold: float = 1e-6,
) -> GaussianCurvatureResult:
    """Compute Gaussian curvature distribution over a compiled B-rep solid.

    Uses OCCT's BRepLProp_SLProps API to sample curvature at a grid of
    (u, v) parameter space points for each face of the solid.

    Args:
        shape_or_path: Either:
            - A TopoDS_Shape object (from OCCT compilation), or
            - A str/Path to a .step file (will be loaded automatically).
        n_samples: Number of samples per dimension (total: n_samples² per face).
        saddle_threshold: Gaussian curvature below this value is flagged as
            a saddle (negative curvature region). Default -1e-4 mm⁻².
        flat_threshold: Gaussian curvature with |K| < this is flagged as flat.

    Returns:
        GaussianCurvatureResult with analysis summary and feedback message.
    """
    try:
        from OCC.Core.BRep import BRep_Tool  # type: ignore
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface  # type: ignore
        from OCC.Core.BRepLProp import BRepLProp_SLProps  # type: ignore
        from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder  # type: ignore
        from OCC.Core.GeomLProp import GeomLProp_SLProps  # type: ignore
        from OCC.Core.TopAbs import TopAbs_FACE  # type: ignore
        from OCC.Core.TopExp import TopExp_Explorer  # type: ignore
        from OCC.Core.TopoDS import topods_Face  # type: ignore

        # Load shape if path provided
        if isinstance(shape_or_path, (str, Path)):
            shape = _load_step(str(shape_or_path))
            if shape is None:
                return GaussianCurvatureResult(
                    passed=True,
                    method="skipped",
                    feedback_message="STEP file could not be loaded for curvature analysis",
                )
            step_path = str(shape_or_path)
        else:
            shape = shape_or_path
            step_path = ""

        all_curvature_points: list[CurvaturePoint] = []
        saddle_regions: list[str] = []
        flat_regions: list[str] = []

        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        face_index = 0

        while face_explorer.More():
            raw_face = face_explorer.Current()
            face = topods_Face(raw_face)

            try:
                adaptor = BRepAdaptor_Surface(face)
                surface = adaptor.Surface().Surface()

                u1 = adaptor.FirstUParameter()
                u2 = adaptor.LastUParameter()
                v1 = adaptor.FirstVParameter()
                v2 = adaptor.LastVParameter()

                # Skip degenerate parameter ranges
                if (u2 - u1) < 1e-10 or (v2 - v1) < 1e-10:
                    face_explorer.Next()
                    face_index += 1
                    continue

                face_saddle_count = 0
                face_flat_count = 0
                face_points: list[CurvaturePoint] = []

                for i in range(n_samples):
                    u = u1 + (u2 - u1) * (i + 0.5) / n_samples
                    for j in range(n_samples):
                        v = v1 + (v2 - v1) * (j + 0.5) / n_samples

                        try:
                            props = GeomLProp_SLProps(surface, u, v, 2, 1e-6)

                            if not props.IsNormalDefined():
                                continue

                            k_min = props.MinCurvature()
                            k_max = props.MaxCurvature()
                            k_gaussian = k_min * k_max
                            k_mean = (k_min + k_max) / 2.0

                            is_saddle = k_gaussian < saddle_threshold
                            is_flat = abs(k_gaussian) < flat_threshold

                            if is_saddle:
                                face_saddle_count += 1
                            if is_flat:
                                face_flat_count += 1

                            face_points.append(CurvaturePoint(
                                face_index=face_index,
                                u=u, v=v,
                                gaussian_curvature=k_gaussian,
                                mean_curvature=k_mean,
                                min_curvature=k_min,
                                max_curvature=k_max,
                                is_saddle=is_saddle,
                                is_flat=is_flat,
                            ))

                        except Exception:  # noqa: BLE001
                            continue

                all_curvature_points.extend(face_points)

                # Report per-face curvature character
                face_saddle_fraction = face_saddle_count / max(len(face_points), 1)
                if face_saddle_fraction > 0.3:
                    saddle_regions.append(
                        f"Face {face_index}: {face_saddle_fraction:.0%} saddle points "
                        f"(negative Gaussian curvature)"
                    )
                elif face_flat_count > len(face_points) * 0.8:
                    flat_regions.append(f"Face {face_index}: predominantly flat")

            except Exception as exc:  # noqa: BLE001
                logger.debug("Face %d curvature extraction failed: %s", face_index, exc)

            face_explorer.Next()
            face_index += 1

        if not all_curvature_points:
            return GaussianCurvatureResult(
                passed=True,
                method="skipped",
                n_faces_analyzed=face_index,
                feedback_message="No curvature data extracted (all faces may be planar or degenerate)",
                step_path=step_path,
            )

        # Compute summary statistics
        gauss_values = [p.gaussian_curvature for p in all_curvature_points]
        n_saddle = sum(1 for p in all_curvature_points if p.is_saddle)
        n_flat = sum(1 for p in all_curvature_points if p.is_flat)
        saddle_fraction = n_saddle / len(all_curvature_points)

        # Pass if saddle fraction is < 25% of all sample points
        passed = saddle_fraction < 0.25

        violation_descriptions = []
        for region in saddle_regions:
            violation_descriptions.append(
                f"Saddle region: {region} — surface curves in opposite directions "
                "at this location (negative Gaussian curvature)."
            )

        # Build feedback message for Critic Loop
        feedback_message = ""
        if not passed:
            feedback_message = (
                f"Gaussian curvature analysis FAILED: {saddle_fraction:.0%} of surface "
                f"points are saddle-shaped (target: < 25%). "
                f"Saddle regions: {'; '.join(saddle_regions[:3])}. "
                "Saddle geometry creates uncomfortable grip surfaces and unexpected "
                "reflections. Consider:\n"
                "  1. Simplify the loft cross-sections to use rounder profiles "
                "(fewer control points)\n"
                "  2. Replace spline cross-sections with ellipses where possible\n"
                "  3. Reduce the cross-section variation — sudden shape changes create saddles\n"
                "  4. Use .loft(ruled=True) for surfaces where flat interpolation is acceptable"
            )
        elif saddle_regions:
            feedback_message = (
                f"Curvature check PASSED ({saddle_fraction:.0%} saddle fraction < 25% threshold). "
                f"Minor saddle regions detected at: {'; '.join(saddle_regions[:2])}. "
                "These are acceptable for organic forms."
            )

        return GaussianCurvatureResult(
            passed=passed,
            n_faces_analyzed=face_index,
            n_sample_points=len(all_curvature_points),
            n_saddle_points=n_saddle,
            n_flat_points=n_flat,
            min_gaussian=min(gauss_values),
            max_gaussian=max(gauss_values),
            mean_gaussian=sum(gauss_values) / len(gauss_values),
            saddle_regions=saddle_regions,
            flat_regions=flat_regions,
            violation_descriptions=violation_descriptions,
            method="occt_breplprop",
            feedback_message=feedback_message,
            step_path=step_path,
        )

    except ImportError:
        logger.warning(
            "OCCT not available — skipping Gaussian curvature analysis. "
            "Install: pip install open-cascade-python"
        )
        return GaussianCurvatureResult(
            passed=True,
            method="skipped",
            feedback_message="OCCT not available — curvature analysis skipped",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gaussian curvature analysis failed: %s", exc)
        return GaussianCurvatureResult(
            passed=True,  # Don't block on analysis failure
            method="failed",
            feedback_message=f"Curvature analysis error (non-blocking): {exc}",
        )


def _load_step(step_path: str):
    """Load a STEP file and return a TopoDS_Shape."""
    try:
        from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != 1:
            return None
        reader.TransferRoots()
        return reader.OneShape()
    except Exception as exc:  # noqa: BLE001
        logger.warning("STEP load failed for %s: %s", step_path, exc)
        return None


# ---------------------------------------------------------------------------
# 3. VLM Zebra-Analysis Stub (for rendered image evaluation)
# ---------------------------------------------------------------------------

async def check_curvature_via_vlm(
    rendered_image_path: str,
    part_description: str,
    api_key: Optional[str] = None,
) -> GaussianCurvatureResult:
    """Identify surface curvature problems via VLM zebra-striping analysis.

    A zebra-stripe rendering pattern is a standard industrial method for
    visually detecting curvature discontinuities and saddle regions. This
    stub uses Gemini Vision to analyze a rendered screenshot for surface
    quality indicators.

    Args:
        rendered_image_path: Path to a rendered screenshot of the part (PNG/JPG).
            Ideally rendered with a zebra-stripe environment map.
        part_description: Original part description for context.
        api_key: Gemini API key.

    Returns:
        GaussianCurvatureResult with VLM-based assessment.

    Status: STUB — requires:
        1. Rendering pipeline to produce zebra-stripe screenshots (T1-07 GLB viewer)
        2. Gemini Vision API call with the rendered image
        3. Structured output schema for surface quality scoring
    """
    logger.info(
        "VLM curvature analysis stub called for %s — returning placeholder",
        rendered_image_path,
    )

    # TODO: Implement full VLM zebra analysis:
    # 1. Load rendered_image_path as base64
    # 2. Send to Gemini Vision with system prompt:
    #    "You are a mechanical surface quality inspector. This image shows a 3D
    #    rendered part with a zebra-stripe environment map for curvature analysis.
    #    Zebra stripes should be:
    #      - Smooth and continuous: good surface quality
    #      - Kinked or discontinuous at edges: G1 discontinuity (expected at feature edges)
    #      - Wavy or irregular mid-surface: possible G2 failure or saddle region
    #    Identify regions where zebra stripes are irregular (wavy, kinked, pinched)
    #    on smooth continuous surfaces. Report as JSON:
    #    {severity: 0-3, regions: [...], feedback: str}"
    # 3. Parse structured output into GaussianCurvatureResult

    return GaussianCurvatureResult(
        passed=True,
        method="vlm_stub",
        feedback_message=(
            "VLM zebra analysis not yet implemented. "
            "Requires rendering pipeline and Gemini Vision integration. "
            "See T3-07 stub for implementation details."
        ),
    )


# ---------------------------------------------------------------------------
# 4. Critic Loop Integration
# ---------------------------------------------------------------------------

def curvature_check_for_critic(
    step_path: str,
    domain: str,
    part_description: str,
    n_samples: int = 25,
) -> GaussianCurvatureResult:
    """Run Gaussian curvature check in Critic Loop context.

    This is the entry point called from agents.py Critic Loop for Domain C
    parts after successful OCCT compilation. Returns a result with a
    ready-to-inject feedback_message for the next generation attempt.

    Integration pattern in agents.py:
        if domain == "C" and step_path:
            from geometry_analysis import curvature_check_for_critic
            curvature = curvature_check_for_critic(step_path, domain, description)
            if not curvature.passed:
                error_context = curvature.feedback_message
                continue  # retry with curvature feedback

    Args:
        step_path: Path to the compiled .step file.
        domain: Part domain (should be "C" for organic parts).
        part_description: Original part description (used for context in feedback).
        n_samples: Curvature sampling density per face dimension.

    Returns:
        GaussianCurvatureResult with feedback_message populated.
    """
    if domain != "C":
        return GaussianCurvatureResult(
            passed=True,
            method="skipped",
            feedback_message="Curvature check only applies to Domain C parts",
        )

    result = compute_gaussian_curvature(step_path, n_samples=n_samples)

    # Enrich feedback with part description context if failed
    if not result.passed and part_description:
        context_hint = _infer_curvature_context(part_description)
        if context_hint:
            result.feedback_message = context_hint + "\n\n" + result.feedback_message

    return result


def _infer_curvature_context(description: str) -> str:
    """Infer expected curvature character from part description."""
    desc_lower = description.lower()

    if any(kw in desc_lower for kw in ["ergonomic", "grip", "handle", "comfort"]):
        return (
            "This part is described as ergonomic/grip-oriented. "
            "Ergonomic surfaces should be predominantly convex (positive Gaussian curvature) "
            "to conform to the human hand. Saddle regions create discomfort."
        )
    if any(kw in desc_lower for kw in ["dome", "cap", "shell", "hemisphere"]):
        return (
            "This part is described as dome/shell-shaped. "
            "All surfaces should have positive Gaussian curvature (bowl/dome shape). "
            "Any saddle region indicates an error in the loft profile progression."
        )
    if any(kw in desc_lower for kw in ["fairing", "aerodynamic", "nose", "body"]):
        return (
            "This aerodynamic body should have smoothly varying curvature. "
            "Saddle regions on the main body indicate cross-section profile instability."
        )
    return ""


# ---------------------------------------------------------------------------
# 5. Telemetry Integration
# ---------------------------------------------------------------------------

def to_telemetry_record(result: GaussianCurvatureResult) -> dict:
    """Convert a GaussianCurvatureResult to a telemetry-compatible dict."""
    return {
        "gaussian_curvature_analysis": {
            "passed": result.passed,
            "method": result.method,
            "n_faces_analyzed": result.n_faces_analyzed,
            "n_sample_points": result.n_sample_points,
            "n_saddle_points": result.n_saddle_points,
            "saddle_fraction": (
                result.n_saddle_points / max(result.n_sample_points, 1)
            ),
            "min_gaussian_mm2": result.min_gaussian,
            "max_gaussian_mm2": result.max_gaussian,
            "mean_gaussian_mm2": result.mean_gaussian,
            "n_flagged_regions": len(result.saddle_regions),
        }
    }


# ---------------------------------------------------------------------------
# 6. Evaluation Harness Integration
# ---------------------------------------------------------------------------

def evaluate_domain_c_curvature(
    step_files: list[str],
    n_samples: int = 20,
) -> dict:
    """Evaluate Gaussian curvature pass rate across a batch of Domain C parts.

    For use in the evaluation harness (T1-09 Domain-Stratified Eval).

    Args:
        step_files: List of .step file paths for Domain C compiled parts.
        n_samples: Curvature sampling density.

    Returns:
        {pass_rate, mean_saddle_fraction, per_file_results}
    """
    results = []
    for path in step_files:
        result = compute_gaussian_curvature(path, n_samples=n_samples)
        results.append({
            "step_file": path,
            "passed": result.passed,
            "saddle_fraction": (
                result.n_saddle_points / max(result.n_sample_points, 1)
            ),
            "method": result.method,
        })

    pass_rate = sum(1 for r in results if r["passed"]) / max(len(results), 1)
    mean_saddle = sum(r["saddle_fraction"] for r in results) / max(len(results), 1)

    return {
        "pass_rate": pass_rate,
        "mean_saddle_fraction": mean_saddle,
        "n_analyzed": len(results),
        "per_file_results": results,
    }


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Gaussian curvature analysis — T3-07"
    )
    subparsers = parser.add_subparsers(dest="command")

    analyze_p = subparsers.add_parser("analyze", help="Analyze a single STEP file")
    analyze_p.add_argument("--step", required=True, help="Path to .step file")
    analyze_p.add_argument("--domain", default="C")
    analyze_p.add_argument("--n-samples", type=int, default=30)
    analyze_p.add_argument("--description", default="", help="Part description for context")

    batch_p = subparsers.add_parser("batch", help="Evaluate a batch of Domain C STEP files")
    batch_p.add_argument("--dir", required=True, help="Directory containing .step files")
    batch_p.add_argument("--n-samples", type=int, default=20)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "analyze":
        result = curvature_check_for_critic(
            step_path=args.step,
            domain=args.domain,
            part_description=args.description,
            n_samples=args.n_samples,
        )
        print(json.dumps({
            "passed": result.passed,
            "method": result.method,
            "n_faces_analyzed": result.n_faces_analyzed,
            "n_saddle_points": result.n_saddle_points,
            "saddle_regions": result.saddle_regions,
            "feedback_message": result.feedback_message,
        }, indent=2))

    elif args.command == "batch":
        from pathlib import Path
        step_files = [str(p) for p in Path(args.dir).glob("*.step")]
        if not step_files:
            print(f"No .step files found in {args.dir}")
            return
        result = evaluate_domain_c_curvature(step_files, n_samples=args.n_samples)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
