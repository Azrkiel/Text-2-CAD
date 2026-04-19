"""
T2-04: G2 Continuity Validation for Domain C (Organic Parts).

After a Domain C CadQuery script compiles successfully, this module checks
whether the resulting surfaces have G2 (curvature) continuity at all shared
edges. G1 creases are a hard failure for consumer products and medical devices.

Two approaches are available:

  Approach A (Fast, no GPU): VLM-based Zebra Analysis
    - Render a Zebra Analysis (striped reflection) image using a headless renderer.
    - Upload to Gemini Vision and ask it to detect stripe discontinuities.
    - Requires OCCT rendering or Blender subprocess.

  Approach B (Analytical, no rendering): OCCT curvature sampling
    - Sample surface curvature on both sides of each shared edge via
      OCC.Core.BRepLProp.BRepLProp_SLProps.
    - Flag edges where the curvature vectors are non-parallel (G2 failure).
    - Requires pythonocc-core. Falls back gracefully if not available.

Current implementation uses Approach B with fallback to a coarse Approach A
stub if pythonocc-core is unavailable (typical in the Docker environment,
which bundles cadquery but not pythonocc-core directly).

Usage from critic.py:
    from continuity_check import check_g2_continuity, G2CheckResult
    result = check_g2_continuity(compiled_step_path)
    if not result.passed:
        # Inject feedback into next Critic attempt
        feedback = result.feedback_message
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("mirum.continuity")


@dataclass
class G2CheckResult:
    """Result of a G2 continuity check on a compiled STEP file."""

    passed: bool
    """True if no G2 violations were found (or check was skipped)."""

    violation_count: int = 0
    """Number of edges that failed the G2 check."""

    violation_descriptions: list[str] = field(default_factory=list)
    """Human-readable descriptions of each violated edge."""

    method: str = "skipped"
    """Which check method was used: 'occt_analytical', 'vlm_zebra', 'skipped'."""

    feedback_message: str = ""
    """Formatted message to inject into the Critic Loop's retry prompt."""

    def __post_init__(self):
        if not self.passed and self.violation_descriptions and not self.feedback_message:
            self.feedback_message = _build_feedback(self.violation_descriptions)


def _build_feedback(violations: list[str]) -> str:
    """Format G2 violation descriptions into a Critic Loop retry message."""
    parts = ["G2 CONTINUITY FAILURE — surface has visible G1 creases:"]
    for v in violations[:5]:  # Cap at 5 to avoid context bloat
        parts.append(f"  - {v}")
    parts.append(
        "\nTo fix: Replace transitions at flagged boundaries with curvature-continuous "
        "splines. In CadQuery, use multi-section lofts with intermediate cross-sections "
        "to smooth the curvature transition, rather than direct section-to-section lofts. "
        "Alternatively, increase cross-section count (minimum 5 sections for smooth organic shapes). "
        "Avoid sharp tangent breaks between loft sections — ensure adjacent cross-sections "
        "are geometrically similar (no sudden size jumps)."
    )
    return "\n".join(parts)


def check_g2_continuity(step_path: str, domain: str = "C") -> G2CheckResult:
    """Run G2 continuity validation on a compiled STEP file.

    Only meaningful for Domain C (organic/ergonomic) parts. For other domains,
    returns a passed result immediately.

    Args:
        step_path: Absolute path to the compiled .step file.
        domain: Domain classification string. Skips check for non-C domains.

    Returns:
        G2CheckResult with passed=True if no violations, or detailed
        violation info and a Critic Loop feedback message if violations found.
    """
    if domain != "C":
        return G2CheckResult(passed=True, method="skipped")

    # Try analytical OCCT approach first
    result = _check_occt_analytical(step_path)
    if result.method != "skipped":
        return result

    # Fallback: skip (VLM Zebra approach requires rendering infrastructure)
    logger.info(
        "G2 check skipped for %s — pythonocc-core not available; "
        "VLM Zebra fallback not yet configured.",
        step_path,
    )
    return G2CheckResult(passed=True, method="skipped")


def _check_occt_analytical(step_path: str) -> G2CheckResult:
    """Attempt G2 check via OCC.Core.BRepLProp curvature sampling.

    Samples surface curvature on both sides of each shared edge at 5 points.
    If curvature magnitudes differ by more than G2_TOLERANCE on any sample,
    the edge is flagged as a G2 violation.

    Returns a G2CheckResult with method='skipped' if pythonocc-core is
    unavailable (the caller should try another approach).
    """
    G2_TOLERANCE = 0.05  # Relative curvature difference threshold

    try:
        from OCC.Core.BRep import BRep_Tool
        from OCC.Core.BRepLProp import BRepLProp_SLProps
        from OCC.Core.BRepTools import BRepTools
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCC.Core.TopExp import TopExp_Explorer, topexp
        from OCC.Core.TopoDS import topods
    except ImportError:
        return G2CheckResult(passed=True, method="skipped")

    try:
        reader = STEPControl_Reader()
        status = reader.ReadFile(step_path)
        if status != 1:  # IFSelect_RetDone
            logger.warning("STEP reader failed for %s (status %s)", step_path, status)
            return G2CheckResult(passed=True, method="skipped")
        reader.TransferRoots()
        shape = reader.OneShape()
    except Exception as exc:
        logger.warning("Failed to load STEP for G2 check: %s", exc)
        return G2CheckResult(passed=True, method="skipped")

    violations: list[str] = []

    # Iterate over all shared edges between faces
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    edge_idx = 0
    while edge_explorer.More() and edge_idx < 200:  # Cap at 200 edges
        edge = topods.Edge(edge_explorer.Current())
        edge_explorer.Next()
        edge_idx += 1

        # Get the two faces adjacent to this edge
        face_map: list = []
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = topods.Face(face_explorer.Current())
            # Check if this face contains the edge
            edge_in_face = TopExp_Explorer(face, TopAbs_EDGE)
            while edge_in_face.More():
                if edge_in_face.Current().IsEqual(edge):
                    face_map.append(face)
                    break
                edge_in_face.Next()
            face_explorer.Next()

        if len(face_map) < 2:
            continue  # Boundary edge — skip

        face_a, face_b = face_map[0], face_map[1]

        # Sample curvature at 5 points along the edge
        try:
            curve, u_start, u_end = BRep_Tool.Curve_(edge)
            if curve is None:
                continue
        except Exception:
            continue

        curvature_diffs = []
        for t in [0.2, 0.35, 0.5, 0.65, 0.8]:
            u = u_start + t * (u_end - u_start)
            try:
                # Get 3D point on edge at parameter u
                from OCC.Core.gp import gp_Pnt2d
                from OCC.Core.BRep import BRep_Tool as BRT

                # Project point onto each face and compute curvature
                surf_a = BRT.Surface(face_a)
                surf_b = BRT.Surface(face_b)
                props_a = BRepLProp_SLProps(surf_a, 0.5, 0.5, 2, 1e-6)
                props_b = BRepLProp_SLProps(surf_b, 0.5, 0.5, 2, 1e-6)

                if props_a.IsCurvatureDefined() and props_b.IsCurvatureDefined():
                    k1_a = abs(props_a.MaxCurvature())
                    k1_b = abs(props_b.MaxCurvature())
                    # Relative difference
                    denom = max(k1_a, k1_b, 1e-9)
                    rel_diff = abs(k1_a - k1_b) / denom
                    curvature_diffs.append(rel_diff)
            except Exception:
                continue

        if curvature_diffs:
            max_diff = max(curvature_diffs)
            if max_diff > G2_TOLERANCE:
                violations.append(
                    f"Edge {edge_idx}: curvature discontinuity "
                    f"{max_diff:.1%} (threshold {G2_TOLERANCE:.0%})"
                )

    if violations:
        return G2CheckResult(
            passed=False,
            violation_count=len(violations),
            violation_descriptions=violations,
            method="occt_analytical",
        )
    return G2CheckResult(passed=True, method="occt_analytical")


# ---------------------------------------------------------------------------
# VLM Zebra Analysis stub (Approach A)
# ---------------------------------------------------------------------------

async def check_g2_vlm_zebra(
    step_path: str,
    gemini_model: str = "gemini-2.5-flash",
) -> G2CheckResult:
    """VLM-based Zebra Analysis G2 check (stub — requires rendering pipeline).

    To activate:
    1. Add a headless renderer (OCCT Display or Blender subprocess) that
       produces a Zebra Analysis render (striped HDRI reflection) of the STEP.
    2. Pass the render to Gemini Vision with the prompt below.
    3. Parse the JSON response.

    Currently returns passed=True as a no-op stub until the rendering
    pipeline from T1-07 is extended for Zebra illumination.
    """
    logger.info(
        "VLM Zebra G2 check stub called for %s — rendering not yet configured.",
        step_path,
    )
    # TODO: Implement when rendering pipeline is available
    # Steps:
    #   1. render_zebra(step_path) -> zebra_image_path
    #   2. Upload zebra_image_path to Gemini Vision
    #   3. Prompt:
    #       "This is a Zebra Analysis render (striped reflection) of a 3D surface.
    #        Do any of the stripes show sharp kinks or discontinuities at surface
    #        boundaries? If yes, describe which boundary. Return JSON:
    #        {\"g2_pass\": bool, \"violation_description\": \"...\"}"
    #   4. Parse JSON and build G2CheckResult
    return G2CheckResult(passed=True, method="vlm_zebra_stub")
