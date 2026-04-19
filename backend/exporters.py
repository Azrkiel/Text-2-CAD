"""
T2-09: Multi-Format CAD Export Module.

Provides export functions for output formats beyond GLB:
  - STEP (.step)    — native OCCT, always available
  - Parasolid (.x_t) — requires CAD Exchanger SDK (commercial license)
  - STL (.stl)      — native OCCT, for 3D printing workflows
  - DXF (.dxf)      — 2D drawing export via CadQuery

SOLIDWORKS PARASOLID EXPORT (T2-09):
  Parasolid is the native kernel format for SOLIDWORKS. Importing a Parasolid
  .x_t file into SOLIDWORKS is more reliable than importing STEP (no topology
  loss, no conversion artifacts). This is the highest-value export for the ICP.

  DEPENDENCY: CAD Exchanger SDK (commercial license required).
    - Developer evaluation: https://cadexchanger.com/pricing/
    - Python bindings: pip install cadexchanger (requires license file)
    - Alternative: Open CASCADE XCAF with Parasolid plugin (unavailable open-source)

  CURRENT STATUS: Scaffold with graceful fallback to STEP if CAD Exchanger
  is not installed. Set the CADEXCHANGER_LICENSE_KEY environment variable
  or place a license.lic file in the working directory.

Usage from pipeline:
    from exporters import export_step, export_parasolid, export_stl
    step_ok = export_step(cadquery_workplane, "output.step")
    xt_ok   = export_parasolid("output.step", "output.x_t")
"""

import logging
import os
from pathlib import Path
from typing import Union

logger = logging.getLogger("mirum.exporters")

# ---------------------------------------------------------------------------
# STEP export (always available via CadQuery / OCCT)
# ---------------------------------------------------------------------------

def export_step(shape, output_path: str) -> bool:
    """Export a CadQuery Workplane or Assembly to STEP format.

    Args:
        shape: cq.Workplane or cq.Assembly object.
        output_path: Absolute path for the output .step file.

    Returns:
        True on success, False on failure.
    """
    try:
        import cadquery as cq
        if isinstance(shape, cq.Assembly):
            shape.save(output_path, exportType="STEP")
        else:
            cq.exporters.export(shape, output_path, exportType="STEP")
        logger.debug("STEP export successful: %s", output_path)
        return True
    except Exception as exc:
        logger.error("STEP export failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Parasolid export (requires CAD Exchanger SDK — commercial license)
# ---------------------------------------------------------------------------

_CADEX_AVAILABLE: bool | None = None  # None = not yet checked


def _check_cadex() -> bool:
    """Check if CAD Exchanger SDK is available and licensed."""
    global _CADEX_AVAILABLE
    if _CADEX_AVAILABLE is not None:
        return _CADEX_AVAILABLE

    try:
        import cadexchanger as cadex  # noqa: F401
        # Attempt to activate license
        license_key = os.environ.get("CADEXCHANGER_LICENSE_KEY", "")
        license_path = Path("license.lic")
        if license_key:
            # Inline license key activation
            cadex.LicenseManager.Activate(license_key)
        elif license_path.exists():
            cadex.LicenseManager.ActivateFile(str(license_path.resolve()))
        _CADEX_AVAILABLE = True
        logger.info("CAD Exchanger SDK available and licensed")
    except ImportError:
        _CADEX_AVAILABLE = False
        logger.info(
            "CAD Exchanger SDK not installed — Parasolid export unavailable. "
            "Install: pip install cadexchanger (requires commercial license). "
            "See: https://cadexchanger.com/pricing/"
        )
    except Exception as exc:
        _CADEX_AVAILABLE = False
        logger.warning("CAD Exchanger license activation failed: %s", exc)

    return _CADEX_AVAILABLE


def export_parasolid(
    step_path: str,
    output_path: str,
    fallback_to_step: bool = True,
) -> tuple[bool, str]:
    """Export a STEP file to Parasolid .x_t format via CAD Exchanger SDK.

    The STEP file is first read into the CAD Exchanger model, then
    re-exported as Parasolid. This performs a kernel-level conversion
    with full B-rep fidelity (no tessellation).

    Args:
        step_path: Absolute path to the input .step file.
        output_path: Absolute path for the output .x_t file.
        fallback_to_step: If True and CAD Exchanger is unavailable, copy
            the STEP file to output_path with .step extension and return
            (False, step_copy_path) so the caller can provide a download.

    Returns:
        (success: bool, actual_output_path: str)
        actual_output_path may be a .step file if fallback was used.
    """
    if not _check_cadex():
        if fallback_to_step:
            # Provide STEP as fallback download
            step_fallback = output_path.replace(".x_t", ".step")
            try:
                import shutil
                shutil.copy2(step_path, step_fallback)
                logger.info(
                    "Parasolid unavailable — serving STEP fallback: %s", step_fallback
                )
                return False, step_fallback
            except Exception as exc:
                logger.error("STEP fallback copy failed: %s", exc)
                return False, ""
        return False, ""

    try:
        import cadexchanger as cadex

        # Read STEP into CAD Exchanger model
        model = cadex.ModelData_Model()
        reader = cadex.STEP_Reader()
        if not reader.Read(cadex.Base_UTF16String(step_path), model):
            raise RuntimeError(f"CAD Exchanger failed to read STEP: {step_path}")

        # Write Parasolid .x_t
        writer = cadex.Parasolid_Writer()
        if not writer.Write(model, cadex.Base_UTF16String(output_path)):
            raise RuntimeError(f"CAD Exchanger failed to write Parasolid: {output_path}")

        logger.info("Parasolid export successful: %s", output_path)
        return True, output_path

    except Exception as exc:
        logger.error("Parasolid export failed: %s", exc)
        if fallback_to_step:
            step_fallback = output_path.replace(".x_t", ".step")
            try:
                import shutil
                shutil.copy2(step_path, step_fallback)
                return False, step_fallback
            except Exception:
                pass
        return False, ""


# ---------------------------------------------------------------------------
# STL export (native OCCT — for 3D printing)
# ---------------------------------------------------------------------------

def export_stl(shape, output_path: str, tolerance: float = 0.1) -> bool:
    """Export a CadQuery Workplane to STL format.

    Args:
        shape: cq.Workplane object.
        output_path: Absolute path for the output .stl file.
        tolerance: Mesh tolerance in mm (lower = finer mesh).

    Returns:
        True on success, False on failure.
    """
    try:
        import cadquery as cq
        cq.exporters.export(
            shape, output_path,
            exportType="STL",
            tolerance=tolerance,
            angularTolerance=0.1,
        )
        logger.debug("STL export successful: %s", output_path)
        return True
    except Exception as exc:
        logger.error("STL export failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# DXF export (2D projection — for drawing generation)
# ---------------------------------------------------------------------------

def export_dxf(shape, output_path: str) -> bool:
    """Export a 2D projection of a CadQuery Workplane to DXF format.

    Projects the top face (>Z) of the shape onto the XY plane.

    Args:
        shape: cq.Workplane object.
        output_path: Absolute path for the output .dxf file.

    Returns:
        True on success, False on failure.
    """
    try:
        import cadquery as cq
        # Project top face to DXF
        projected = shape.section()
        cq.exporters.export(projected, output_path, exportType="DXF")
        logger.debug("DXF export successful: %s", output_path)
        return True
    except Exception as exc:
        logger.error("DXF export failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Export availability check for API response
# ---------------------------------------------------------------------------

def get_available_formats() -> dict[str, bool]:
    """Return which export formats are available in this installation."""
    return {
        "glb":       True,           # Always available (CadQuery)
        "step":      True,           # Always available (OCCT)
        "stl":       True,           # Always available (OCCT)
        "dxf":       True,           # Always available (CadQuery)
        "parasolid": _check_cadex(), # Requires CAD Exchanger SDK
    }
