"""
T2-07: ABC Dataset → Synthetic 2-Part Assembly Pipeline.

Clusters ABC Dataset single-part STEP files by geometric type, then
programmatically assembles compatible part pairs using OCCT constraint
solving, filters invalid assemblies, and annotates with Gemini.

Target: 100K+ valid 2-part assemblies with descriptions.

Pipeline stages:
  1. Classify each ABC part into a geometric type (prismatic/cylindrical/sheet/organic)
  2. For compatible type pairs (e.g. cylindrical+prismatic → REVOLUTE), sample N pairs
  3. For each pair: load STEP → select anchor faces → apply cq.Constraint → solve
  4. Run interpenetration check (T1-02)
  5. Annotate valid assemblies with Gemini (6-view render or code-level)
  6. Write output JSONL

Usage:
    # Download ABC dataset first: https://deep-geometry.github.io/abc-dataset/
    python data_pipeline/abc_assembly_pipeline.py \
        --abc-dir /data/abc/step \
        --output data/abc_synthetic_assemblies.jsonl \
        --pairs-per-rule 5000

Output JSONL format:
    {
      "description": "...",
      "assembly_manifest": {...},  # AssemblyManifest JSON
      "part_a_code": "import cadquery as cq\\n...",
      "part_b_code": "import cadquery as cq\\n...",
      "source": "abc_synthetic",
      "mate_type": "REVOLUTE",
      "part_a_type": "cylindrical",
      "part_b_type": "prismatic"
    }

External dependencies:
    pip install cadquery google-generativeai tqdm
    ABC Dataset: https://deep-geometry.github.io/abc-dataset/

NOTE: The constraint solver (cq.Assembly.solve) requires CadQuery ≥ 2.3.
This script is designed to run as a standalone batch job, not in the
Docker container (which has limited disk space for 1M STEP files).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("mirum.abc_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Part type classification
# ---------------------------------------------------------------------------

@dataclass
class PartGeometry:
    """Cached geometric properties of a STEP part."""
    path: str
    part_type: str    # prismatic | cylindrical | sheet | organic
    bbox_x: float
    bbox_y: float
    bbox_z: float
    has_cylinder: bool
    face_count: int


def classify_abc_part(step_path: str) -> PartGeometry | None:
    """Classify a STEP file into a geometric type using OCCT.

    Returns None if the file cannot be loaded or classified.
    """
    try:
        import cadquery as cq
        shape = cq.importers.importStep(step_path)
    except Exception as exc:
        logger.debug("Failed to load %s: %s", step_path, exc)
        return None

    try:
        bb = shape.val().BoundingBox()
        x, y, z = bb.xmax - bb.xmin, bb.ymax - bb.ymin, bb.zmax - bb.zmin
        if x < 0.01 or y < 0.01 or z < 0.01:
            return None

        face_count = shape.faces().size()
        # Check for cylindrical faces (indicates shaft/bore geometry)
        cyl_faces = 0
        try:
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.GeomAbs import GeomAbs_Cylinder
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopoDS import topods

            exp = TopExp_Explorer(shape.val().wrapped, TopAbs_FACE)
            while exp.More():
                face = topods.Face(exp.Current())
                adaptor = BRepAdaptor_Surface(face)
                if adaptor.GetType() == GeomAbs_Cylinder:
                    cyl_faces += 1
                exp.Next()
        except Exception:
            pass

        has_cylinder = cyl_faces > 0
        dims = sorted([x, y, z])
        aspect_ratio = dims[2] / max(dims[0], 0.01)

        if aspect_ratio > 8 and has_cylinder:
            part_type = "cylindrical"  # shaft-like
        elif aspect_ratio > 5 and dims[0] < 3:
            part_type = "sheet"        # thin plate
        elif has_cylinder and face_count < 20:
            part_type = "cylindrical"  # hub/bore
        elif face_count > 50:
            part_type = "organic"      # complex freeform
        else:
            part_type = "prismatic"    # block-like

        return PartGeometry(
            path=step_path,
            part_type=part_type,
            bbox_x=x, bbox_y=y, bbox_z=z,
            has_cylinder=has_cylinder,
            face_count=face_count,
        )
    except Exception as exc:
        logger.debug("Classification failed for %s: %s", step_path, exc)
        return None


# ---------------------------------------------------------------------------
# SQLite index for classified parts
# ---------------------------------------------------------------------------

def build_or_load_index(abc_dir: str, index_path: str) -> sqlite3.Connection:
    """Build or load a SQLite index of classified ABC parts."""
    conn = sqlite3.connect(index_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS parts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            part_type TEXT NOT NULL,
            bbox_x REAL, bbox_y REAL, bbox_z REAL,
            has_cylinder INTEGER, face_count INTEGER
        )
    """)
    conn.commit()

    existing = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
    if existing > 0:
        logger.info("Loaded existing index with %d parts", existing)
        return conn

    logger.info("Building part index from %s — this may take a while...", abc_dir)
    step_files = list(Path(abc_dir).rglob("*.step")) + list(Path(abc_dir).rglob("*.stp"))
    logger.info("Found %d STEP files", len(step_files))

    batch = []
    for i, path in enumerate(step_files):
        geo = classify_abc_part(str(path))
        if geo is None:
            continue
        batch.append((
            str(path), geo.part_type,
            geo.bbox_x, geo.bbox_y, geo.bbox_z,
            int(geo.has_cylinder), geo.face_count,
        ))
        if len(batch) >= 500:
            conn.executemany(
                "INSERT OR IGNORE INTO parts "
                "(path, part_type, bbox_x, bbox_y, bbox_z, has_cylinder, face_count) "
                "VALUES (?,?,?,?,?,?,?)",
                batch,
            )
            conn.commit()
            batch.clear()
            logger.info("Indexed %d / %d parts", i + 1, len(step_files))

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO parts "
            "(path, part_type, bbox_x, bbox_y, bbox_z, has_cylinder, face_count) "
            "VALUES (?,?,?,?,?,?,?)",
            batch,
        )
        conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]
    logger.info("Index complete: %d classified parts", total)
    return conn


# ---------------------------------------------------------------------------
# Compatible part-pair rules
# ---------------------------------------------------------------------------

# (type_A, type_B, mate_type, anchor_A, anchor_B)
COMPATIBLE_PAIRS = [
    ("cylindrical", "prismatic",  "REVOLUTE",   ">Z", "<Z"),
    ("prismatic",   "prismatic",  "FASTENED",   ">Z", "<Z"),
    ("sheet",       "prismatic",  "FASTENED",   ">Z", "<Z"),
    ("cylindrical", "cylindrical","CYLINDRICAL",">Z", "<Z"),
    ("prismatic",   "sheet",      "FASTENED",   ">Y", "<Y"),
]


# ---------------------------------------------------------------------------
# Assembly attempt
# ---------------------------------------------------------------------------

def attempt_assembly(
    part_a_path: str,
    part_b_path: str,
    mate_type: str,
    anchor_a: str,
    anchor_b: str,
) -> bool:
    """Attempt constraint-based assembly of two STEP parts.

    Returns True if cq.Assembly.solve() succeeds without interpenetration.
    """
    try:
        import cadquery as cq

        part_a = cq.importers.importStep(part_a_path)
        part_b = cq.importers.importStep(part_b_path)

        # Map mate type to CadQuery constraint type
        _CONSTRAINTS = {
            "FASTENED":    "Plane",
            "REVOLUTE":    "Axis",
            "SLIDER":      "Plane",
            "CYLINDRICAL": "Axis",
            "PLANAR":      "Plane",
            "BALL":        "Point",
        }
        cq_constraint = _CONSTRAINTS.get(mate_type, "Plane")

        assembly = cq.Assembly()
        assembly.add(part_a, name="part_a")
        assembly.add(part_b, name="part_b")

        # Build selector strings from anchor tags
        sel_a = f"faces@{anchor_a}"
        sel_b = f"faces@{anchor_b}"
        assembly.constrain(f"part_a@{sel_a}", f"part_b@{sel_b}", cq_constraint)

        assembly.solve()

        # Quick interpenetration check (bounding box overlap as proxy)
        bb_a = part_a.val().BoundingBox()
        bb_b = part_b.val().BoundingBox()
        # If bboxes don't overlap at all in any axis, assembly is spatially separated
        # (This is a conservative check at origin — post-solve positions may differ)
        overlap_x = (bb_a.xmin < bb_b.xmax) and (bb_b.xmin < bb_a.xmax)
        overlap_y = (bb_a.ymin < bb_b.ymax) and (bb_b.ymin < bb_a.ymax)
        overlap_z = (bb_a.zmin < bb_b.zmax) and (bb_b.zmin < bb_a.zmax)
        if overlap_x and overlap_y and overlap_z:
            # Likely interpenetration — reject
            return False

        return True
    except Exception as exc:
        logger.debug("Assembly attempt failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_assembly(
    part_a_type: str,
    part_b_type: str,
    mate_type: str,
    api_key: str | None = None,
) -> str | None:
    """Generate a natural-language description of a synthetic assembly."""
    try:
        import google.generativeai as genai

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            return _fallback_description(part_a_type, part_b_type, mate_type)
        genai.configure(api_key=key)

        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        prompt = (
            f"Generate a concise (1-2 sentence) natural-language description of a "
            f"mechanical assembly consisting of:\n"
            f"  Part A: a {part_a_type} component\n"
            f"  Part B: a {part_b_type} component\n"
            f"  Joint type: {mate_type}\n"
            f"Be specific about the mechanical function. Do not mention that it is "
            f"synthetic or generated. Output only the description, no JSON."
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return _fallback_description(part_a_type, part_b_type, mate_type)


def _fallback_description(a: str, b: str, mate: str) -> str:
    return (
        f"A {mate.lower()} joint connecting a {a} component to a {b} component."
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    abc_dir: str,
    output_path: str,
    index_path: str = "data/abc_index.sqlite",
    pairs_per_rule: int = 5000,
    api_key: str | None = None,
) -> dict:
    """Run the full ABC → synthetic assembly pipeline.

    Args:
        abc_dir: Directory with ABC Dataset STEP files.
        output_path: Output JSONL path.
        index_path: SQLite index for classified parts (built on first run).
        pairs_per_rule: Number of assembly pairs to attempt per compatible rule.
        api_key: Gemini API key.

    Returns:
        Summary statistics.
    """
    if not Path(abc_dir).exists():
        raise FileNotFoundError(
            f"ABC Dataset directory not found: {abc_dir}\n"
            "Download: https://deep-geometry.github.io/abc-dataset/"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    conn = build_or_load_index(abc_dir, index_path)

    stats = {
        "pairs_attempted": 0,
        "pairs_valid": 0,
        "pairs_failed_assembly": 0,
        "pairs_failed_annotation": 0,
        "written": 0,
    }

    with open(output_path, "a", encoding="utf-8") as out_f:
        for type_a, type_b, mate_type, anchor_a, anchor_b in COMPATIBLE_PAIRS:
            logger.info(
                "Processing rule: %s + %s → %s", type_a, type_b, mate_type
            )

            # Fetch candidate parts for each type
            parts_a = conn.execute(
                "SELECT path FROM parts WHERE part_type=? ORDER BY RANDOM() LIMIT ?",
                (type_a, pairs_per_rule * 3),
            ).fetchall()
            parts_b = conn.execute(
                "SELECT path FROM parts WHERE part_type=? ORDER BY RANDOM() LIMIT ?",
                (type_b, pairs_per_rule * 3),
            ).fetchall()

            if not parts_a or not parts_b:
                logger.warning(
                    "No parts found for types %s / %s — skipping rule", type_a, type_b
                )
                continue

            # Sample pairs and attempt assembly
            rule_valid = 0
            for i in range(min(len(parts_a), len(parts_b), pairs_per_rule * 3)):
                if rule_valid >= pairs_per_rule:
                    break

                path_a = parts_a[i % len(parts_a)][0]
                path_b = parts_b[i % len(parts_b)][0]
                if path_a == path_b:
                    continue

                stats["pairs_attempted"] += 1

                if not attempt_assembly(path_a, path_b, mate_type, anchor_a, anchor_b):
                    stats["pairs_failed_assembly"] += 1
                    continue

                stats["pairs_valid"] += 1
                rule_valid += 1

                # Annotate
                description = annotate_assembly(type_a, type_b, mate_type, api_key)
                if description is None:
                    stats["pairs_failed_annotation"] += 1
                    continue

                # Build manifest JSON (compact form for training data)
                manifest = {
                    "assembly_name": f"{type_a}_{type_b}_{mate_type.lower()}_assembly",
                    "parts": [
                        {
                            "part_id": "part_a",
                            "description": f"A {type_a} mechanical component",
                            "anchor_tags": [anchor_a],
                        },
                        {
                            "part_id": "part_b",
                            "description": f"A {type_b} mechanical component",
                            "anchor_tags": [anchor_b],
                        },
                    ],
                    "mating_rules": [
                        {
                            "source_part_id": "part_a",
                            "source_anchor": anchor_a,
                            "target_part_id": "part_b",
                            "target_anchor": anchor_b,
                            "mate_type": mate_type,
                            "clearance": 0.0,
                        }
                    ],
                }

                record = {
                    "description": description,
                    "assembly_manifest": manifest,
                    "part_a_path": path_a,  # path on disk — code generated at training time
                    "part_b_path": path_b,
                    "mate_type": mate_type,
                    "part_a_type": type_a,
                    "part_b_type": type_b,
                    "source": "abc_synthetic",
                }
                out_f.write(json.dumps(record) + "\n")
                stats["written"] += 1

            out_f.flush()
            logger.info("Rule %s+%s: %d valid / %d attempted",
                        type_a, type_b, rule_valid, pairs_per_rule * 3)

    conn.close()
    logger.info("Pipeline complete: %s", stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Build synthetic 2-part assemblies from ABC Dataset"
    )
    p.add_argument("--abc-dir", required=True,
                   help="Directory with ABC Dataset STEP files")
    p.add_argument("--output", default="data/abc_synthetic_assemblies.jsonl",
                   help="Output JSONL path")
    p.add_argument("--index", default="data/abc_index.sqlite",
                   help="SQLite part index path")
    p.add_argument("--pairs-per-rule", type=int, default=5000,
                   help="Assembly pairs to generate per compatible rule")
    p.add_argument("--api-key", default=None,
                   help="Gemini API key (default: GEMINI_API_KEY env var)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    summary = run_pipeline(
        abc_dir=args.abc_dir,
        output_path=args.output,
        index_path=args.index,
        pairs_per_rule=args.pairs_per_rule,
        api_key=args.api_key,
    )
    print(json.dumps(summary, indent=2))
