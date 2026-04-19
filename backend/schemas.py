"""
Constraint-Based Assembly Schemas for Text-to-CAD Pipeline.

These Pydantic models define the strict inter-agent communication contract.
They are used as `response_schema` in Gemini API calls to enforce structured,
hallucination-resistant output from every subagent.

CORE RULE: All spatial relationships are TOPOLOGICAL and RELATIVE.
Absolute floating-point coordinates (x, y, z) are FORBIDDEN for mating.
Parts are connected via named anchor tags that reference CadQuery selectors.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class MateType(str, Enum):
    # Kinematic constraint types (core)
    FASTENED = "FASTENED"        # fully rigid — no relative motion
    REVOLUTE = "REVOLUTE"        # rotation about shared axis only
    SLIDER = "SLIDER"            # translation along shared axis only
    CYLINDRICAL = "CYLINDRICAL"  # rotation + translation along axis
    PLANAR = "PLANAR"            # translation in contact plane, free rotation
    BALL = "BALL"                # free rotation about contact point

    # Mechanism-level coupling constraints (T2-03)
    GEAR = "GEAR"                # coupled revolute: angular velocity ratio
    SCREW = "SCREW"              # coupled revolute + slider: pitch rate
    RACK_PINION = "RACK_PINION"  # revolute ↔ slider coupling
    CAM = "CAM"                  # surface-contact (follower on cam profile)


class PartDefinition(BaseModel):
    """A single mechanical part to be manufactured by the Machinist subagent.

    Each part is generated in COMPLETE ISOLATION — the Machinist receives only
    this definition and must produce a self-contained CadQuery script that
    creates the part centered at the origin. The part's relationship to the
    broader assembly is handled entirely by anchor_tags and MatingRules —
    the Machinist must NEVER embed assembly-level positioning logic.

    EXCEPTION — Coupled Geometric Subsystems (Domain D Aerospace):
    When a part represents a tightly coupled aerodynamic subsystem (e.g., a
    wing segment with internal ribs and spars), the Machinist is permitted —
    and REQUIRED — to generate the entire subsystem within a single self-
    contained script. This is necessary because internal structures depend on
    a shared tooling body (the ``inner_void`` — the directly-lofted airfoil
    core solid). Splitting such a subsystem across multiple PartDefinition
    entries makes it impossible to share that master geometry, causing ribs
    to clip through shell walls. In these cases the Machinist must produce a
    ``cq.Assembly`` containing all sub-bodies (skin, ribs, spars) and assign
    it to ``result``.
    """

    part_id: str = Field(
        ...,
        max_length=200,
        description=(
            "A unique, snake_case identifier for this part (e.g., 'base_plate', "
            "'motor_housing', 'm5_bolt'). This ID is referenced by MatingRules "
            "to connect parts in the assembly. Must be unique across the entire "
            "AssemblyManifest."
        ),
    )

    description: str = Field(
        ...,
        max_length=2000,
        description=(
            "A detailed physical description of the part that the Machinist "
            "subagent will use to write CadQuery code. Include ALL geometric "
            "details: dimensions (in millimeters), shape primitives (box, "
            "cylinder, etc.), features (holes, fillets, chamfers, slots), and "
            "material intent. Be explicit and unambiguous. "
            "Example: 'A 100mm x 60mm x 5mm rectangular base plate with four "
            "M5 through-holes at each corner, 8mm from each edge, and a 2mm "
            "fillet on all top edges.' "
            "DO NOT include any absolute (x, y, z) positioning relative to "
            "other parts. Each part is modeled at the origin in isolation."
        ),
    )

    anchor_tags: list[str] = Field(
        ...,
        description=(
            "A list of named anchors on this part where other parts can attach. "
            "Each anchor tag MUST be one of the following: "
            "(1) A valid CadQuery face/edge selector string such as '>Z' (top "
            "face), '<Z' (bottom face), '>X' (right face), '<X' (left face), "
            "'>Y' (front face), '<Y' (back face). "
            "(2) A descriptive named tag for a specific feature, e.g., "
            "'hole_center', 'slot_midpoint', 'flange_top'. The Machinist will "
            "use cq.Assembly().tag() to register these in the CadQuery script. "
            "CRITICAL: Anchor tags are TOPOLOGICAL REFERENCES, not coordinates. "
            "NEVER use tuples like '(10, 20, 0)' or any numeric coordinate as "
            "an anchor tag. The Assembler subagent uses these tags to mate "
            "parts together via CadQuery's constraint system."
        ),
    )


class MatingRule(BaseModel):
    """A constraint that connects two parts and specifies placement coordinates.

    MatingRules define both the logical connection between parts (via anchors)
    and the precise 3D translation for the Assembler to position each part.
    """

    source_part_id: str = Field(
        ...,
        description=(
            "The part_id of the first part in this mating pair. Must exactly "
            "match a part_id defined in the AssemblyManifest.parts list."
        ),
    )

    source_anchor: str = Field(
        ...,
        description=(
            "The anchor tag on the source part where the connection originates. "
            "Must exactly match one of the anchor_tags defined in the source "
            "part's PartDefinition. Example: '>Z'."
        ),
    )

    target_part_id: str = Field(
        ...,
        description=(
            "The part_id of the second part in this mating pair. Must exactly "
            "match a part_id defined in the AssemblyManifest.parts list. "
            "Must be different from source_part_id."
        ),
    )

    target_anchor: str = Field(
        ...,
        description=(
            "The anchor tag on the target part where the connection terminates. "
            "Must exactly match one of the anchor_tags defined in the target "
            "part's PartDefinition. Example: '<Z'."
        ),
    )

    mate_type: Optional[MateType] = Field(
        default=None,
        description=(
            "The kinematic constraint type. When provided, the Assembler uses "
            "cq.Constraint-based solving instead of absolute XYZ positioning. "
            "FASTENED = fully rigid (no relative motion). "
            "REVOLUTE = rotation about shared axis only. "
            "SLIDER = translation along shared axis only. "
            "CYLINDRICAL = rotation + translation along axis. "
            "PLANAR = translation in contact plane. "
            "BALL = free rotation about contact point. "
            "Prefer mate_type over translation for all new assemblies."
        ),
    )

    translation: Optional[str] = Field(
        default=None,
        description=(
            "DEPRECATED — use mate_type instead. "
            "Absolute 3D translation 'X, Y, Z' in mm for the target part. "
            "Used only when mate_type is not provided (backward compatibility). "
            "The base/first part is always at '0, 0, 0'."
        ),
    )

    clearance: float = Field(
        ...,
        description=(
            "Gap in millimeters between the two mated surfaces. "
            "Use 0.0 for flush contact. Use small positive values for "
            "mechanical clearance (e.g., 0.1 for a slip fit, 0.5 for a loose "
            "fit)."
        ),
    )

    # T2-02: DoF range annotation for bounded kinematic constraints
    dof_min: Optional[float] = Field(
        default=None,
        description=(
            "Minimum range for the unconstrained degree of freedom. "
            "Degrees for REVOLUTE/CYLINDRICAL, mm for SLIDER/CYLINDRICAL. "
            "Example: 0.0 for a hinge that cannot close past flush. "
            "Leave None for unbounded joints (bearings, free sliders)."
        ),
    )

    dof_max: Optional[float] = Field(
        default=None,
        description=(
            "Maximum range for the unconstrained degree of freedom. "
            "Degrees for REVOLUTE/CYLINDRICAL, mm for SLIDER/CYLINDRICAL. "
            "Example: 90.0 for a hinge that opens 90°. "
            "Leave None for unbounded joints."
        ),
    )

    dof_unit: Optional[str] = Field(
        default=None,
        description=(
            "Unit for dof_min / dof_max. Must be 'deg' (rotational) or "
            "'mm' (translational). Required when dof_min or dof_max is set."
        ),
    )

    # T2-03: Mechanical coupling parameters for GEAR, SCREW, RACK_PINION
    coupling_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Coupling ratio for mechanism-level mate types. "
            "GEAR: teeth_driven / teeth_driver (e.g., 2.0 = 2:1 reduction). "
            "SCREW: lead in mm per revolution (e.g., 1.5 for M1.5 pitch). "
            "RACK_PINION: mm of rack travel per radian of pinion rotation. "
            "Not applicable to kinematic constraint types (FASTENED, REVOLUTE, etc.)."
        ),
    )

    @model_validator(mode="after")
    def _require_mate_type_or_translation(self) -> "MatingRule":
        if self.mate_type is None and self.translation is None:
            raise ValueError(
                "MatingRule must have either 'mate_type' or 'translation'. "
                "Provide mate_type (preferred) for constraint-based assembly, "
                "or translation 'X, Y, Z' for legacy positioning."
            )
        return self


class AssemblyManifest(BaseModel):
    """The complete blueprint for a multi-part mechanical assembly.

    This is the output of the Draftsman (Planner) subagent and the input to
    the orchestration pipeline. It defines WHAT to build (parts) and HOW to
    connect them (mating_rules).

    The manifest is the single source of truth for the entire assembly. Every
    downstream subagent receives only its relevant slice of this manifest:
    - The Machinist gets one PartDefinition at a time.
    - The Assembler gets all validated part scripts + the mating_rules.

    ANTI-HALLUCINATION CONTRACT:
    - Parts must be self-contained and origin-centered.
    - All inter-part spatial relationships are encoded ONLY in mating_rules.
    - No absolute coordinates anywhere in this manifest.
    """

    assembly_name: str = Field(
        ...,
        max_length=200,
        description=(
            "A descriptive, snake_case name for the overall assembly "
            "(e.g., 'flanged_bearing_mount', 'gearbox_housing'). Used as the "
            "filename stem for the exported .glb file."
        ),
    )

    parts: list[PartDefinition] = Field(
        ...,
        description=(
            "An ordered list of all individual parts in the assembly. Each part "
            "will be manufactured independently by the Machinist subagent. "
            "Every part_id must be unique. List parts in logical build order "
            "(base/foundation first, fasteners last)."
        ),
    )

    mating_rules: list[MatingRule] = Field(
        ...,
        description=(
            "The list of constraints that define how parts connect to each "
            "other. Every mating rule references exactly two parts by their "
            "part_id and specifies which anchor_tags to mate. The Assembler "
            "subagent translates these into CadQuery constraint calls. "
            "RULES: "
            "(1) All referenced part_ids must exist in the parts list. "
            "(2) All referenced anchors must exist in the respective part's "
            "anchor_tags. "
            "(3) The assembly graph must be connected — no orphan parts. "
            "(4) NEVER encode absolute positions. Positioning is emergent "
            "from the chain of relative constraints."
        ),
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> "AssemblyManifest":  # noqa: C901
        # --- Size limits (not expressible in Gemini's schema) ---
        if len(self.parts) > 50:
            raise ValueError(
                f"Too many parts ({len(self.parts)}). Maximum is 50."
            )
        if len(self.mating_rules) > 100:
            raise ValueError(
                f"Too many mating rules ({len(self.mating_rules)}). Maximum is 100."
            )
        for part in self.parts:
            if len(part.anchor_tags) > 20:
                raise ValueError(
                    f"Part '{part.part_id}' has {len(part.anchor_tags)} "
                    f"anchor tags. Maximum is 20."
                )

        # --- Referential integrity ---
        valid_ids = {p.part_id for p in self.parts}
        for rule in self.mating_rules:
            if rule.source_part_id not in valid_ids:
                raise ValueError(
                    f"Mating rule references undefined part: {rule.source_part_id}"
                )
            if rule.target_part_id not in valid_ids:
                raise ValueError(
                    f"Mating rule references undefined part: {rule.target_part_id}"
                )
        return self


# ---------------------------------------------------------------------------
# T2-12: Requirements Engineering Agent (REA) schema
# ---------------------------------------------------------------------------

class DesignRequirements(BaseModel):
    """Structured requirements extracted from a user's design brief.

    The REA parses the raw prompt into this document before it reaches
    the Planner. The Planner receives both the raw prompt AND this
    structured document, treating the requirements as the authoritative spec.
    """

    primary_function: str = Field(
        ...,
        description=(
            "One sentence describing what the assembly or part must do. "
            "Example: 'A rotating joint to connect two structural arms with 90° range of motion.'"
        ),
    )

    key_dimensions: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Map of dimension name → value in mm. Only include dimensions "
            "explicitly stated or clearly implied by the prompt. "
            "Example: {'shaft_diameter_mm': 20.0, 'total_length_mm': 150.0}"
        ),
    )

    material_class: Optional[str] = Field(
        default=None,
        description=(
            "Material family inferred from context. "
            "One of: aluminum, steel, stainless_steel, plastic, carbon_fiber, "
            "titanium, brass, ceramic, rubber, or None if unspecified."
        ),
    )

    environment: Optional[str] = Field(
        default=None,
        description=(
            "Operating environment. "
            "One of: standard, outdoor, corrosive, high_temp, cryogenic, "
            "vacuum, underwater, or None if unspecified."
        ),
    )

    connecting_interfaces: list[str] = Field(
        default_factory=list,
        description=(
            "Named interfaces or features that mate with external parts. "
            "Example: ['M8 through bolt holes on flange', 'keyed bore for 20mm shaft']"
        ),
    )

    production_volume: Optional[str] = Field(
        default=None,
        description=(
            "Expected production scale. "
            "One of: prototype, low_volume, production, or None if unspecified."
        ),
    )

    inferred_domain: Optional[str] = Field(
        default=None,
        description=(
            "Predicted geometric domain. "
            "A = structural/prismatic, B = mechanical/parametric, "
            "C = organic/ergonomic, D = aerospace/aerodynamic."
        ),
    )

    confidence: float = Field(
        default=1.0,
        description=(
            "REA confidence score 0–1. Below 0.5 means the prompt is too vague "
            "to extract reliable requirements — the Clarifier should be invoked."
        ),
    )


# ---------------------------------------------------------------------------
# T2-08: Multi-turn refinement schemas
# ---------------------------------------------------------------------------

class ManifestDiff(BaseModel):
    """Differential update to an existing AssemblyManifest.

    The RefinerAgent emits only the changes needed — unchanged parts are
    omitted. The orchestrator merges this diff with the original manifest
    before re-running affected Machinists.
    """

    modified_parts: list[str] = Field(
        default_factory=list,
        description=(
            "part_ids of parts whose description has changed and must be "
            "re-generated by the Machinist. Scripts for all other parts "
            "are reused from the original run."
        ),
    )

    added_parts: list[PartDefinition] = Field(
        default_factory=list,
        description="New PartDefinition entries to add to the assembly.",
    )

    removed_parts: list[str] = Field(
        default_factory=list,
        description="part_ids to remove from the assembly and manifest.",
    )

    modified_mates: list[MatingRule] = Field(
        default_factory=list,
        description=(
            "Updated MatingRule entries. Rules are matched by "
            "(source_part_id, target_part_id) pair. New rules are appended; "
            "existing rules with matching IDs are replaced."
        ),
    )

    updated_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of part_id -> updated description for modified parts. "
            "The Machinist uses the updated description; the part_id is unchanged."
        ),
    )


class RefinementRequest(BaseModel):
    """Request body for the /refine endpoint."""

    original_manifest: AssemblyManifest = Field(
        ...,
        description="The AssemblyManifest from the original /generate run.",
    )

    original_scripts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of part_id -> CadQuery Python script from the original run. "
            "Scripts for unchanged parts are reused directly."
        ),
    )

    refinement_prompt: str = Field(
        ...,
        max_length=2000,
        description=(
            "Natural-language description of the desired change. "
            "Examples: 'make the shaft 20% longer', 'add a chamfer to the "
            "top flange', 'replace the ball joint with a revolute joint'."
        ),
    )

    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for telemetry grouping of refinement chains.",
    )
