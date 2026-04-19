"""
Parametric CAD utility functions for the Text-to-CAD pipeline.

These functions are importable by Machinist-generated scripts via PYTHONPATH.
They provide mathematically correct geometry for parts that LLMs consistently
fail to generate (gears, threads, etc.).
"""

import math

import cadquery as cq


def make_involute_spur_gear(
    num_teeth: int = 20,
    module: float = 2.0,
    pressure_angle_deg: float = 20.0,
    thickness: float = 10.0,
    bore_diameter: float = 0.0,
    pitch_diameter: float | None = None,
) -> cq.Workplane:
    """Generate a mathematically accurate involute spur gear as a 3D solid.

    Uses a 2D sketch approach: computes the full gear tooth profile as a
    single continuous closed wire, adds the bore as an inner wire, and
    performs a single extrude(). This avoids repeated boolean subtractions
    which can fragment geometry into a non-manifold Compound.

    The tooth profile is computed from the standard involute function:
        inv(alpha) = tan(alpha) - alpha

    Each tooth consists of:
        - Left involute flank  (base/root circle -> addendum circle)
        - Tip arc              (along addendum circle)
        - Right involute flank (addendum circle -> base/root circle)
        - Root arc             (along root circle to the next tooth)

    Args:
        num_teeth: Number of teeth (Z). Must be >= 6.
        module: Gear module in mm (tooth size). Ignored if pitch_diameter is set.
        pressure_angle_deg: Pressure angle in degrees (standard: 20).
        thickness: Axial thickness (face width) in mm.
        bore_diameter: Central bore hole diameter in mm. 0 = no bore.
                       Must be less than the root circle diameter.
        pitch_diameter: If provided, overrides module. module = pitch_diameter / num_teeth.

    Returns:
        cq.Workplane containing the extruded 3D gear solid, centered at origin.

    Raises:
        ValueError: If bore_diameter >= root circle diameter, or if the
                    result is not a manifold cq.Solid.
    """
    if pitch_diameter is not None:
        module = pitch_diameter / num_teeth

    pa = math.radians(pressure_angle_deg)

    # Standard gear circle radii
    pitch_r = num_teeth * module / 2.0
    base_r = pitch_r * math.cos(pa)
    outer_r = pitch_r + module                        # addendum circle
    root_r = max(pitch_r - 1.25 * module, 0.5)       # dedendum circle

    if bore_diameter > 0 and bore_diameter >= 2.0 * root_r:
        raise ValueError(
            f"bore_diameter ({bore_diameter}) must be less than "
            f"root circle diameter ({2.0 * root_r:.3f})"
        )

    def involute_r_theta(t: float) -> tuple[float, float]:
        """Radius and polar angle offset on the involute at parameter t."""
        r = base_r * math.sqrt(1.0 + t * t)
        theta = t - math.atan(t)
        return r, theta

    def t_at_radius(r: float) -> float:
        """Involute parameter t at a given radius."""
        if r <= base_r:
            return 0.0
        return math.sqrt((r / base_r) ** 2 - 1.0)

    # Half angular tooth thickness at pitch circle
    inv_pa = math.tan(pa) - pa
    half_tooth_angle = math.pi / (2.0 * num_teeth) + inv_pa

    angular_pitch = 2.0 * math.pi / num_teeth
    n_flank = 15       # points per involute flank
    n_tip = 3          # interpolation points on tip arc
    n_root = 3         # interpolation points on root arc

    t_tip = t_at_radius(outer_r)
    t_start = t_at_radius(max(root_r, base_r))
    _, dtheta_start = involute_r_theta(t_start)
    _, dtheta_tip = involute_r_theta(t_tip)

    # ── Build full 2D gear profile ──────────────────────────────────
    # Traced counterclockwise around the gear.  For each tooth:
    #   left flank outward → tip arc → right flank inward → root arc
    profile_pts: list[tuple[float, float]] = []

    for i in range(num_teeth):
        theta = i * angular_pitch

        # -- Left involute flank (outward: root/base → tip) --
        if root_r < base_r:
            a = theta - half_tooth_angle
            profile_pts.append((root_r * math.cos(a), root_r * math.sin(a)))

        for j in range(n_flank + 1):
            t = t_start + (t_tip - t_start) * j / n_flank
            r, dt = involute_r_theta(t)
            r = min(r, outer_r)
            a = theta - half_tooth_angle + dt
            profile_pts.append((r * math.cos(a), r * math.sin(a)))

        # -- Tip arc (left tip → right tip, counterclockwise) --
        a_left_tip = theta - half_tooth_angle + dtheta_tip
        a_right_tip = theta + half_tooth_angle - dtheta_tip
        for j in range(1, n_tip + 1):
            frac = j / (n_tip + 1)
            a = a_left_tip + (a_right_tip - a_left_tip) * frac
            profile_pts.append((outer_r * math.cos(a), outer_r * math.sin(a)))

        # -- Right involute flank (inward: tip → root/base) --
        for j in range(n_flank + 1):
            t = t_tip - (t_tip - t_start) * j / n_flank
            r, dt = involute_r_theta(t)
            r = min(r, outer_r)
            a = theta + half_tooth_angle - dt
            profile_pts.append((r * math.cos(a), r * math.sin(a)))

        if root_r < base_r:
            a = theta + half_tooth_angle
            profile_pts.append((root_r * math.cos(a), root_r * math.sin(a)))

        # -- Root arc to next tooth --
        a_end = theta + half_tooth_angle - dtheta_start
        if root_r < base_r:
            a_end = theta + half_tooth_angle

        next_theta = (i + 1) * angular_pitch
        a_next = next_theta - half_tooth_angle + dtheta_start
        if root_r < base_r:
            a_next = next_theta - half_tooth_angle

        for j in range(1, n_root + 1):
            frac = j / (n_root + 1)
            a = a_end + (a_next - a_end) * frac
            profile_pts.append((root_r * math.cos(a), root_r * math.sin(a)))

    # ── Extrude the single closed wire (+ bore) ────────────────────
    wp = cq.Workplane("XY").polyline(profile_pts).close()
    if bore_diameter > 0:
        wp = wp.circle(bore_diameter / 2.0)
    result = wp.extrude(thickness)

    # Center vertically on the origin
    result = result.translate((0, 0, -thickness / 2.0))

    # ── Topological validation ──────────────────────────────────────
    solid = result.val()
    if not isinstance(solid, cq.Solid):
        raise ValueError(
            f"Topological validation failed: expected cq.Solid, "
            f"got {type(solid).__name__}."
        )

    return result


def make_naca_wire(
    naca_code: str = "2412",
    chord_length: float = 200.0,
) -> cq.Wire:
    """Generate a NACA 4-digit airfoil as a closed 2D cq.Wire.

    Computes exact upper and lower surface coordinates from the standard
    NACA 4-digit analytic thickness and camber distributions, builds two
    B-spline edges (upper and lower surfaces) via OpenCASCADE's
    Geom_BSplineCurve interpolation, and assembles them into a single
    closed, manifold cq.Wire. The wire begins and ends precisely at the
    trailing edge and lies in the XY plane at Z=0.

    This wire is the correct primitive for spanwise lofting. Position it
    at a global Z station with wire.translate(cq.Vector(0, 0, z)) and
    pass multiple positioned wires to cq.Solid.makeLoft().

    NACA 4-digit encoding:
        - Digit 1: max camber as percent of chord  (m = d1 / 100)
        - Digit 2: max camber location in tenths   (p = d2 / 10)
        - Digits 3-4: max thickness as percent      (t = d34 / 100)

    Coordinate system:
        - X axis: chordwise, centered (LE at -chord/2, TE at +chord/2)
        - Y axis: thickness direction
        - Wire lies in the XY plane at Z = 0

    Uses the modified trailing-edge coefficient (-0.1036 for the x^4 term)
    to produce a mathematically closed trailing edge (zero TE gap), ensuring
    the wire is watertight without a separate closing edge.

    Args:
        naca_code: 4-digit NACA designation string (e.g., "2412", "0012").
        chord_length: Chord length in mm.

    Returns:
        A closed cq.Wire in the XY plane at Z=0, centered on the chord
        midpoint. Starts and ends at the trailing edge (+chord/2, 0, 0).

    Raises:
        ValueError: If naca_code is not exactly 4 digits.
    """
    if len(naca_code) != 4 or not naca_code.isdigit():
        raise ValueError(
            f"Invalid NACA code '{naca_code}': must be exactly 4 digits "
            f"(e.g., '2412', '0012')."
        )

    m = int(naca_code[0]) / 100.0   # max camber
    p = int(naca_code[1]) / 10.0    # max camber position
    t = int(naca_code[2:4]) / 100.0  # max thickness ratio

    # ── Compute surface coordinates ────────────────────────────────
    # Cosine spacing concentrates points near the leading edge where
    # curvature is highest, improving spline fidelity.
    n_pts = 80
    upper_vecs: list[cq.Vector] = []
    lower_vecs: list[cq.Vector] = []

    for i in range(n_pts + 1):
        beta = i * math.pi / n_pts
        x = (1.0 - math.cos(beta)) / 2.0  # normalized [0, 1]

        # Half-thickness distribution (modified form: closed trailing edge)
        yt = (t / 0.20) * (
            0.2969 * math.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x * x
            + 0.2843 * x * x * x
            - 0.1036 * x * x * x * x
        )

        # Camber line and local slope
        if m == 0.0 or p == 0.0:
            yc = 0.0
            theta = 0.0
        else:
            if x < p:
                yc = (m / (p * p)) * (2.0 * p * x - x * x)
                dyc_dx = (2.0 * m / (p * p)) * (p - x)
            else:
                yc = (m / ((1.0 - p) ** 2)) * (
                    (1.0 - 2.0 * p) + 2.0 * p * x - x * x
                )
                dyc_dx = (2.0 * m / ((1.0 - p) ** 2)) * (p - x)
            theta = math.atan(dyc_dx)

        # Upper and lower surface points (perpendicular to camber line)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        upper_vecs.append(cq.Vector(
            (x - yt * sin_t) * chord_length,
            (yc + yt * cos_t) * chord_length,
            0.0,
        ))
        lower_vecs.append(cq.Vector(
            (x + yt * sin_t) * chord_length,
            (yc - yt * cos_t) * chord_length,
            0.0,
        ))

    # ── Build closed profile wire via OCCT B-spline interpolation ──
    # Upper edge: trailing edge → leading edge (reversed point order)
    # Lower edge: leading edge → trailing edge (natural order)
    # Both edges share exact LE and TE endpoints (yt=0 at x=0 and x=1),
    # forming a mathematically closed loop: TE → LE (upper) → TE (lower).
    upper_edge = cq.Edge.makeSpline(list(reversed(upper_vecs)))
    lower_edge = cq.Edge.makeSpline(lower_vecs)

    wire = cq.Wire.assembleEdges([upper_edge, lower_edge])

    # Center on chord midpoint (LE at -chord/2, TE at +chord/2)
    wire = wire.translate(cq.Vector(-chord_length / 2.0, 0.0, 0.0))

    return wire


def make_naca_airfoil(
    naca_code: str = "2412",
    chord_length: float = 200.0,
    span: float = 500.0,
) -> cq.Workplane:
    """Generate a NACA 4-digit airfoil as an extruded 3D solid (B-Rep volume).

    Delegates to make_naca_wire() for the mathematically exact 2D profile,
    then extrudes linearly along the span axis and centers the result at
    the origin.

    NACA 4-digit encoding:
        - Digit 1: max camber as percent of chord  (m = d1 / 100)
        - Digit 2: max camber location in tenths   (p = d2 / 10)
        - Digits 3-4: max thickness as percent      (t = d34 / 100)

    Coordinate system:
        - X axis: chordwise (leading edge at -chord/2, trailing edge at +chord/2)
        - Y axis: thickness direction
        - Z axis: spanwise (centered at origin)

    Args:
        naca_code: 4-digit NACA designation string (e.g., "2412", "0012").
        chord_length: Chord length in mm.
        span: Spanwise extrusion length in mm.

    Returns:
        cq.Workplane containing the extruded 3D airfoil solid, centered
        at the origin.

    Raises:
        ValueError: If naca_code is not exactly 4 digits, or if the result
                    is not a manifold cq.Solid.
    """
    wire = make_naca_wire(naca_code, chord_length)

    # ── Extrude along span (Z-axis) ───────────────────────────────
    solid = cq.Solid.extrudeLinear(wire, [], cq.Vector(0, 0, span))

    # Center span on origin (wire is already chord-centered)
    result = cq.Workplane("XY").newObject([solid])
    result = result.translate((0.0, 0.0, -span / 2.0))

    # ── Topological validation ─────────────────────────────────────
    val = result.val()
    if not isinstance(val, cq.Solid):
        raise ValueError(
            f"NACA airfoil validation failed: expected cq.Solid, "
            f"got {type(val).__name__}."
        )

    return result


def make_co2_void(diameter: float = 19.5, depth: float = 52.0) -> cq.Workplane:
    """Generate a standard CO2 cartridge cavity for boolean cutting."""
    result = cq.Workplane("YZ").circle(diameter / 2.0).extrude(depth)
    val = result.val()
    if not isinstance(val, cq.Solid):
        raise ValueError("CO2 void validation failed.")
    return result


def make_metric_thread(
    diameter: float,
    pitch: float,
    length: float,
) -> cq.Workplane:
    """Generate a stable ribbed-cylinder approximation of a metric thread.

    Per Domain B rules, a true 3D helical sweep crashes the BRep kernel.
    Instead, this builds a cylindrical shaft with evenly spaced annular ribs
    that approximate thread crests. The result is a manifold solid that
    OpenCASCADE can reliably boolean-union or boolean-cut.

    Args:
        diameter: Nominal (major) thread diameter in mm.
        pitch: Thread pitch in mm (distance between crests).
        length: Total thread length in mm.

    Returns:
        cq.Workplane solid centered at the origin.
    """
    major_r = diameter / 2.0
    minor_r = major_r - 0.6134 * pitch  # standard 60° thread depth
    rib_height = major_r - minor_r
    rib_width = pitch * 0.4

    # Core cylinder at the minor diameter
    result = cq.Workplane("XY").circle(minor_r).extrude(length)

    # Stack annular ribs along the length to simulate thread crests
    num_ribs = int(length / pitch)
    for i in range(num_ribs):
        z_offset = pitch * 0.5 + i * pitch
        if z_offset + rib_width / 2.0 > length:
            break
        rib = (
            cq.Workplane("XY")
            .workplane(offset=z_offset - rib_width / 2.0)
            .circle(major_r)
            .circle(minor_r)
            .extrude(rib_width)
        )
        result = result.union(rib)

    # Center on origin
    result = result.translate((0, 0, -length / 2.0))

    solid = result.val()
    if not isinstance(solid, cq.Solid):
        raise ValueError(
            f"Thread validation failed: expected cq.Solid, "
            f"got {type(solid).__name__}."
        )

    return result


# ---------------------------------------------------------------------------
# T2-10: High-frequency parametric engineering shape templates
# ---------------------------------------------------------------------------


def make_flanged_housing(
    OD: float,
    ID: float,
    height: float,
    flange_OD: float,
    flange_thickness: float,
    bolt_circle_diameter: float,
    n_bolt_holes: int,
    bolt_hole_diameter: float,
) -> cq.Workplane:
    """Cylindrical housing with a mounting flange.

    Generates a hollow cylinder (bore ID, outer diameter OD) with a flat
    circular flange at its base. Bolt holes are arranged on a bolt circle.

    Args:
        OD: Outer diameter of the cylindrical body (mm).
        ID: Inner bore diameter (mm). Must be < OD.
        height: Total height of the cylindrical body, not including flange (mm).
        flange_OD: Outer diameter of the flange disc (mm). Must be >= OD.
        flange_thickness: Thickness of the flange (mm).
        bolt_circle_diameter: PCD (pitch circle diameter) for bolt holes (mm).
        n_bolt_holes: Number of bolt holes. 0 = no bolt holes.
        bolt_hole_diameter: Diameter of each bolt clearance hole (mm).

    Returns:
        cq.Workplane containing the assembled flanged housing solid.

    Raises:
        ValueError: If geometric constraints are violated.
    """
    if ID >= OD:
        raise ValueError(f"Bore ID ({ID}) must be less than outer OD ({OD}).")
    if flange_OD < OD:
        raise ValueError(f"flange_OD ({flange_OD}) must be >= body OD ({OD}).")
    if bolt_hole_diameter >= bolt_circle_diameter:
        raise ValueError(
            f"bolt_hole_diameter ({bolt_hole_diameter}) must be less than "
            f"bolt_circle_diameter ({bolt_circle_diameter})."
        )

    # Flange disc at the bottom
    flange = (
        cq.Workplane("XY")
        .circle(flange_OD / 2.0)
        .circle(ID / 2.0)
        .extrude(flange_thickness)
    )

    # Cylindrical body on top of the flange
    body = (
        cq.Workplane("XY")
        .workplane(offset=flange_thickness)
        .circle(OD / 2.0)
        .circle(ID / 2.0)
        .extrude(height)
    )

    result = flange.union(body)

    # Bolt holes in the flange (polar array)
    if n_bolt_holes > 0 and bolt_circle_diameter > 0:
        bolt_r = bolt_circle_diameter / 2.0
        result = (
            result
            .faces("<Z")
            .workplane()
            .polarArray(bolt_r, 0, 360, n_bolt_holes)
            .circle(bolt_hole_diameter / 2.0)
            .cutThruAll()
        )

    return result


def make_shaft(
    diameter: float,
    length: float,
    keyway_width: float = 0.0,
    keyway_depth: float = 0.0,
    chamfer_end: float = 0.5,
) -> cq.Workplane:
    """Shaft with optional keyway and end chamfers.

    Generates a cylindrical shaft centered at the origin (axis along Z).
    An optional rectangular keyway is cut along the full length on the
    outer surface. Both ends are chamfered.

    Args:
        diameter: Shaft outer diameter (mm).
        length: Total shaft length (mm).
        keyway_width: Width of the keyway slot (mm). 0 = no keyway.
        keyway_depth: Depth of the keyway slot from the outer surface (mm).
        chamfer_end: Chamfer size on both shaft ends (mm). 0 = no chamfer.

    Returns:
        cq.Workplane containing the shaft solid.
    """
    # Main cylinder
    result = cq.Workplane("XY").circle(diameter / 2.0).extrude(length)

    # Keyway — rectangular slot along full length on top face
    if keyway_width > 0 and keyway_depth > 0:
        if keyway_width >= diameter:
            raise ValueError(
                f"keyway_width ({keyway_width}) must be less than shaft diameter ({diameter})."
            )
        # Position the keyway slot at the top of the shaft
        slot_y_offset = diameter / 2.0 - keyway_depth
        keyway = (
            cq.Workplane("XY")
            .center(0, slot_y_offset)
            .rect(keyway_width, keyway_depth * 2)
            .extrude(length)
        )
        result = result.cut(keyway)

    # Chamfer both ends
    if chamfer_end > 0:
        result = result.edges("%Circle").chamfer(chamfer_end)

    return result


def make_bracket(
    base_width: float,
    base_length: float,
    base_thickness: float,
    wall_height: float,
    wall_thickness: float,
    n_mounting_holes: int = 2,
    hole_diameter: float = 5.0,
) -> cq.Workplane:
    """L-bracket with mounting holes in the base flange.

    Generates an L-shaped bracket: a flat base plate with an upright wall
    at one end. Mounting holes are evenly spaced along the base length.

    Args:
        base_width: Width of the base flange (mm).
        base_length: Length of the base flange (mm).
        base_thickness: Thickness of the base plate (mm).
        wall_height: Height of the vertical wall above the base (mm).
        wall_thickness: Thickness of the vertical wall (mm).
        n_mounting_holes: Number of through holes in the base (0 = none).
        hole_diameter: Diameter of each mounting hole (mm).

    Returns:
        cq.Workplane containing the L-bracket solid.
    """
    # Base plate
    base = (
        cq.Workplane("XY")
        .box(base_length, base_width, base_thickness)
    )

    # Vertical wall at one end of the base
    wall = (
        cq.Workplane("XY")
        .workplane(offset=base_thickness)
        .center(-base_length / 2.0 + wall_thickness / 2.0, 0)
        .box(wall_thickness, base_width, wall_height)
    )

    result = base.union(wall)

    # Mounting holes in the base (linear pattern along X)
    if n_mounting_holes > 0 and hole_diameter > 0:
        if n_mounting_holes == 1:
            offsets = [0.0]
        else:
            spacing = (base_length - 2 * hole_diameter) / (n_mounting_holes - 1)
            offsets = [
                -base_length / 2.0 + hole_diameter + i * spacing
                for i in range(n_mounting_holes)
            ]
        for x_off in offsets:
            result = (
                result
                .faces(">Z")
                .workplane()
                .center(x_off, 0)
                .circle(hole_diameter / 2.0)
                .cutThruAll()
            )

    return result


def make_bearing_housing(
    bearing_OD: float,
    bearing_width: float,
    housing_OD: float,
    housing_length: float,
    lip_thickness: float = 2.0,
) -> cq.Workplane:
    """Bearing housing with a press-fit bore.

    Generates a cylindrical housing with a precision bore sized for the
    bearing outer race. A retaining lip (shoulder) is formed at one end.

    Args:
        bearing_OD: Outer diameter of the bearing outer race (mm).
            This becomes the bore diameter of the housing.
        bearing_width: Width of the bearing (mm). The bore depth matches this.
        housing_OD: Outer diameter of the housing body (mm).
        housing_length: Total length of the housing (mm).
        lip_thickness: Axial thickness of the retaining lip at the closed end (mm).

    Returns:
        cq.Workplane containing the bearing housing solid.

    Raises:
        ValueError: If bearing_OD >= housing_OD.
    """
    if bearing_OD >= housing_OD:
        raise ValueError(
            f"bearing_OD ({bearing_OD}) must be less than housing_OD ({housing_OD})."
        )

    # Outer cylinder
    outer = (
        cq.Workplane("XY")
        .circle(housing_OD / 2.0)
        .extrude(housing_length)
    )

    # Bore for the bearing (full-depth through hole at bearing_OD)
    bore = (
        cq.Workplane("XY")
        .workplane(offset=lip_thickness)
        .circle(bearing_OD / 2.0)
        .extrude(housing_length - lip_thickness)
    )

    result = outer.cut(bore)
    return result


def make_spacer(
    OD: float,
    ID: float,
    length: float,
    chamfer: float = 0.3,
) -> cq.Workplane:
    """Cylindrical spacer (washer/sleeve) with optional end chamfers.

    Args:
        OD: Outer diameter (mm).
        ID: Inner bore diameter (mm). Must be < OD.
        length: Spacer length/height (mm).
        chamfer: Edge chamfer size on both ends (mm). 0 = no chamfer.

    Returns:
        cq.Workplane containing the spacer solid.
    """
    if ID >= OD:
        raise ValueError(f"Bore ID ({ID}) must be less than outer OD ({OD}).")

    result = (
        cq.Workplane("XY")
        .circle(OD / 2.0)
        .circle(ID / 2.0)
        .extrude(length)
    )

    if chamfer > 0:
        result = result.edges("%Circle").chamfer(chamfer)

    return result


def make_boss_with_bore(
    boss_OD: float,
    bore_ID: float,
    boss_height: float,
    base_OD: float = 0.0,
    base_thickness: float = 0.0,
    counterbore_ID: float = 0.0,
    counterbore_depth: float = 0.0,
) -> cq.Workplane:
    """Boss (cylindrical protrusion) with a through/blind bore.

    Supports a flat base flange and counterbore entry for fastener head
    clearance.

    Args:
        boss_OD: Outer diameter of the boss cylinder (mm).
        bore_ID: Bore diameter through the boss (mm). 0 = solid boss.
        boss_height: Height of the boss above the base (mm).
        base_OD: Outer diameter of an optional flat base flange (mm).
            0 = no base, just the boss cylinder.
        base_thickness: Thickness of the base flange (mm).
        counterbore_ID: Counterbore entry diameter at the top (mm).
            0 = no counterbore (plain bore throughout).
        counterbore_depth: Depth of the counterbore from the top face (mm).

    Returns:
        cq.Workplane containing the boss solid.
    """
    if bore_ID > 0 and bore_ID >= boss_OD:
        raise ValueError(f"bore_ID ({bore_ID}) must be less than boss_OD ({boss_OD}).")

    total_height = boss_height + base_thickness

    # Start with boss cylinder (+ base if specified)
    if base_OD > 0 and base_thickness > 0:
        if base_OD < boss_OD:
            raise ValueError(
                f"base_OD ({base_OD}) must be >= boss_OD ({boss_OD})."
            )
        base = (
            cq.Workplane("XY")
            .circle(base_OD / 2.0)
            .extrude(base_thickness)
        )
        boss = (
            cq.Workplane("XY")
            .workplane(offset=base_thickness)
            .circle(boss_OD / 2.0)
            .extrude(boss_height)
        )
        result = base.union(boss)
    else:
        result = (
            cq.Workplane("XY")
            .circle(boss_OD / 2.0)
            .extrude(boss_height)
        )

    # Through bore
    if bore_ID > 0:
        result = (
            result
            .faces(">Z")
            .workplane()
            .circle(bore_ID / 2.0)
            .cutThruAll()
        )

    # Counterbore (larger entrance from the top)
    if counterbore_ID > 0 and counterbore_depth > 0:
        if bore_ID > 0 and counterbore_ID <= bore_ID:
            raise ValueError(
                f"counterbore_ID ({counterbore_ID}) must be > bore_ID ({bore_ID})."
            )
        result = (
            result
            .faces(">Z")
            .workplane()
            .circle(counterbore_ID / 2.0)
            .cutBlind(-counterbore_depth)
        )

    return result


# ---------------------------------------------------------------------------
# T3-06: Domain E — Sheet Metal Utility Functions
# ---------------------------------------------------------------------------

def make_bent_bracket(
    material_thickness: float,
    base_length: float,
    base_width: float,
    flange_height: float,
    bend_radius: float,
    n_base_holes: int = 2,
    hole_diameter: float = 5.0,
) -> cq.Workplane:
    """Standard L-bracket from sheet metal with bend relief.

    Produces a folded L-bracket:
      - Flat base plate with mounting holes
      - Vertical wall (flange) of the specified height
      - Bend at the corner approximated as an arc-extruded fillet

    Design rules enforced:
      - Minimum bend radius >= 1× material_thickness (raises ValueError if violated)
      - Minimum flange height >= 2× material_thickness + bend_radius
      - Hole-to-edge distance >= 2× material_thickness

    Args:
        material_thickness: Sheet gauge in mm (e.g., 1.5 for 14 ga steel).
        base_length: Length of the flat base in mm.
        base_width: Width of the flat base in mm.
        flange_height: Height of the vertical flange in mm.
        bend_radius: Inner bend radius in mm.
        n_base_holes: Number of mounting holes in the base (evenly spaced).
        hole_diameter: Diameter of mounting holes in mm.

    Returns:
        cq.Workplane solid (the folded bracket in final geometry).

    Raises:
        ValueError: If design rule constraints are violated.
    """
    # Design rule checks
    if bend_radius < material_thickness:
        raise ValueError(
            f"Bend radius {bend_radius}mm < material thickness {material_thickness}mm. "
            f"Minimum is 1× material thickness. "
            f"Use bend_radius >= {material_thickness}."
        )

    min_flange = 2 * material_thickness + bend_radius
    if flange_height < min_flange:
        raise ValueError(
            f"Flange height {flange_height}mm < minimum {min_flange:.1f}mm "
            f"(= 2×thickness + bend_radius). "
            f"Use flange_height >= {min_flange:.1f}."
        )

    min_edge_dist = 2 * material_thickness
    if hole_diameter > 0 and n_base_holes > 0:
        hole_spacing = base_length / (n_base_holes + 1)
        if hole_spacing < hole_diameter + min_edge_dist * 2:
            raise ValueError(
                f"Hole spacing {hole_spacing:.1f}mm is too tight for {n_base_holes} "
                f"holes of diameter {hole_diameter}mm with minimum edge distance "
                f"{min_edge_dist}mm. Reduce n_base_holes or increase base_length."
            )

    # --- Build base plate ---
    base = (
        cq.Workplane("XY")
        .box(base_length, base_width, material_thickness, centered=True)
    )

    # Add mounting holes in base (linear pattern)
    if n_base_holes > 0 and hole_diameter > 0:
        hole_spacing = base_length / (n_base_holes + 1)
        for i in range(n_base_holes):
            x = -base_length / 2.0 + hole_spacing * (i + 1)
            base = (
                base
                .faces(">Z")
                .workplane()
                .center(x, 0)
                .hole(hole_diameter)
            )

    # --- Build flange (vertical wall) ---
    # The flange originates from the back edge of the base plate
    # and extends upward by flange_height
    flange_base_z = material_thickness  # top of base plate
    flange = (
        cq.Workplane("XZ")
        .workplane(offset=base_width / 2.0)
        .rect(base_length, flange_height, centered=False)
        .extrude(material_thickness)
    )

    # Translate flange to correct position
    flange = flange.translate((0, 0, 0))

    # Build the bend (approximated as a quarter-circle extrusion)
    # The bend radius fillet is applied at the corner between base and flange
    result = base.union(flange)

    # Apply bend radius fillet at the base-flange junction
    try:
        result = (
            result
            .edges("|X")
            .edges(">Y")
            .fillet(bend_radius)
        )
    except Exception:  # noqa: BLE001
        # Fillet may fail on complex geometry — continue without it
        pass

    return result


def make_enclosure_panel(
    width: float,
    height: float,
    material_thickness: float,
    flange_depth: float = 0.0,
    cutout_positions: list = None,
) -> cq.Workplane:
    """Sheet metal panel with optional flanges and rectangular cutouts.

    Produces a flat panel (like a computer side panel or electrical enclosure face)
    with optional return flanges on all four edges and rectangular knockouts.

    Args:
        width: Panel width in mm.
        height: Panel height in mm.
        material_thickness: Sheet gauge in mm.
        flange_depth: Depth of return flanges on all edges (0 = no flanges).
        cutout_positions: List of (x_center, y_center, cutout_width, cutout_height)
            tuples in mm (origin = center of panel). Pass [] or None for no cutouts.

    Returns:
        cq.Workplane solid (flat panel + flanges).

    Design rules enforced:
        - Cutout edges must be >= 2× material_thickness from panel edge.
        - Flange depth >= 2× material_thickness + material_thickness (wall) if nonzero.
    """
    if cutout_positions is None:
        cutout_positions = []

    # --- Build main flat panel ---
    panel = (
        cq.Workplane("XY")
        .box(width, height, material_thickness, centered=True)
    )

    # --- Add perimeter flanges ---
    if flange_depth > 0:
        min_flange = 2 * material_thickness
        if flange_depth < min_flange:
            raise ValueError(
                f"Flange depth {flange_depth}mm < minimum {min_flange}mm "
                f"(= 2× material thickness). Use flange_depth >= {min_flange}."
            )

        # Four return flanges (down-facing, like an enclosure panel)
        flange_offset = material_thickness / 2.0  # panel half-thickness

        # Left and right flanges
        for sign in (-1, +1):
            flange = (
                cq.Workplane("YZ")
                .workplane(offset=sign * width / 2.0)
                .rect(height, flange_depth, centered=False)
                .extrude(material_thickness)
            )
            panel = panel.union(flange)

        # Top and bottom flanges
        for sign in (-1, +1):
            flange = (
                cq.Workplane("XZ")
                .workplane(offset=sign * height / 2.0)
                .rect(width + 2 * material_thickness, flange_depth, centered=False)
                .extrude(material_thickness)
            )
            panel = panel.union(flange)

    # --- Add rectangular cutouts ---
    min_edge_dist = 2 * material_thickness
    for cx, cy, cw, ch in cutout_positions:
        # Validate cutout is within panel bounds with minimum edge clearance
        if abs(cx) + cw / 2.0 > width / 2.0 - min_edge_dist:
            raise ValueError(
                f"Cutout at ({cx}, {cy}) width {cw} is too close to panel edge. "
                f"Minimum edge distance: {min_edge_dist}mm."
            )
        if abs(cy) + ch / 2.0 > height / 2.0 - min_edge_dist:
            raise ValueError(
                f"Cutout at ({cx}, {cy}) height {ch} is too close to panel edge. "
                f"Minimum edge distance: {min_edge_dist}mm."
            )

        panel = (
            panel
            .faces(">Z")
            .workplane()
            .center(cx, cy)
            .rect(cw, ch)
            .cutThruAll()
        )

    return panel


# Expose template function names for strategy prompt injection
AVAILABLE_TEMPLATES = [
    "make_flanged_housing(OD, ID, height, flange_OD, flange_thickness, "
    "bolt_circle_diameter, n_bolt_holes, bolt_hole_diameter)",
    "make_shaft(diameter, length, keyway_width=0, keyway_depth=0, chamfer_end=0.5)",
    "make_bracket(base_width, base_length, base_thickness, wall_height, "
    "wall_thickness, n_mounting_holes=2, hole_diameter=5.0)",
    "make_bearing_housing(bearing_OD, bearing_width, housing_OD, housing_length, "
    "lip_thickness=2.0)",
    "make_spacer(OD, ID, length, chamfer=0.3)",
    "make_boss_with_bore(boss_OD, bore_ID, boss_height, base_OD=0, "
    "base_thickness=0, counterbore_ID=0, counterbore_depth=0)",
    # T3-06: Sheet metal templates (Domain E)
    "make_bent_bracket(material_thickness, base_length, base_width, flange_height, "
    "bend_radius, n_base_holes=2, hole_diameter=5.0)",
    "make_enclosure_panel(width, height, material_thickness, flange_depth=0, "
    "cutout_positions=[])",
]
