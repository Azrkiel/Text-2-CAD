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
