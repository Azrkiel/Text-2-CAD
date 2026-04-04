"""
NLP Classification Tier for the Text-to-CAD Strategy Pattern.

Classifies user part descriptions into geometric domains so the Machinist
can apply the correct CadQuery generation strategy. This is a fast, cheap
Gemini call that runs before the Machinist to prevent blind primitive-only
generation.

Domains:
  A (Primitives/Structural) — boxes, plates, brackets, enclosures
  B (Mechanical/Parametric) — gears, bearings, threads, cams, springs
  C (Organic/Ergonomic)     — handles, shells, fairings, grips, spoons
"""

import google.generativeai as genai

MODEL = "gemini-2.5-flash"

# Gemini-compatible schema (no unsupported fields)
_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {
            "type": "string",
            "enum": ["A", "B", "C"],
            "description": (
                "The geometric domain: "
                "'A' for primitives/structural (boxes, plates, enclosures, brackets, frames). "
                "'B' for mechanical/parametric (gears, bearings, threads, cams, pulleys, springs, fasteners). "
                "'C' for organic/ergonomic (handles, mouse shells, fairings, grips, bottles, spoons, helmets)."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "One sentence explaining why this domain was chosen.",
        },
        "key_params": {
            "type": "string",
            "description": (
                "Extracted engineering parameters as a comma-separated string. "
                "For Domain A: 'width=X, height=Y, depth=Z' (bounding box). "
                "For Domain B: 'num_teeth=X, module=Y, bore=Z' or similar parametric values. "
                "For Domain C: 'length=X, max_width=Y, cross_sections=N' or similar. "
                "Extract whatever numeric values are present in the description."
            ),
        },
    },
    "required": ["domain", "reasoning", "key_params"],
}


async def classify_part(description: str) -> dict:
    """Classify a part description into a geometric domain.

    Args:
        description: The part's physical description from PartDefinition.

    Returns:
        {"domain": "A"|"B"|"C", "reasoning": str, "key_params": str}
    """
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=(
            "You are a mechanical engineering classifier. Given a part description, "
            "determine which geometric domain it belongs to. "
            "\n"
            "Domain A (Primitives/Structural): Parts best made with extrusions and "
            "boolean operations on basic shapes. Examples: base plates, enclosures, "
            "L-brackets, mounting blocks, simple frames, spacers, standoffs. "
            "Key signal: described mainly by length/width/height, holes, slots, fillets. "
            "\n"
            "Domain B (Mechanical/Parametric): Parts requiring precise mathematical "
            "curves or parametric formulas. Examples: spur gears, helical gears, "
            "bearings, cams, pulleys, lead screws, threads, sprockets, turbine blades. "
            "Key signal: mentions teeth, pitch, module, pressure angle, involute, "
            "thread pitch, helix, or similar engineering parameters. "
            "\n"
            "Domain C (Organic/Ergonomic): Parts with smooth, flowing surfaces that "
            "cannot be made with simple extrusions. Examples: ergonomic handles, "
            "mouse shells, aerodynamic fairings, bottle shapes, helmet visors, "
            "phone cases with contoured backs, joystick grips. "
            "Key signal: described by curves, comfort, aerodynamics, organic shapes, "
            "or cross-sectional profiles that vary along an axis. "
            "\n"
            "When in doubt between A and B, choose B (parametric is safer than primitive). "
            "When in doubt between A and C, choose A (structural is more reliable). "
            "Extract all numeric parameters you can find from the description."
        ),
    )

    response = await model.generate_content_async(
        f"Classify this part:\n{description}",
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": _CLASSIFICATION_SCHEMA,
        },
    )

    import json
    return json.loads(response.text)
