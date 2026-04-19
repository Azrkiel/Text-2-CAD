# Aerospace Beachhead Demo Assemblies

Five demonstration assemblies for Mirum's aerospace/defense beachhead.
Each entry contains the exact prompt, expected part decomposition, mate types,
and generation notes. Run each prompt through the `/generate` endpoint.

---

## Demo 1: Wing Structural Assembly

**Prompt:**
```
Generate a wing structural assembly with 5 equally spaced ribs and 2 spars.
NACA 2412 profile, root chord 200mm, tip chord 120mm, span 600mm.
Ribs are 2mm thick. Front spar at 25% chord, rear spar at 60% chord, 
both 2mm thick. Ribs fastened to spars at their intersections.
```

**Expected decomposition:**
- Single PartDefinition `wing_segment_assembly` (Domain D — coupled subsystem)
- Machinist generates oml_solid + 5 ribs + 2 spars in one script
- Self-referencing mating rule (Domain D exception)

**Key validation points:**
- `make_naca_wire(naca_code='2412', chord_length=200)` at root
- Tip wire translated `cq.Vector(0, 600, 0)`
- Ribs intersect oml_solid (NOT hollow/shell approach)
- Assembly exported to both `.glb` and `.step`

---

## Demo 2: Landing Gear Mechanism

**Prompt:**
```
Design a simplified landing gear mechanism with 3 components:
1. An oleo strut: a hollow cylindrical outer housing 80mm OD, 70mm ID, 200mm tall
2. An inner piston: a solid cylinder 68mm diameter, 180mm long, slides inside the housing
3. A wheel hub: a disc 120mm diameter, 30mm thick, with a 20mm central bore

The piston slides axially inside the housing (cylindrical joint, 0-150mm travel).
The wheel hub rotates freely on the end of the piston (revolute joint).
```

**Expected decomposition:**
- `oleo_housing`: cylindrical housing, Domain A
- `inner_piston`: cylinder, Domain A
- `wheel_hub`: disc with bore, Domain A
- Mates: housing↔piston CYLINDRICAL (dof_min=0, dof_max=150, dof_unit='mm')
- Mates: piston↔wheel_hub REVOLUTE

**Demonstrates:** T2-02 DoF range annotation, T1-01 constraint assembly

---

## Demo 3: Engine Nacelle

**Prompt:**
```
Generate an engine nacelle assembly consisting of:
1. Outer cowling: organic teardrop fairing, 300mm long, max diameter 180mm at midpoint,
   tapers to 80mm at inlet and 100mm at exit
2. Inner duct: concentric cylinder 140mm OD inside the cowling, 300mm long, 5mm wall
3. Forward frame: structural ring 180mm OD, 140mm ID, 15mm thick, 8 equally spaced 
   M6 bolt holes on a 160mm bolt circle
4. Aft frame: same ring dimensions as forward frame

Forward frame fastened to front face of outer cowling and inner duct.
Aft frame fastened to rear face.
```

**Expected decomposition:**
- `outer_cowling`: Domain C (organic fairing)
- `inner_duct`: Domain A (cylinder)  
- `forward_frame`: Domain A (flanged ring with bolt circle)
- `aft_frame`: Domain A (same as forward)
- Mates: FASTENED between frames and cowling/duct end faces

**Demonstrates:** Mixed-domain assembly (C + A), T2-10 flanged ring template

---

## Demo 4: Avionics Bay Bracket Assembly

**Prompt:**
```
Generate an avionics bay bracket assembly with:
1. Three L-shaped mounting brackets: each 80mm base x 60mm wall x 4mm thick,
   two M4 holes in the base on 60mm spacing
2. An equipment tray: flat plate 200mm x 150mm x 3mm, 
   with slots to match the bracket bolt pattern

Brackets are fastened to the tray at their base holes.
```

**Expected decomposition:**
- `bracket_1`, `bracket_2`, `bracket_3`: Domain A (L-bracket — uses `make_bracket()`)
- `equipment_tray`: Domain A (flat plate with slots)
- 3 FASTENED mates: each bracket↔tray

**Demonstrates:** T2-10 `make_bracket()` template, parallel Machinist execution

---

## Demo 5: Hydraulic Manifold

**Prompt:**
```
Generate a hydraulic manifold assembly:
1. Main manifold body: steel rectangular block 100mm x 80mm x 60mm with:
   - 4 intersecting 12mm diameter flow channels (2 horizontal, 2 vertical)
   - M12 threaded ports at each channel end (6 ports total)
2. Two end caps: circular discs 80mm diameter, 8mm thick, M12 boss
3. Four port adapters: cylindrical fittings 30mm OD, 20mm long, M12 threads

End caps fastened to the two exposed lateral faces of the manifold.
Port adapters fastened at the four remaining ports.
```

**Expected decomposition:**
- `manifold_body`: Domain B (complex bore intersections)
- `end_cap_left`, `end_cap_right`: Domain A
- `adapter_port_1..4`: Domain A
- Mates: all FASTENED

**Demonstrates:** 7-part assembly, Domain B bore geometry, maximum part count test

---

## Running the Demos

```bash
# Start the stack
docker-compose up -d

# Run a demo (replace with actual prompt)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "<paste prompt here>"}' \
  -N --no-buffer

# Or use the Streamlit UI at http://localhost:8501
```

## Export to SOLIDWORKS

After generation, export to Parasolid for SOLIDWORKS import:

```bash
# Requires CAD Exchanger license (T2-09)
curl -X POST http://localhost:8000/export/parasolid \
  -H "Content-Type: application/json" \
  -d '{"script": "<CadQuery script from generate response>"}' \
  | jq -r '.data_b64' | base64 -d > assembly.x_t
```

## Validation Checklist

For each demo, verify:
- [ ] Script executes without error (Critic Loop passes)
- [ ] GLB renders correctly in 3D viewer
- [ ] STEP file imports into FreeCAD or SOLIDWORKS
- [ ] Constraint solver reports SOLVER_SUCCESS
- [ ] No interpenetration detected (T1-02)
- [ ] LVM score ≥ 0.7 (T1-07)
- [ ] Parasolid export round-trips cleanly (T2-09)
