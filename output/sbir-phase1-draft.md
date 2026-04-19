# SBIR Phase I Application Draft — Mirum Text-to-CAD

**Program:** DoD SBIR Phase I  
**Topic Area:** AI-Assisted Engineering Design / Digital Engineering / Model-Based Engineering (MBE)  
**Relevant Offices:** Air Force AFWERX, DARPA I2O, Navy NAVAIR  
**Phase I Award:** ~$250,000 / 6 months

---

## Cover Page

**Company:** [Company Name]  
**PI:** [Principal Investigator]  
**Technical Abstract:** Mirum is an assembly-first text-to-CAD generation system that converts natural-language design briefs into multi-part 3D assemblies with typed kinematic constraints, targeting aerospace/defense structural and mechanism design automation.

---

## 1. Technical Abstract (250 words)

We propose Mirum, a novel AI system that generates multi-part mechanical assemblies from natural-language design specifications. Unlike prior text-to-CAD systems that generate isolated single parts, Mirum produces complete assemblies with topologically correct constraint relationships (FASTENED, REVOLUTE, CYLINDRICAL, BALL joints) verified by the OCCT geometric constraint solver.

The system implements a hierarchical agentic architecture: a Requirements Engineering Agent (REA) extracts structured design specifications; a Planner decomposes the assembly into parts with typed kinematic mates; concurrent Machinist agents generate parametric CadQuery/OCCT Python code for each part; a Critic Loop validates and self-corrects each part through up to three iterations; and a Deterministic Assembler applies constraint solving to position all parts without LLM involvement.

Domain D (Aerospace/Aerodynamics) receives specialized generation strategies implementing the OML Master Modeling workflow: NACA airfoil wire generation, reversed loft ordering for correct B-rep normals, and rib/spar geometry via intersection with the solid outer mold line. This approach eliminates the Boolean subtraction failures that plague naive approaches to aerostructural CAD generation.

The Phase I objective is to demonstrate Mirum on five aerospace assembly types: wing structural assemblies, landing gear mechanisms, engine nacelles, avionics bay bracket assemblies, and hydraulic manifolds. Success is measured by STEP export fidelity (importable into SOLIDWORKS/Inventor without geometry errors), kinematic constraint satisfaction rate, and LVM semantic accuracy score.

---

## 2. Problem Statement

**The bottleneck:** CAD model creation is the single largest time sink in the engineering design process. For aerospace structures, a wing section with 5 ribs and 2 spars — a routine subassembly — requires 4–8 hours of skilled CAD time. The same assembly has a fixed geometric structure that can be specified in two sentences.

**Why existing tools fail:**
- Natural language → CAD tools (Zoo.dev, Autodesk AI) generate isolated single parts, not assemblies. Assemblies require manually placing and constraining parts.
- Script-based generation (CadQuery, OpenSCAD) requires programming expertise that most design engineers lack.
- FEA-integrated tools (ANSYS Discovery) require geometry to already exist — they accelerate analysis, not creation.
- LLMs (GPT-4, Gemini) can generate CAD code for simple parts but fail on assemblies because they cannot reliably perform 3D spatial arithmetic required for absolute coordinate placement.

**The DoD relevance:**
Digital Engineering (DoD Directive 5000.97) mandates Model-Based Engineering across all major acquisition programs. Every DoD program office maintains teams of CAD engineers manually creating and updating system models. Mirum directly reduces this bottleneck by generating draft CAD assemblies from requirements documents, enabling faster iteration in conceptual and preliminary design phases.

**Target use cases (ACAT I programs):**
- Rapid concept generation during pre-milestone A (Materiel Solution Analysis)
- Trade study support: generate 10 housing geometries in 10 minutes for mass/volume comparison
- Digital thread integration: NL specification → parametric CAD → FEA mesh

---

## 3. Technical Objectives

**Phase I Objective:** Demonstrate Mirum on 5 aerospace assembly types with STEP export verified against SOLIDWORKS import.

**Specific Aims:**
1. Implement and validate assembly constraint solving (cq.Constraint + OCCT solver) for all 6 mate types
2. Achieve <5% invalidity rate on aerospace assemblies (vs. current ~15% baseline)
3. Demonstrate 5 aerospace demo assemblies (wing structural, landing gear, nacelle, avionics bracket, hydraulic manifold)
4. Implement Parasolid (.x_t) export via CAD Exchanger SDK for SOLIDWORKS-native round-trips
5. Deploy evaluation harness measuring invalidity rate, constraint satisfaction rate, and LVM semantic accuracy

**Phase II (planned):** Fine-tune assembly Planner on Fusion 360 Gallery (8,251 real assemblies) and synthetic ABC Dataset assembly pairs (100K+ examples). Target <1% invalidity rate competitive with human CAD drafts.

---

## 4. Technical Approach

### 4.1 Agentic Pipeline Architecture

```
User NL Prompt
    │
    ├─► Requirements Engineering Agent (REA) ──► DesignRequirements JSON
    │
    ▼
Planner (Gemini 2.5 Flash, structured output)
    │ AssemblyManifest: parts[] + mating_rules[] with typed mate_type
    │
    ▼ (parallel, semaphore ≤ 3)
Machinist × N (per-part CadQuery generation + Critic Loop)
    │ .step files per part
    ▼
Deterministic Assembler (cq.Constraint + assembly.solve())
    │ No LLM — pure OCCT constraint solving
    ▼
GLB + STEP + Parasolid (.x_t) output
```

### 4.2 Domain D Aerospace Strategy

The Machinist for aerospace parts implements the OML Master Modeling workflow:

1. Generate two NACA wire profiles (root + tip) using `make_naca_wire()`
2. Rotate both wires into the XZ plane (chord=X, thickness=Z, span=Y)
3. Loft in reversed order (tip→root) for correct outward normals
4. Generate ribs and spars by intersecting oversized blanks with oml_solid
5. Package into cq.Assembly for dual-format export

This approach eliminates all Boolean skin subtractions (the primary failure mode for naive approaches) and produces mathematically exact rib cross-sections guaranteed to fit within the aerodynamic envelope.

### 4.3 Assembly Constraint Model

The kinematic mate vocabulary maps to OCCT constraint types:

| Mate Type | OCCT Constraint | DoF |
|-----------|----------------|-----|
| FASTENED | Plane | 0 |
| REVOLUTE | Axis | 1 rot |
| SLIDER | Plane | 1 trans |
| CYLINDRICAL | Axis | 1 rot + 1 trans |
| PLANAR | Plane | 2 trans + 1 rot |
| BALL | Point | 3 rot |
| GEAR | (metadata) | coupled revolute |
| SCREW | (metadata) | coupled rot+trans |

Coupling mates (GEAR, SCREW) do not participate in static constraint solving but are recorded as kinematic metadata for simulation tool integration.

### 4.4 Evaluation Framework

Metrics tracked per generation:
- **Part invalidity rate**: % of Machinist runs failing OCCT compilation
- **Assembly invalidity rate**: % of assemblies where solver fails
- **Mate constraint satisfaction**: % of typed constraints satisfied post-solve
- **LVM Score**: Gemini Vision semantic accuracy vs. original prompt (0–1)
- **SOLIDWORKS round-trip**: Import fidelity of Parasolid export

---

## 5. Innovation

**What has not been done before:**

1. **Assembly-first generation with constraint solving**: Prior work (Text2CAD, ShapeAssembly, CADFusion) targets single parts or uses absolute coordinate placement. Mirum is the first system to use a typed kinematic constraint vocabulary interpreted by an OCCT constraint solver.

2. **Domain-specialized aerospace strategies**: The OML Master Modeling workflow (NACA wire generation → reversed loft → intersection-based ribs/spars) is a novel approach to LLM-generated aerostructural CAD that eliminates the primary failure modes of naive approaches.

3. **Hierarchical agentic verification**: The Requirements Engineering Agent → Planner → Machinist × N → Critic Loop → Deterministic Assembler pipeline separates concerns in a way that allows each stage to be independently validated, fine-tuned, and replaced without breaking the others.

4. **Parasolid export for SOLIDWORKS**: No existing open-source or research text-to-CAD system produces Parasolid output. STEP conversion artifacts are eliminated, enabling reliable round-trips with SOLIDWORKS-based workflows (the dominant CAD tool at defense prime contractors).

---

## 6. Commercial Application

**Primary market:** Defense prime contractors (Boeing Defense, Lockheed Martin, Raytheon, Northrop Grumman, BAE Systems) and Tier 1 aerospace suppliers.

**Buyer profile:** Engineering design teams using SOLIDWORKS or CATIA V5/V6, working on ACAT I/II programs requiring Model-Based Engineering compliance. Pain point: CAD bottleneck during conceptual/preliminary design.

**Pricing model (validated via ICP interviews):**
- Per-generation pricing: $0.50–$2.00 per assembly generation
- Monthly subscription: $200–$500/seat for unlimited use
- Enterprise/program license: $50K–$200K/year for program-level access

**Phase II commercial path:**
- Integrate with SOLIDWORKS via add-in (SOLIDWORKS API / SolidWorks PDM integration)
- CATIA V5 compatibility via Parasolid round-trip
- Siemens Teamcenter integration for PDM/lifecycle management

**SBIR Commercialization:** Phase II deliverable includes SOLIDWORKS add-in pilot deployed at one defense prime contractor for 90-day evaluation. Commercial license negotiations to follow.

---

## 7. Phase I Milestones

| Month | Milestone |
|-------|-----------|
| 1 | Constraint-based assembly validated on 50 test cases; invalidity rate baseline established |
| 2 | Domain D OML strategy verified on 5 airfoil types; 3 aerospace demos complete |
| 3 | Parasolid export integrated; SOLIDWORKS round-trip verified on all 5 demo assemblies |
| 4 | Evaluation harness operational; all 5 metrics instrumented |
| 5 | Remaining 2 aerospace demos complete; LVM score ≥ 0.7 on demo set |
| 6 | Phase I final report; Phase II proposal submitted |

---

## 8. Team

**PI:** [Name, credentials — mechanical engineering + ML background]  
**Co-I:** [Name, credentials — OCCT/CadQuery domain expert]  
**Consultant:** [Name — aerospace structures, former Boeing/Lockheed]

---

## Appendix: SBIR Topic Identification

Search `sbir.gov/solicitations` for the following keywords in DoD solicitations:
- "AI-assisted CAD" / "automated CAD generation"
- "Model-Based Engineering automation"
- "digital engineering tools"
- "generative design for aerospace"
- "natural language to CAD" / "text-to-3D"

Likely relevant offices:
- **AFWERX (Air Force)**: Commercial Solutions Opening, "digital engineering" thrust
- **DARPA I2O**: "AI-accelerated design" programs
- **NAVAIR**: "Advanced manufacturing and design automation"
- **NIST MED**: "Smart manufacturing" solicitations

File under NAICS: 541330 (Engineering Services), 541715 (Research and Development in the Physical, Engineering, and Life Sciences)
