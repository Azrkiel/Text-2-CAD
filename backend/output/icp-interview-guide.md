# ICP Discovery Interview Guide

## Objective
Identify the Ideal Customer Profile (ICP) for Mirum Text-to-CAD through structured discovery interviews with potential users. The goal is to understand which mechanical engineering segments, company sizes, and use cases will derive the most value from constraint-based assembly generation.

## Pre-Interview Preparation
- Research the prospect's company, role, and recent projects
- Identify their current CAD workflow (software used, pain points)
- Prepare 2-3 example Mirum outputs relevant to their domain
- Estimate their typical assembly complexity (single-part vs 50+ parts)

---

## Section A: Background & Context (5 minutes)

### A1: Role & Responsibilities
- What is your current role in the organization?
- What CAD activities do you spend the most time on each week?
- What percentage of your work involves assemblies vs single parts?

### A2: Current Workflow
- What CAD software do you primarily use? (Fusion 360, FreeCAD, Inventor, SolidWorks, etc.)
- How long does it typically take you to model an assembly with N parts? (Ask for specific examples)
- What are the bottlenecks in your current CAD process?
- Do you ever outsource CAD work? Under what conditions?

### A3: Project Characteristics
- What types of mechanical assemblies does your team design? (Provide examples if helpful)
- What is the typical size of an assembly? (avg part count, assembly duration)
- How iterative is your design process? (Do you generate many variations?)
- Are there any physical constraints or standards your designs must meet?

---

## Section B: Problem Validation (7 minutes)

### B1: Pain Points with Current Tools
- What frustrates you most about your current CAD software?
- How often do you find yourself manually positioning parts by hand?
- Have you ever tried to script or automate part generation? What happened?
- If you could automate one specific task in CAD, what would it be?

### B2: Time & Cost Impact
- Roughly, what is the cost of a CAD engineer's time per hour in your organization?
- How many hours per month does your team spend on assembly generation?
- If you could reduce that by 50%, what would you do with the freed-up time?

### B3: Constraint-Based Assembly Resonance
- When you hear "describe your assembly in text and the system generates it," what's your first reaction?
- Would it help if the system automatically positioned parts based on kinematic relationships (revolute, slider, etc.) rather than hard-coded XYZ coordinates?
- Have you ever wished you could "just tell the computer what moves" instead of manually positioning?

---

## Section C: Technical Feasibility (5 minutes)

### C1: Design Domain
- Does your team work in any of these domains? (Check all that apply)
  - [ ] Mechanical assemblies (gearboxes, fasteners, brackets)
  - [ ] Aerospace/aviation (wings, fuselages, control surfaces)
  - [ ] Robotics (joints, actuators, linkages)
  - [ ] Consumer products (housings, enclosures)
  - [ ] Industrial machinery (pumps, conveyors)
  - [ ] Other: _______________

### C2: Complexity & Variations
- What's the most complex assembly you've ever modeled? (part count, feature count)
- Do you generate many variants of similar assemblies? (e.g., custom configurations for different customers)
- Are there assemblies you currently avoid modeling because they're too tedious?

### C3: Standards & Constraints
- Do you work with standard parts (fasteners, bearings, stock profiles)?
- Are there design standards or libraries you enforce (ANSI, ISO, DIN)?
- How much of your work involves kinematic analysis (motion simulation)?

---

## Section D: Value Perception (5 minutes)

### D1: Willingness to Adopt
- On a scale of 1–10, how interested would you be in trying a text-based CAD system for 10% of your current projects?
- What would need to be true for you to adopt this tool into your workflow?
- What's the biggest risk or concern you'd have?

### D2: Success Metrics
- If Mirum could save you [estimated hours from B2], would that be valuable?
- What would "success" look like for a tool like this in your organization?
- How would you measure ROI on adopting new CAD tooling?

### D3: Deal-Breaker Questions
- Does the generated CAD need to be 100% collision-free, or is 95% acceptable?
- Would you accept using a text-based design process for initial drafts, then hand-tuning in your primary CAD tool?
- If the system generated 80% of an assembly correctly and you refined the remaining 20%, would that still be valuable?

---

## Section E: Organizational Fit (3 minutes)

### E1: Buying Process
- Who else would need to approve adoption of a new CAD tool in your organization?
- What's your typical procurement timeline for software?
- Would this be a subscription, one-time purchase, or something else?

### E2: Integration & Workflow
- How important is integration with your existing CAD tools? (file formats, plugins, API access)
- Do you prefer cloud-based or desktop-based solutions?
- How critical is data security and IP protection for your organization?

### E3: Company Context
- What's the approximate size of your engineering team?
- How innovative is your organization? (early adopters of new tools, or conservative?)
- Are there any compliance or regulatory requirements we should know about?

---

## Section F: Specific Use Cases (5 minutes)

### F1: Best Fit Scenario
- Describe a specific assembly from your recent work that would have been perfect for a text-to-CAD system.
  - What was the complexity? (part count, features)
  - How long did it take to model?
  - Would Mirum have saved you time?

### F2: Worst Fit Scenario
- Now describe an assembly that would NOT work well with text-based generation.
  - Why would this fail?
  - What would need to improve?

### F3: Dream Use Case
- If you could describe any assembly in text and have it perfectly modeled in 30 seconds, what would that assembly be?
- Why does that specific use case matter to you?

---

## Section G: Closing & Next Steps (3 minutes)

### G1: Interest & Commitment
- Based on this conversation, on a scale of 1–10, how likely are you to pilot Mirum in the next 3 months?
  - If < 7: What would need to change?
  - If ≥ 7: What's the next step?

### G2: Referrals & Feedback
- Do you know anyone else in your network who should know about Mirum?
- Would you be willing to share feedback on a prototype in 4 weeks?
- May we follow up with you in 2 weeks to report progress?

### G3: Closing Question
- Is there anything I should have asked you that I didn't?

---

## Post-Interview Analysis

### Scoring System (ICP Fit)

For each interview, score the prospect on these dimensions (1–5 scale):

1. **Problem Resonance** (1–5)
   - How keenly did they feel the pain points?
   - Did they have a "lightbulb moment"?

2. **Domain Fit** (1–5)
   - Does their primary work domain match Mirum's strengths?
   - (Mechanical assemblies, constraint-based systems = high fit)

3. **Volume & Frequency** (1–5)
   - How much time do they spend on assembly modeling?
   - Do they generate variations/iterations?

4. **Technical Sophistication** (1–5)
   - Can they adopt new tools without extensive training?
   - Do they value automation and scripting?

5. **Buying Authority** (1–5)
   - Do they have decision-making power?
   - Or are they influencers who can advocate?

6. **Organizational Readiness** (1–5)
   - Is their company innovative/early-adopter?
   - Do they have budget for new tools?

**ICP Score = Average of 1–6** (≥ 4.0 = strong ICP fit)

---

## Synthesis & Iteration

### After 5–10 Interviews:
1. Identify common themes in high-scoring prospects
2. Refine targeting: company size, role, domain, use case
3. Update ICP hypothesis
4. Conduct follow-up interviews with refined questions
5. Measure: Are we converting interest into pilots/sales?

### Key Metrics to Track:
- % of interviews that convert to pilots
- Average deal size (value perceived per prospect)
- Time-to-value (how long before ROI is realized)
- Churn risk (early signs of adoption failures)
- NPS score (likelihood to recommend)

---

## Interview Record Template

### Prospect Information
- Name: ___________________
- Company: ___________________
- Role: ___________________
- Domain: ___________________
- Company Size: ___________________
- Date: ___________________

### Scores (1–5)
- Problem Resonance: ___
- Domain Fit: ___
- Volume & Frequency: ___
- Technical Sophistication: ___
- Buying Authority: ___
- Organizational Readiness: ___
- **ICP Score (Avg): ___**

### Key Quotes & Insights
[Notes from the interview]

### Action Items
- [ ] Send follow-up resources
- [ ] Schedule prototype feedback session
- [ ] Introduce to [specific team member]
- [ ] Other: ___________________

### Next Steps
___________________

---

## Appendix: Sample Scenarios

### Scenario 1: Mechanical Fastener Assembly
**Domain:** Manufacturing / Product Design
**Company Size:** 20–100 engineers
**Use Case:** "We design custom bracket assemblies with 3–8 parts. Currently takes 4–6 hours per design. We build hundreds of variations per year."
**ICP Fit:** HIGH (high volume, clear pain point, repeatable)

### Scenario 2: Aerospace Wing Structure
**Domain:** Aerospace / Aerostructures
**Company Size:** 500+ engineers
**Use Case:** "We model wing segments with 200+ internal ribs and spars. Automation is critical, but everything must be parametric and compliant with certification standards."
**ICP Fit:** MEDIUM–HIGH (high complexity, regulatory constraints, large team)

### Scenario 3: One-Off Custom Engineering
**Domain:** Custom Machinery / R&D
**Company Size:** < 10 engineers
**Use Case:** "Every project is unique. We rarely repeat assemblies. Automation wouldn't help much."
**ICP Fit:** LOW (low volume, high customization, limited ROI)

---

## Contact & Feedback
For questions about this interview guide or ICP discovery process, contact: [Product Manager Contact Info]
