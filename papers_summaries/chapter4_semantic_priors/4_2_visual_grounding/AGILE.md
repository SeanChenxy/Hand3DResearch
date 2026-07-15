# AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation (Cross-reference)

## Summary
This entry is a cross-reference to the detailed summary in Chapter 3 (3D Geometry Priors, section 3.2 Shape Completion). AGILE shifts the paradigm from reconstruction to agentic generation, using a VLM-guided agentic pipeline that synthesizes a complete, watertight object mesh with high-fidelity texture, propagates pose via an anchor-and-track strategy, and applies contact-aware optimization to enforce physical plausibility.

## 1. Problem and Setting
- Dynamic 4D reconstruction of hand-object interactions from monocular videos.
- Input: monocular RGB video of hand-object interaction, especially in-the-wild footage.
- Output: complete textured 3D object mesh, MANO hand parameters over time, object 6D pose trajectory.
- Visual grounding prior: the VLM provides visual-grounded planning and critique, identifying the object category and guiding the reconstruction process.

## 2. Core Method
- An agentic pipeline where a VLM guides a 3D generative model to synthesize a complete, watertight object mesh.
- An anchor-and-track strategy bypasses fragile SfM: object pose is initialized at a single interaction onset frame using a foundation model and propagated temporally.
- A contact-aware optimization integrates semantic, geometric, and interaction stability constraints.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models: VLM for planning/critique; 3D generative model for shape/texture; foundation model for pose initialization.
- Domain knowledge: hand model (MANO); contact physics constraints.
- Training data: foundation models are pre-trained; no HOI-specific fine-tuning.
- Assumption: video exhibits at least one clear interaction onset frame for anchor initialization.

## 4. Experiments and Findings
- Datasets: HO3D, DexYCB, ARCTIC, in-the-wild videos.
- AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences.
- The agentic refinement loop consistently improves reconstruction quality over single-pass generation.

## 5. Strengths and Limitations
### Strengths
- Novel agentic paradigm leveraging VLM reasoning for iterative refinement.
- Produces complete, textured 3D outputs usable for graphics/VR.
- Robust to in-the-wild footage where SfM-based methods fail.
- Validated on real-to-sim retargeting for robotics.

### Limitations
- Multi-agent loop introduces computational cost and latency.
- Generation quality is bounded by the underlying 3D generative models.
- Requires careful prompt engineering.
- Dependency on multiple FMs increases system complexity.

## 6. Takeaway
AGILE demonstrates that replacing reconstruction with agentic generation can overcome the long-standing robustness issues of neural rendering and SfM for dynamic HOI. In the context of visual grounding (chapter 4), the VLM serves as the central reasoning component that grounds the reconstruction in visual semantics. See chapter 3 section 3.2 for the full technical details.
