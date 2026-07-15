# AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation

## Summary
A robust framework that shifts the paradigm from reconstruction to agentic generation for hand-object interaction learning, replacing fragile neural rendering and SfM with a VLM-guided agentic pipeline that synthesizes a complete watertight object mesh with high-fidelity texture, then propagates pose via an anchor-and-track strategy and contact-aware optimization to enforce physical plausibility.

## 1. Problem and Setting
- Dynamic 4D (3D + time) reconstruction of hand-object interactions from monocular videos.
- Input: monocular RGB video of hand-object interaction, especially challenging in-the-wild footage.
- Output: complete textured 3D object mesh, MANO hand parameters over time, object 6D pose trajectory, and contact-aware consistency.
- Task: dynamic hand-object motion reconstruction with shape completion via generative priors (not neural rendering).

## 2. Core Method
- An agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions.
- An anchor-and-track strategy bypasses fragile SfM: object pose is initialized at a single interaction onset frame using a foundation model and propagated temporally by leveraging the strong visual similarity between the generated asset and video observations.
- A contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility.
- How FM priors are injected: the VLM provides semantic understanding; 3D generative FMs provide shape/texture priors; a foundation model provides pose initialization. The agentic loop coordinates these priors for coherent output.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models used: VLM for planning/coordination, 3D generative model for shape and texture synthesis, foundation model for initial pose estimation.
- Domain knowledge: hand model (MANO), contact physics constraints, physical plausibility heuristics built into the optimizer.
- Training data: foundation models are pre-trained; no HOI-specific fine-tuning. The system uses in-context reasoning and FM coordination at inference.
- Assumption: video exhibits at least one clear interaction onset frame for anchor initialization.

## 4. Experiments and Findings
- Datasets: HO3D, DexYCB, ARCTIC, and in-the-wild videos.
- Metrics: global geometric accuracy (Chamfer, F-score), visual rendering quality, robustness on challenging sequences, real-to-sim retargeting validity.
- AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior arts frequently collapse.
- By prioritizing physical validity, the method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.

## 5. Strengths and Limitations
### Strengths
- Novel agentic paradigm that replaces fragile neural rendering with coordinated FM generation.
- Produces complete, watertight, simulation-ready 3D objects with high-fidelity texture.
- Robust to in-the-wild footage where SfM-based methods fail.
- Validated on real-to-sim retargeting for robotics.

### Limitations
- Multi-agent loop introduces significant computational cost and latency.
- Generation quality is bounded by the underlying 3D generative models.
- Requires VLM-quality reasoning for agent coordination.
- Dependency on multiple foundation models increases system complexity.

## 6. Takeaway
AGILE demonstrates that replacing reconstruction with agentic generation can overcome the long-standing robustness issues of neural rendering and SfM for dynamic HOI. By using foundation models not as passive constraints but as active reasoning components that plan, generate, and track, AGILE produces simulation-ready 3D objects from in-the-wild video — a step toward fully automated agentic AI systems for HOI reconstruction.
