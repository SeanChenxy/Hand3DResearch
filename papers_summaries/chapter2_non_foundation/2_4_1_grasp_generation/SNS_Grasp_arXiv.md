# SNS-Grasp: Semantic-guided Noise Scaling for Grasp Generation

## Summary
Proposes a semantic-aware noise scheduling strategy for diffusion-based grasp generation, where the noise scale adapts to the kinematic sensitivity of different hand joints, improving semantic alignment and physical feasibility of generated grasps.

## 1. Problem and Setting
- Task: generate semantically meaningful hand grasps that align with task intent (e.g., "grasp to pour" vs. "grasp to hold") using diffusion models.
- Input: 3D object mesh + optional task semantics; output: MANO hand grasp parameters.
- Key challenge: standard diffusion models use isotropic noise schedules (same noise level for all parameters), but different hand joints have vastly different sensitivity — small changes in finger flexion can drastically alter contact while wrist translation is more robust. This mismatch causes diffusion models to produce grasps with suboptimal semantic alignment or physical feasibility.

## 2. Core Method
- Per-joint noise scaling: instead of adding isotropic Gaussian noise to all MANO parameters during the diffusion forward process, SNS-Grasp scales the noise per parameter group (wrist pose, finger joints, hand shape) based on (a) kinematic sensitivity (how much a small parameter change moves the fingertip position) and (b) semantic sensitivity (how much each joint contributes to task-specific contact patterns).
- Semantic-guided noise schedule: joints critical for the specified task semantics (e.g., index finger and thumb for a precision pinch) receive lower noise (preserved more faithfully), while task-irrelevant joints receive higher noise (more flexibility for diversity).
- Standard diffusion backbone (DDPM or similar) with this modified noise schedule; MANO parameter prediction at inference.
- Key innovation: semantically adaptive, non-isotropic noise scheduling that respects the heterogeneous sensitivity of hand articulation.

## 3. Knowledge, Supervision, and Assumptions
- Training data: GRAB or similar grasp datasets with task annotations, or synthetically annotated with per-grasp semantic labels.
- Supervision: standard diffusion denoising loss but computed with scaled noise.
- Domain knowledge: MANO kinematic chain; Jacobian-based joint sensitivity analysis; task-to-joint importance mapping.
- Assumption: per-grasp semantic labels are available; kinematic sensitivity is a reasonable proxy for semantic importance.

## 4. Experiments and Findings
- Datasets: GRAB (with task labels), ObMan (with heuristic task labels).
- Metrics: grasp success rate (simulation), semantic alignment (how well the generated grasp matches the intended task type, often via user study), physical feasibility (penetration, contact quality).
- Main findings: SNS-Grasp improves semantic alignment over isotropic-noise baselines without sacrificing physical plausibility; per-joint noise scaling is especially beneficial for precision tasks (e.g., pinch grasps) where isotropic noise tends to "wash out" fine finger coordination; the noise scaling is lightweight and compatible with any diffusion backbone.

## 5. Strengths and Limitations
### Strengths
- Simple but well-motivated insight: not all joints are equally sensitive, and diffusion noise should reflect that.
- Compatible with existing diffusion architectures; minimal overhead.

### Limitations
- Requires per-joint sensitivity analysis, which is task-dependent and may need re-computation for new task types.
- Semantic-to-joint importance mapping is heuristic; may not capture all task nuances.
- Limited to static grasping; no temporal dynamics.

## 6. Takeaway
SNS-Grasp highlights an often-overlooked aspect of diffusion-based grasp generation: the isotropic noise assumption breaks down for articulated bodies like the human hand, where different joints have different kinematic and semantic importance. Adapting noise schedules to joint sensitivity is a simple, effective improvement that bridges the gap between generative flexibility and task-specific precision.
