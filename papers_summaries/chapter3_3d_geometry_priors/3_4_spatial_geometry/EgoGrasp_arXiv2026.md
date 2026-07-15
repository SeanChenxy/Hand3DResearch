# EgoGrasp: World-Space Hand-Object Interaction Estimation from Dynamic Egocentric Videos

## Summary
EgoGrasp is the first method to reconstruct world-space hand-object interactions (W-HOI) from dynamic egocentric videos, supporting open-vocabulary objects, through a multi-stage framework combining foundation-model-based preprocessing, body-guided diffusion for hand pose estimation, and an HOI-prior-informed diffusion for hand-aware 6DoF object pose infilling.

## 1. Problem and Setting
- World-space hand-object interaction (W-HOI) reconstruction from dynamic egocentric videos, supporting open-vocabulary objects.
- Input: dynamic egoview video showing hand-object interaction; no templates required.
- Output: world-space hand poses and 6DoF object pose trajectories, with hand-object interaction constraints.
- Task: world-space hand-object interaction estimation; uses multiple foundation model priors.

## 2. Core Method
- Three-stage framework:
  1. Robust preprocessing pipeline leveraging vision foundation models for initial 3D scene, hand, and object reconstruction.
  2. Body-guided diffusion model that incorporates explicit egocentric body priors for hand pose estimation.
  3. HOI-prior-informed diffusion model for hand-aware 6DoF object pose infilling, ensuring physically plausible and temporally consistent W-HOI estimation.
- How FM priors are injected: vision foundation models (for initial 3D scene/hand/object reconstruction) + diffusion models (conditioned on body and HOI priors) for hand and object pose estimation.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models: vision foundation models (e.g., DINOv2, SAM) for preprocessing; body-guided diffusion for hand; HOI-prior-informed diffusion for object.
- Domain knowledge: egocentric body priors, hand-object interaction physical constraints, open-vocabulary object support.
- Training data: large-scale egocentric video datasets; diffusion models trained on hand-object interaction data.
- Assumption: hand-object categories are open-vocabulary but can be inferred from visual cues.

## 4. Experiments and Findings
- Datasets: egocentric HOI benchmarks (HOT3D, Epic-Kitchens, ARCTIC).
- Metrics: world-space hand pose accuracy, 6DoF object pose accuracy, hand-object alignment, temporal consistency.
- Achieves state-of-the-art performance in W-HOI reconstruction, handling multiple and open-vocabulary objects robustly.
- The combination of foundation-model preprocessing + body-guided diffusion + HOI-prior diffusion is critical for performance.

## 5. Strengths and Limitations
### Strengths
- First W-HOI method supporting open-vocabulary objects from dynamic egocentric videos.
- Multi-stage framework leverages complementary FM strengths.
- Body-guided and HOI-prior diffusion provide physically grounded pose estimation.
- Handles occlusions and open-set categories robustly.

### Limitations
- Multi-stage pipeline is complex and may have error accumulation.
- Diffusion-based inference is slower than feed-forward methods.
- Requires significant compute for training and inference.
- Depends on the quality of foundation model preprocessing.

## 6. Takeaway
EgoGrasp demonstrates that reconstructing world-space hand-object interactions from dynamic egocentric videos requires a careful orchestration of multiple foundation model capabilities — preprocessing, body-guided generation, and HOI-prior diffusion — each addressing a specific challenge. The work pushes the boundary of HOI reconstruction from local camera coordinates to global world-space, opening up applications in embodied AI and AR/VR that require spatially consistent hand-object understanding.
