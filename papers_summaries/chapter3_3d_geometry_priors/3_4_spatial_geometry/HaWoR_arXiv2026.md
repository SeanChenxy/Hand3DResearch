# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

## Summary
HaWoR reconstructs high-fidelity hand motion in world coordinates from egocentric videos by decoupling the task into camera-space hand motion reconstruction and world-frame camera trajectory estimation via an adaptive egocentric SLAM, plus a motion infiller for out-of-view frames, achieving state-of-the-art performance on hand motion reconstruction and world-frame camera trajectory estimation.

## 1. Problem and Setting
- World-space (global) hand motion reconstruction from egocentric videos captured by moving cameras.
- Input: egocentric RGB video from a moving camera.
- Output: 3D hand meshes in world coordinates over time, plus the camera trajectory in the world coordinate system.
- Task: hand motion reconstruction in world space; uses 3D scene geometry priors (SLAM, 3D foundation).

## 2. Core Method
- Decouples the task by:
  1. Reconstructing hand motion in camera space.
  2. Estimating the camera trajectory in the world coordinate system.
- An adaptive egocentric SLAM framework provides robust camera trajectory estimation under challenging camera dynamics.
- A motion infiller network robustly completes the hand motion trajectory when the hands move out of view frustum.
- How FM prior is injected: 3D scene geometry priors (likely from a DUSt3R-style 3D foundation model) provide the camera trajectory estimation; the hand motion infiller uses temporal generative priors.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: 3D scene foundation model for SLAM-style camera pose estimation.
- Domain knowledge: hand model (MANO); SLAM principles adapted for egocentric settings; hand motion infilling.
- Training data: egocentric video datasets; hand tracking datasets.
- Assumption: the camera's world-frame trajectory can be reliably estimated from the egocentric video.

## 4. Experiments and Findings
- Datasets: egocentric benchmarks (HOT3D, Epic-Kitchens, Aria, etc.).
- Metrics: world-space hand motion accuracy (MPJPE in world frame), camera trajectory accuracy.
- Achieves state-of-the-art performance on both hand motion reconstruction and world-frame camera trajectory estimation.
- The motion infiller effectively handles out-of-view frames, a key failure mode of prior egocentric hand methods.

## 5. Strengths and Limitations
### Strengths
- Decoupled design cleanly separates hand and camera trajectory estimation.
- Adaptive egocentric SLAM robust to dynamic camera motion.
- Motion infiller handles out-of-view frames.
- World-space output is more useful for AR/VR and embodied AI than camera-space.

### Limitations
- Two-stage pipeline (hand + camera) may have error accumulation.
- SLAM can fail in featureless or highly dynamic scenes.
- Motion infiller may produce implausible trajectories for long out-of-view periods.
- Hand-only; no object reconstruction or interaction modeling.

## 6. Takeaway
HaWoR demonstrates that decoupling world-space hand motion reconstruction into camera-space hand tracking and world-frame camera trajectory estimation is an effective strategy, especially when complemented by a motion infiller for out-of-view frames. The work bridges single-image hand reconstruction and SLAM, providing a complete pipeline for world-space hand motion that is directly applicable to AR/VR and embodied AI scenarios where hand position in the world matters.
