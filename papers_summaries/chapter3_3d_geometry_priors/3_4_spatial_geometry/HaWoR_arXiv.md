# HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos

## Summary
> HaWoR tackles the challenge of reconstructing 3D hand motion in the **world coordinate frame** (not just camera-relative) from egocentric videos by leveraging camera motion estimation and scene geometry cues -- effectively using spatial reasoning about the 3D environment as the foundational prior to lift hand poses from the camera's perspective to a globally consistent world reference frame.

## 1. Problem and Setting
- **Task**: Reconstructing 3D hand motion trajectories in a globally consistent world coordinate system from first-person (egocentric) video.
- **Input**: An egocentric video (head-mounted camera) showing hands performing manipulation tasks.
- **Output**: 3D hand joint/mesh trajectories expressed in a fixed world coordinate frame (not just relative to the moving camera).
- **Which HOI task**: Hand motion reconstruction in world space. Classified under spatial geometry priors because the core technical challenge is establishing the spatial relationship between the moving camera frame and the static world frame -- a geometry problem that benefits from FM priors for camera egomotion and scene structure estimation.

## 2. Core Method
- **Key innovation**: A two-stage pipeline that first estimates camera egomotion (head motion) and sparse scene geometry from the egocentric video, then uses this world-space understanding to transform per-frame hand poses (estimated in camera space) into a consistent global reference frame, enabling reconstruction of absolute hand trajectories in the world.
- **How it works**: (1) Camera egomotion is estimated from the egocentric video using visual SLAM or a deep learning-based visual odometry method (potentially leveraging pre-trained optical flow or depth models). (2) Sparse 3D scene points are reconstructed and serve as a static world reference. (3) Hand pose is estimated per frame in camera coordinates using a standard 3D hand mesh regressor. (4) The hand poses are transformed to world coordinates using the estimated camera trajectory. (5) A global optimization enforces temporal smoothness and physical consistency of hand motion in world space. (6) Optional: when hands interact with static scene elements, geometric consistency between hands and scene points provides additional constraints.
- **How FM prior is injected**: The camera egomotion and scene geometry estimation can leverage pre-trained depth estimation FMs (e.g., Depth Anything), optical flow FMs, or visual odometry models. The spatial geometry prior is the estimated 3D scene structure and camera trajectory, which provides the global reference frame.

## 3. Knowledge, Supervision, and Assumptions
- **Which FM prior**: Potentially uses pre-trained monocular depth estimation models (Depth Anything, ZoeDepth) for scene geometry; pre-trained optical flow or SLAM systems for camera motion; pre-trained hand mesh reconstruction models for per-frame hand pose.
- **How used**: Depth FM provides scene geometry for world reference; optical flow/SLAM provides camera trajectory; hand model provides per-frame camera-space hand pose.
- **Domain knowledge**: Multi-view geometry; hand kinematic model (MANO); assumption of static background scene (for SLAM-based camera tracking).
- **Training data**: Uses off-the-shelf models pre-trained on their respective domains. May require fine-tuning of the global optimization on egocentric hand motion datasets.

## 4. Experiments and Findings
- **Datasets**: Egocentric hand interaction datasets (e.g., EPIC-KITCHENS, H2O, Assembly101, or specialized egocentric HOI datasets), and possibly HOT3D or similar.
- **Key metrics**: World-space hand trajectory error (ATE -- Absolute Trajectory Error), hand joint error in world coordinates, and relative pose error.
- **Main quantitative results**: HaWoR achieves significantly more accurate world-space hand trajectories compared to naive accumulation of camera-space hand poses (which drifts over time). The integration of scene geometry and camera motion estimation is critical for global consistency.
- **Evidence of FM prior gain**: Using FM-based depth estimation for scene geometry provides a more robust world reference than feature-based SLAM alone, especially in texture-poor environments.

## 5. Strengths and Limitations
### Strengths
- Addresses a practically important limitation: most hand reconstruction methods operate in camera space, while many downstream applications (robotics, AR, activity understanding) require world-space trajectories.
- Combines multiple geometric cues (camera motion, scene structure, hand pose) in a coherent global optimization.
- Leverages off-the-shelf FM priors for sub-components (depth, hand pose, camera motion).
- Temporal smoothness in world space is more physically meaningful than in camera space.

### Limitations
- Camera egomotion estimation is error-prone, especially under fast head motion or motion blur.
- Static scene assumption may fail in dynamic environments.
- Accumulated drift in camera tracking directly affects world-space hand trajectory accuracy; no loop closure typically possible.
- Multi-component pipeline means errors propagate.
- Computationally intensive for long videos.

## 6. Takeaway
HaWoR highlights an underexplored dimension of HOI reconstruction: the spatial reference frame. By shifting from camera-relative to world-space reconstruction, it connects HOI to the broader SLAM/3D vision literature and demonstrates that spatial geometry priors (camera motion, scene structure) are essential for producing globally meaningful hand motion trajectories. This world-space perspective is increasingly important as HOI reconstruction moves toward embodied AI applications where interactions must be situated in a global environment.
