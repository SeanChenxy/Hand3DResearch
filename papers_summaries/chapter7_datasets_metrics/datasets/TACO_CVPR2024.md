# TACO (CVPR 2024)

> Liu, Yang, Si, Liu, Li, Zhang, Liu, Yi (Tsinghua, Shanghai AI Lab, Shanghai Qi Zhi, BUPT, BIT). *TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding.* CVPR 2024. Zotero Key: `QQWDGZIS`. Project: https://taco2024.github.io.

## Summary
TACO is a large-scale 4D HOI dataset and benchmark focused on "generalizable bi-manual + multi-object + tool use": 2.5K motion sequences, 5.2M video frames, 12 third-person cameras + 1 egocentric, 131 tool-action-object triplets, 20 object categories, 196 object instances, 15 daily actions. It is the flagship benchmark for cross-object / cross-triplet generalization evaluation of bi-manual tool use.

## 1. Dataset Purpose
- Solves the problem that "existing HOI datasets are dominated by unimanual or single-object bi-manual, lacking 'tool-action-object' triplets + cross-object generalization". TACO explicitly takes "generalization to unseen object geometries and novel behavior triplets" as the core evaluation target.
- Tasks: (1) compositional action recognition (identifying tool-action-object triplets); (2) generalizable hand-object motion forecasting (predicting future hand-object motion); (3) cooperative grasp synthesis (synthesizing cooperative grasping under new objects).
- Anchors "bi-manual + tool use + cross-object generalization" as an independent sub-task; forms a "small-scale vs large-scale + cross-generalization" contrast with KIT (only 12 objects, <20 triplets).
- Forms a unique "bi-manual + multi-object" dimension with ARCTIC (rigid + articulated), DexYCB (single-hand + single-object), and H2O (bi-manual + single-object).

## 2. Data Composition
- Source: real capture, multi-view + optical mocap joint pipeline.
- Viewpoint: 12 synchronized FLIR industrial cameras (4096×3000, 30 Hz) + 1 egocentric RealSense L515 (1920×1080) + 6 NOKOV infrared mocap cameras (for marker tracking).
- Scale: 2.5K motion sequences, 5.2M video frames.
- Object and action: 20 object categories, 196 object instances, 15 actions, 131 <tool, action, target> triplets.
- Key design: different levels of overlap between triplets, defining semantic distance, supporting "different generalization degrees" research.
- Each object mesh is obtained by an EinScan industrial 3D scanner (≤100K triangular faces).

## 3. Annotation and Supervision
- Object: 6D pose (4 mocap markers attached to the object surface, tracked by the mocap system); high-precision 3D mesh (≤100K faces).
- Hand: precise hand mesh reconstruction (markerless pipeline: hand keypoint localization + hand pose optimization with multiple losses: L2D / L3D / Langle / Ltc / La / Lp).
- Contact: dense hand-object contact mesh (automatic pipeline inference + optimization).
- Interaction: tool-action-object triplet label (131 categories).
- Scene: 12-view RGB, egocentric RGBD, mocap marker positions, camera extrinsics, object mesh.
- No language instruction; no robot teleoperation annotation.

## 4. Supported Evaluation
- Benchmark tasks: (1) compositional action recognition (triplet classification Top-1); (2) generalizable hand-object motion forecasting (next-N-frame hand pose / motion tendency); (3) cooperative grasp synthesis (physical simulation success rate of generated grasps on new objects).
- Key metrics: action triplet Top-1, motion forecast MPJPE / Mesh Error, grasp synthesis success rate.
- Provides different unseen splits by "object geometry" / "object category" / "triplet" to test different generalization degrees.
- Provides a markerless automatic pipeline, which is theoretically extensible to more objects / triplets.

## 5. Why It Matters
- The first real bi-manual dataset to take "tool-action-object triplet + cross-object generalization" as a unified evaluation dimension.
- 2.5K sequences + 5.2M frames + 131 triplets were the largest scale in "bi-manual tool use" at the time.
- The automatic markerless annotation pipeline solves the difficult problem of accurately annotating "bi-manual + tools".
- Promotes "cross-triplet generalization" as a standard dimension of HOI evaluation.
- A core anchor shared by "semantic prior" in Ch4, "motion generative prior" in Ch5, and "robot learning" in Ch6.
- Inspires subsequent "cross-object / cross-task generalization" HOI paper designs.

## 6. Limitations and Biases
- The scale of 2.5K sequences is still small for cross-triplet generalization.
- Only 20 categories and 196 instances: cross-object diversity is limited by the collection cost.
- No language instruction (only triplet labels), limiting direct VLA application.
- No explicit tracking of bi-manual 6D pose joint (relies on markerless mesh estimation).
- No articulated-object joint tracking (compared with ARCTIC, TACO's tools are mostly rigid).
- No tactile, no force, no dynamic contact (sliding / rolling) annotation.
- The marker attachment of the mocap system has limitations for small / transparent objects.
- The systematic bias of the automated annotation pipeline propagates to all sequences.

## 7. Takeaway
TACO is best for demonstrating the capability of "bi-manual tool-action-object triplet HOI understanding + cross-object generalization". **Not suitable** for evaluating articulated 4D, language-conditioned VLA, in-the-wild tasks, or tactile-rich tasks. In this survey, TACO plays the role of "bi-manual tool use + cross-triplet generalization main benchmark" and serves as the core anchor shared by "semantic prior" in Ch4, "motion generative prior" in Ch5, and "structured HOI supervision" in Ch6.
