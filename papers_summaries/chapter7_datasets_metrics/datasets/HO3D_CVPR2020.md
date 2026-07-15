# HO-3D / HOnnotate (CVPR 2020)

> Hampali, Rad, Oberweger, Lepetit. *HOnnotate: A Method for 3D Annotation of Hand and Object Poses.* CVPR 2020. DOI: 10.1109/CVPR42600.2020.00326. Zotero Key: `V9JIHJUS`.

## Summary
HOnnotate proposes a multi-view RGB-D joint-optimization auto-annotation pipeline and uses it to construct the HO-3D dataset (v1/v2, 77,558 frames, 10 subjects, 10 YCB objects) — the first 3D hand-object interaction benchmark that provides 6D object pose together with full MANO hand-mesh ground truth.

## 1. Dataset Purpose
- Addresses the difficulty of obtaining "3D hand + 6D object" ground truth in real images. Earlier datasets either provided hands only (PANOPTIC) or annotated hand and object separately (H2O, FPHA), and most relied on mocap / markers.
- Tasks: 3D hand pose estimation, 6D object pose estimation, joint hand-object reconstruction, and grasp understanding.
- Bridges from hand-only to hand-object with single hand + rigid object; no bi-manual interaction and no articulated objects.
- Anchors the joint hand-object reconstruction evaluation paradigm.

## 2. Data Composition
- Source: real capture. Multiple subjects grasp and manipulate YCB objects with a single hand on a tabletop.
- Viewpoint: third-person multi-view RGB-D (1–5 Intel RealSense cameras), with synchronized depth per sequence.
- Scale (v1/v2): 77,558 frames, 68 sequences, 10 subjects, 10 YCB objects.
- Object and action coverage: drill, scissors, can, box, bleach, mug, pitcher, sugar, knife, hammer, etc., covering tools, kitchen items, and packaging containers.
- Actions: single-hand grasping, lifting, manipulating, and placing; includes natural hand-object self-occlusion and mutual occlusion.
- v3 extends to 103,462 frames with the same 10 objects / 10 subjects.

## 3. Annotation and Supervision
- Hand: 3D 21 joints (multi-view bootstrapping); MANO shape β and pose θ (multi-view optimization + manual verification); hand mesh.
- Object: YCB-Video CAD models; 6D object pose obtained via multi-view RGB-D joint optimization and then manually refined.
- Interaction: grasp type (functional vs non-functional), contact state (inferred from hand-object proximity).
- Scene: per-camera RGB + depth, camera intrinsics/extrinsics, object segmentation masks.
- No robot / language / contact map ground truth.

## 4. Supported Evaluation
- Benchmark tasks: (1) 3D hand pose (MPJPE / PA-MPJPE / Mesh Error); (2) 6D object pose (ADD(-S) / AUC); (3) joint hand-object reconstruction.
- Key metrics: hand MPJPE / PA-MPJPE / Mesh Error; object ADD / ADD-S / <0.1d AUC.
- Serves both as a training set and an evaluation set: the mainstream supervised benchmark for 3D HOI reconstruction.
- Limited cross-object generalization (only 10 YCB objects); cross-subject splits are available.

## 5. Why It Matters
- Provides, for the first time in real images, "hard" ground truth of hand mesh + 6D object pose simultaneously.
- The multi-view optimization method becomes the foundation of later multi-view annotation pipelines for HO-3D v3, HOT3D, ARCTIC, etc.
- Establishes "joint hand-object reconstruction" as an independent sub-task, adopted by Hasson 2019, CPF, HMP, and follow-up works.
- Demonstrates that occlusion — not hand articulation complexity — is the critical bottleneck of hand-object performance, driving the community toward occlusion-robust / contact-aware methods.
- Serves as a default benchmark in nearly every 3D hand-object reconstruction paper.

## 6. Limitations and Biases
- Only 10 YCB objects: methods can memorize the appearance and geometry of specific objects.
- Tabletop / controlled lighting: significant distribution gap to in-the-wild egocentric video.
- Single-hand grasping dominates: no bi-manual, no articulated objects (the scissors joint is not tracked), and no tool-use dynamics.
- Annotation depends on depth sensors: systematic bias in MANO fitting near depth discontinuities.
- No contact map, no affordance, no language, no robot teleoperation annotation.

## 7. Takeaway
HO-3D (HOnnotate) is best suited to demonstrate the accuracy of joint hand-object 3D reconstruction under real occlusion, particularly the joint recovery of MANO hand mesh and 6D object pose. **Not suitable** for evaluating bi-manual, articulated-object, in-the-wild generalization, or robot-relevant manipulation. In this survey, it is the foundational 3D HOI reconstruction benchmark that extends hand-only (FreiHAND) to hand-object and defines the standard evaluation protocol for the following decade.
