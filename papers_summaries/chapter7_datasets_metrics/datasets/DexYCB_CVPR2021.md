# DexYCB (CVPR 2021)

> Chao, Yang, Xiang, Molchanov, Handa, Tremblay, Narang, Van Wyk, Iqbal, Birchfield, Kautz, Fox. *DexYCB: A Benchmark for Capturing Hand Grasping of Objects.* CVPR 2021. DOI: 10.1109/CVPR46437.2021.00893. Zotero Key: `27SZYANI`.

## Summary
DexYCB is a large-scale, object-diverse, controlled-lighting 3D hand-object grasping benchmark: 8 synchronized RGB-D cameras record 10 subjects grasping 20 YCB objects in 582K grasping sequences. It is the standard evaluation source for grasp pose estimation, handover, and 6D object pose tasks.

## 1. Dataset Purpose
- Addresses the bottleneck of small object count (≤10) and few subjects/actions in existing 3D HOI datasets such as HO-3D, providing a 100×-scale 3D hand-object grasping dataset.
- Tasks: 6D object pose estimation, 3D hand pose estimation, grasp type classification, single-view / multi-view HOI reconstruction.
- Focused on "grasping": single-hand grasping dominates; does not cover in-hand manipulation, bi-manual, articulated objects, or in-the-wild.
- Tightly connected to the NVIDIA Isaac Sim and imitation learning communities, providing massive "demonstration" data.

## 2. Data Composition
- Source: real capture. 10 subjects (5 male, 5 female) grasping 20 YCB-Video objects under controlled lighting.
- Viewpoint: third-person 8 calibrated RealSense RGB-D cameras — a front view plus 6 external cameras surrounding the tabletop.
- Scale: 582,000 grasping sequences, 1000+ grasps per (object × subject × viewpoint) combination; about 50K sequences are used in the standard train/val/test split.
- Object and action: 20 YCB objects; each subject performs 10 grasps per object (5 natural grasping types + 5 functional intents).
- Contains natural occlusion, self-occlusion, and significantly more subject / object diversity than HO-3D.
- No bi-manual, no articulated object, no human-object conversational action.

## 3. Annotation and Supervision
- Hand: 3D 21 joints (multi-view optimization + MANO fitting), MANO shape / pose parameters.
- Object: 6D pose (multi-view + YCB-Video CAD model), aligned per frame.
- Interaction: gripper-style grasp type labels (power, pinch, tripod, etc.); no contact-map ground truth.
- Scene: 8-view RGB + depth, camera intrinsics / extrinsics, object masks, 3D scene point-cloud fusion.
- No robot annotation, no tactile, no language.

## 4. Supported Evaluation
- Benchmark tasks: (1) hand pose estimation (MPJPE, PA-MPJPE, Mesh Error); (2) 6D object pose (ADD-S, AUC); (3) joint hand-object reconstruction; (4) grasp type classification.
- Standard split: S0 (seen subject / seen object / seen grasp) → S4 (unseen subject / unseen object).
- Metrics: MPJPE / PA-MPJPE / Mesh Error / ADD-S / AUC-ADDS / Proj2D / MSSD / MSPD.
- Primarily used for evaluation (test set public), also as a pretraining corpus.

## 5. Why It Matters
- The first truly "large-dataset-scale" 3D hand-object benchmark (HO-3D ≈ 100K, GRAB ≈ 1.2M frames but mocap synthetic, HOT3D ≈ 3.7M but released only in 2025).
- The multi-split protocol for cross-subject / cross-object becomes a standard reference for later GRAB, HOI4D, and HOT3D.
- With its large scale and complete metadata, it is widely used by imitation learning, hand-pose-from-RGB, and behavior-cloning methods as a training set.
- The 20-object collection becomes a "medium-diversity" evaluation band commonly used in 6D pose / 9D pose / category-level pose papers.

## 6. Limitations and Biases
- Only 20 YCB objects: methods can still memorize specific object appearances.
- Tabletop / controlled environment: a clear performance drop when generalizing to in-the-wild egocentric video.
- Only "grasping": missing in-hand manipulation (re-grasp, tool use, bi-manual handoff).
- No contact-map ground truth: contact-based metrics are not available for joint reconstruction evaluation.
- No articulated object, no dynamic contact (sliding, rolling), no tactile.
- Annotations are obtained by 8-camera multi-view optimization and remain biased on transparent / reflective objects.

## 7. Takeaway
DexYCB is best for demonstrating 3D hand-object reconstruction accuracy at "larger scale + more diverse objects + cross-subject evaluation", especially the robustness of hand pose, object pose, and joint reconstruction under seen/unseen combinations. **Not suitable** for evaluating bi-manual, articulated, dynamic contact, or in-the-wild egocentric scenarios. In this survey, it plays the role of "large-scale 3D HOI reconstruction evaluation" and serves as the main evaluation source for "6D-pose-aware hand-object methods" in Ch2 / Ch3 / Ch5.
