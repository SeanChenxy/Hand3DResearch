# ContactPose (ECCV 2020)

> Brahmbhatt, Tang, Twigg, Kemp, Hays. *ContactPose: A Dataset of Grasps with Object Contact and Hand Pose.* ECCV 2020. DOI: 10.1007/978-3-030-58601-0_22. Zotero Key: `WT5JAMT8`.

## Summary
ContactPose provides contact map + hand pose + object pose + RGB-D data (2.9M images) for 2,306 unique grasps × 25 household objects × 50 subjects × 2 functional intents. It is the pioneering dataset for "grasp-level contact" research.

## 1. Dataset Purpose
- Solves the gap that "existing grasp datasets only provide hand pose, not dense contact". ContactPose takes "contact" as the core annotation of the dataset.
- Tasks: (1) contact map prediction (hand surface / object surface contact region); (2) hand pose estimation (at grasp time); (3) 6D object pose estimation; (4) grasp type / functional intent classification.
- Anchors "contact modeling" as an independent sub-task; provides ground truth for subsequent methods such as CPF, CP3, and ContactGen.

## 2. Data Composition
- Source: real capture. 50 subjects grasp 25 household objects in a controlled studio.
- Viewpoint: third-person multi-camera RGB-D (multi-view Kinect, etc.), providing depth fusion.
- Scale: 2,306 unique grasps × 2 functional intents × 50 subjects × 25 objects; totaling 2.9M RGB-D grasp images.
- Object and action: 25 household objects (box, mug, knife, bottle, bowl, etc.); 2 functional intent categories (use / use-other-hand).
- Each grasp corresponds to a specific "functional" action intent.

## 3. Annotation and Supervision
- Hand: 3D 21 joints (multi-view + MANO fitting), MANO β / θ.
- Object: 6D pose (multi-view), 3D object mesh.
- Contact: vertex-level contact map (hand surface + object surface), contact area values.
- Interaction: grasp type labels, functional intent labels, hand configuration labels.
- Scene: multi-view RGB-D, camera intrinsics / extrinsics, point-cloud fusion.
- No language, no robot, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) contact map prediction (vertex-level F-score @ threshold); (2) hand pose (MPJPE / PA-MPJPE); (3) object pose (ADD / AUC); (4) grasp type classification.
- Key metrics: contact F-score @ 1mm / 5mm / 10mm, MPJPE, AUC-ADDS, grasp Top-1.
- Provides an "unseen object" split to evaluate cross-object generalization.
- The de facto standard evaluation source for contact modeling.

## 5. Why It Matters
- The first dataset to publicly take "dense contact map" as the core ground truth for grasping.
- 2,306 unique grasps + 50 subjects were among the largest "grasp-level" datasets at the time.
- The 2 functional intents enable "intent-conditioned" contact modeling.
- Inspired follow-up work such as ContactDB, AffordPose, and ContactGen.
- The core reference dataset of the "spatial geometry prior" in Ch3 and the "affordance prior" in Ch4.

## 6. Limitations and Biases
- Only 25 household objects: object diversity is limited.
- 2 functional intents: coarse granularity (other datasets can have 6+ categories).
- Still controlled studio: a large distribution gap to in-the-wild egocentric video.
- No bi-manual, articulated, tool use, or dynamic manipulation.
- No language instruction, no robot teleoperation annotation.
- Annotation depends on multi-view RGB-D optimization and may fail on reflective / transparent objects.

## 7. Takeaway
ContactPose is best for demonstrating the capability of "grasp-level dense contact prediction", especially functional-intent-conditioned contact modeling. **Not suitable** for evaluating bi-manual, articulated, RGB-based in-the-wild, language-conditioned, or robot-relevant manipulation. In this survey, ContactPose plays the role of "dense contact modeling main benchmark" and serves as the unified anchor for evaluating "spatial geometry prior" in Ch3 and "affordance prior" in Ch4 for contact modeling.
