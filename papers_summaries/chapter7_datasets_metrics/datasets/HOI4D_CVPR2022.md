# HOI4D (CVPR 2022)

> Liu, Liu, Jiang, Lyu, Wan, Shen, Liang, Fu, Wang, Yi. *HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction.* CVPR 2022. DOI: 10.1109/CVPR52688.2022.02034. Zotero Key: `X6ZWMZS9`.

## Summary
HOI4D is a large-scale 4D egocentric HOI dataset: 2.4M RGB-D frames, 4000+ sequences, 4 subjects, 800 object instances, 16 categories, 610 rooms, with panoptic / motion / 3D hand / category-level object pose / action and other annotations. It is a composite benchmark for category-level egocentric HOI reconstruction + 4D segmentation + action understanding.

## 1. Dataset Purpose
- Solves the bottleneck of "small scale (≤10 objects), lack of category-level annotation, lack of 4D temporal information" of existing egocentric HOI datasets. HOI4D is the first to take "4D egocentric + category-level" as a unified evaluation dimension.
- Tasks: (1) 4D dynamic point cloud semantic segmentation; (2) category-level 6D object pose tracking; (3) egocentric action segmentation; (4) 3D hand pose estimation.
- Anchors "category-level HOI reconstruction" and "4D dynamic HOI" as independent sub-tasks.
- Complements HOT3D: HOI4D's strength is category-level + 4D, while HOT3D's strength is multi-view + egocentric mocap accuracy.

## 2. Data Composition
- Source: real capture. 4 subjects interact with 800 object instances in 610 different indoor rooms.
- Viewpoint: first-person (egocentric) RGB-D (Kinect-like).
- Scale: 2.4M frames, 4000+ sequences, 4 subjects, 16 object categories, 800 instances, 610 rooms.
- Object and action: 16 categories covering chair, table, cup, bottle, box, bag, and other household items; actions include pick, place, open, use, and handover.
- Contains natural egocentric motion, dynamic object / hand / camera motion.

## 3. Annotation and Supervision
- Hand: 3D 21 joints (multi-view + automatic pipeline).
- Object: category-level 6D pose (independent of specific CAD) + 3D mesh (partial).
- Interaction: action label, contact state, panoptic segmentation (semantic + instance), motion segmentation.
- Scene: 4D dynamic point cloud (changing over time), RGB-D, scene mesh.
- No language instruction, no robot annotation, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) 4D panoptic segmentation (PQ / mIoU); (2) category-level 6D object pose tracking (ADD-S / AUC); (3) egocentric action segmentation (segmental F1 / edit distance); (4) 3D hand pose (MPJPE / PA-MPJPE).
- Key metrics: 4D PQ, mIoU, ADD-S, AUC-ADDS, segmental F1, MPJPE.
- Provides 4 standard splits; cross-subject split for evaluation.
- Promotes the transfer of "category-level 6D pose" from rigid objects to hand-object interaction scenarios.

## 5. Why It Matters
- The first large-scale HOI dataset of "4D egocentric + category-level + 800 object instances".
- 16 categories + 800 instances make "category-level 6D pose" a possible evaluation sub-task in HOI scenarios.
- 4D panoptic + motion segmentation annotations inspire HOI understanding work on 4D dynamic point clouds.
- Together with the new-generation egocentric datasets such as HOT3D (2025), it forms a "4D vs multi-view" dual track.
- The core anchor of the "spatial geometry prior" in Ch3 and the "semantic prior" in Ch4.

## 6. Limitations and Biases
- 4 subjects: subject diversity is low, and cross-cultural style differences are limited in coverage.
- 16 categories: relatively limited, cannot cover industrial / outdoor objects.
- Annotation depends on category-level CAD + multi-view pipeline, and there is still bias on severely occluded frames.
- No contact-map ground truth, no affordance label.
- No articulated-object joint annotation (only rigid objects).
- No language, no robot, no tactile.
- Under the first-person view, self-occlusion is severe, and the 3D hand pose accuracy is affected by the depth sensor.

## 7. Takeaway
HOI4D is best for demonstrating the capability of "4D egocentric + category-level HOI reconstruction and segmentation". **Not suitable** for evaluating hand-only mesh, bi-manual, articulated 4D, language-conditioned, or in-the-wild egocentric (already controlled environments) tasks. In this survey, HOI4D plays the role of "4D egocentric HOI + category-level 6D pose main benchmark" and serves as the hard anchor for evaluating "spatial geometry prior" in Ch3 and "semantic prior" in Ch4 for category-level HOI.
