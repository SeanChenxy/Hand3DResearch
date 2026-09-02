# AffordPose: A Large-scale Dataset of Hand-Object Interactions with Affordance-driven Hand Pose

**Authors:** Juntao Jian, Xiuping Liu, Manyi Li, Ruizhen Hu, Jian Liu  
**Date:** 2023 (ICCV 2023)  
**Identifier:** DOI `10.1109/ICCV51070.2023.01352`  
**Zotero item:** `F99HQWYJ` ([Zotero](zotero://select/library/items/F99HQWYJ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

AffordPose addresses the gap that prior hand-object interaction datasets encode interaction purpose only as coarse human objectives (e.g., use, handover) or object-specific scripts, which does not explain where and how the hand should interact. The paper introduces a large-scale dataset of 26.7K fine-grained hand-object interactions over 641 3D objects, where each interaction is driven by a part-level, hand-centered affordance label (twist, pull, handle-grasp, etc.) paired with a manually adjusted MANO hand pose. Evaluation covers affordance understanding from hand poses and affordance-conditioned hand-object interaction generation. The data shows that hand poses correlate strongly with affordance while retaining diversity from personal habit.

## Background and Motivation

The paper argues that performing appropriate hand-object interaction requires an agent to understand the object's functional role, select a contacting location, and adopt a suitable hand pose. Existing datasets such as HO-3D, DexYCB, and Obman focus on general stable grasping, while intent-annotated datasets use either human objectives (e.g., use, pass, lift in GRAB and OakInk) that treat objects as generic shapes, or object-centric scripts (e.g., "pour juice" in FPHAB) with limited generalization. AffordPose instead adopts part-level hand-centered affordances that localize the interaction and generalize across object categories, motivating a dataset that reveals how affordance shapes the detailed arrangement of hand poses.

## Dataset Construction

Data is built in two volunteer-driven stages. First, 641 objects from 13 categories of PartNet and PartNet-Mobility are annotated with part-level affordance labels chosen from 8 hand-centered types (handle-grasp, press, lift, pull, twist, wrap-grasp, support, lever); five volunteers discuss and reach consensus per part on the finest PartNet hierarchical segmentation, with non-functional parts marked as no affordance. Second, 14 volunteers manually adjust the position, rotation, and 16 joint angles of a MANO hand model (fixed shape parameter) in the GraspIt! simulator, guided by the colored affordance parts and checked by force analysis to avoid penetration and implausible poses. The result is 26,712 interactions, averaging 42 annotations per object model with at least 28 per object, each storing the 3D object shape, affordance label, and hand pose parameters. Per-category counts range from 1,400 (earphone, laptop) to 3,052 (mug) interactions.

## Evaluation Protocol

Two main tasks use an 8-1-1 train-val-test split. Affordance understanding feeds hand pose parameters plus an object point cloud into a DGCNN-based network that either classifies the affordance label (accuracy) or predicts per-point part labels (IoU), with inputs restricted to intrinsic joint configurations or all pose parameters. Interaction generation compares the GrabNet baseline (object-only) against AffordPoseNet, which concatenates a one-hot affordance condition with the object feature; metrics are penetration depth, solid intersection volume, contact ratio, and affordance accuracy of the contacting part. The generation test set is expanded with 211 objects that have affordance labels but no hand poses, since these metrics need no ground-truth hand. RGB-based variants render 3 random viewpoints per interaction for ResNet-18 interaction classification and an I2LMeshNet-based mesh recovery with the affordance as input condition.

## Findings and Analysis

Affordance understanding reaches 94.40% mean classification accuracy and 95.36% mean localization IoU with intrinsic parameters only, rising to 98.39% and 96.29% with all pose parameters, supporting the paper's claim of high affordance-pose correlation; the pull affordance is hardest (77.78% IoU) because its contacting regions, such as bag zippers, are small. In generation, AffordPoseNet and GrabNet produce similar physical quality (AffordPoseNet slightly worse in solid intersection volume), but AffordPoseNet matches the requested affordance at 83.51% mean accuracy while GrabNet tends to output poses of the most frequent affordance for multi-affordance objects; pull and twist remain low (affordance accuracy 0 and 53.13% in the reported table) for small targets. RGB classification attains 97.31% mean precision and 97.29% recall, and mesh recovery yields 16.4 mm MPVPE and 0.1892 MPJRE, with pull, lift, and support worst due to pose diversity and occlusion.

## Contributions

The paper contributes the AffordPose dataset of 26.7K affordance-driven hand-object interactions with 3D shapes, part-level affordance labels, and manually annotated MANO poses; a data analysis linking affordances to joint configurations, contacting fingers, and per-joint variability; and benchmark protocols for affordance understanding and affordance-oriented interaction generation plus image-based classification and mesh recovery.

## Limitations

The dataset contains static, simulator-annotated single-hand interactions on synthetic object meshes rather than dynamic real-world captures. Generated hands often contact the wrong region for affordances with small targets, and the paper states that post-processing with a physical simulator would be needed to fix contact. The conclusions also note that scaling to dynamic, bimanual, and human-robot cooperative interactions requires more efficient, semi-automatic annotation methods.
