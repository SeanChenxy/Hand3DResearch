# AssemblyHands: Towards Egocentric Activity Understanding via 3D Hand Pose Estimation

**Authors:** Takehiko Ohkawa, Kun He, Fadime Sener, Tomas Hodan, Luan Tran, Cem Keskin  
**Date:** 2023 (CVPR 2023)  
**Identifier:** DOI `10.1109/CVPR52729.2023.01249`  
**Zotero item:** `MBPGMKXZ` ([Zotero](zotero://select/library/items/MBPGMKXZ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

AssemblyHands is a large-scale benchmark of 3D hand pose annotations for egocentric images with challenging hand-object interactions, built on top of Assembly101. Because the original Assembly101 hand poses (from an egocentric tracker) average 27.55 mm error, the authors build a multi-view exocentric annotation network with iterative refinement that reaches 4.20 mm, an 85% error reduction, and scale it to 3.0M annotated images including 490K egocentric ones. A single-view egocentric pose baseline (SVEgoNet) and a verb-classification evaluation show that more accurate hand poses directly improve egocentric action recognition.

## Background and Motivation

Assembly101 showed that 3D hand poses predict procedural actions better than video features, but its egocentric-tracker annotations are inaccurate under hand-object occlusion and narrow stereo baselines, leaving open how pose quality affects action recognition. Most hand pose datasets target static exocentric cameras; egocentric benchmarks like FPHA use magnetic sensors with few subjects. AssemblyHands therefore combines synchronized egocentric and exocentric images from Assembly101 with high-quality automatic multi-view annotation to create the largest egocentric 3D hand pose benchmark and to evaluate poses through downstream action classification.

## Dataset Construction

Images are sampled from Assembly101's rig of 8 static RGB and 4 headset monochrome cameras. Manual annotation covers 62 video sequences sampled at 1 Hz: multi-view 2D keypoints are labeled and triangulated into 21 world-space joints per hand, producing 22K frames (19.2K train / 3.0K eval) from 14 subjects. These train MVExoNet, a volumetric multi-view network (EfficientNet encoders, learnable feature triangulation, V2V-PoseNet, soft-argmax) that is run with iterative refinement rounds to re-crop hands and re-center the volume. Applied to a 30 Hz subset, the automatic pipeline labels 2.81M frames (468K egocentric) from 20 disjoint subjects — 21 times more labels than manual effort. In total, AssemblyHands (M+A) provides 3.0M images and 34 subjects; the paper notes it surpasses InterHand2.6M in total annotated images and offers eight times H2O's subjects.

## Evaluation Protocol

Annotation quality is measured by MPJPE (mm) and PCK-AUC against the manually annotated test set, and for generalization on the Desktop Activities subset of the Aria Pilot Dataset with 12 exocentric cameras and novel YCB objects. For pose estimation, a single egocentric image input requires predicting 21 wrist-relative 3D joints; the SVEgoNet baseline uses a ResNet-50 with 2.5D heatmaps and a hand-identity classification branch (left/right/both). The novel downstream task is verb classification: MS-G3D consumes 42-keypoint sequences (both hands) and classifies six movement-heavy verbs (pick up, position, screw, put down, remove, unscrew), comparing poses from SVEgoNet, the original UmeTrack tracker, and the automatic annotations as an upper bound.

## Findings and Analysis

The annotation network cuts error from 27.55 mm (egocentric-only, the original Assembly101 labels) and 7.97 mm (2D + triangulation) to 4.20 mm after three refinement rounds; on Desktop Activities it degrades gracefully to 13.38 mm while triangulation collapses to 49.21 mm. SVEgoNet trained on combined manual and automatic annotations reaches 21.92 mm MPJPE, 33% lower than UmeTrack (32.91 mm). On verb classification, SVEgoNet poses yield 54.7% average accuracy versus 50.3% for UmeTrack poses and 60.0% for upper-bound annotations — 91.1% relative performance versus 83.8% — directly supporting the claim that annotation quality drives recognition; per-verb gains reach 13.1% for "position" while "remove" drops 1.8%.

## Contributions

The paper contributes the AssemblyHands benchmark (3.0M images, 490K egocentric, 34 subjects), an automatic multi-view annotation pipeline with 85% error reduction over prior egocentric annotation, a strong single-view egocentric pose baseline, and a pose-based verb classification protocol that quantifies how pose quality translates into action recognition.

## Limitations

The benchmark annotates hands only: object cues such as object pose are absent, and the authors state that annotating the many small assembly parts is a bigger challenge. Coverage of the 30 Hz automatic annotations is a subset of Assembly101, and the authors plan to extend annotation to the full dataset at higher sampling rates and to add object-level annotations such as bounding boxes.
