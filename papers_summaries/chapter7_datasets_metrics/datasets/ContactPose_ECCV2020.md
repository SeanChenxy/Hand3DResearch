# ContactPose: A Dataset of Grasps with Object Contact and Hand Pose

**Authors:** Samarth Brahmbhatt, Chengcheng Tang, Christopher D. Twigg, Charles C. Kemp, James Hays  
**Date:** 2020 (ECCV 2020)  
**Identifier:** [arXiv:2007.09545](https://arxiv.org/abs/2007.09545); DOI `10.1007/978-3-030-58601-0_22`  
**Zotero item:** `WT5JAMT8` ([Zotero](zotero://select/library/items/WT5JAMT8))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

ContactPose is the first dataset of hand-object contact paired with hand pose, object pose, and RGB-D images, addressing the lack of ground truth that has kept contact modeling under-explored. It contains 2306 unique grasps of 25 household objects with 2 functional intents by 50 participants, plus more than 2.9M RGB-D grasp images. The authors analyze hand pose-contact relationships and rigorously evaluate data representations, heuristics, and learning methods for contact prediction, showing that learning with rich hand geometry features outperforms heuristics from the literature.

## Background and Motivation

Contact regions in a grasp are usually occluded from visible-light imaging. Tactile gloves influence natural grasping and miss contact on the object surface, while hand-object model intersection cannot account for soft tissue deformation since rigid hand models do not deform. Prior contact datasets (e.g., ContactDB, tactile gloves) lacked paired hand pose or grasp images, preventing association of contacted areas to hand parts. The paper adopts the thermal imaging approach of ContactDB, observing heat transferred from the warm hand to the object after the grasp, and adds a multi-view RGB-D capture protocol with a computational hand-pose reconstruction algorithm that avoids instrumenting hands.

## Dataset Construction

Participants move each grasped object for 10-15 s in a capture area recorded by 7 OptiTrack Prime 13W cameras (object pose), 3 Kinect v2 RGB-D cameras, and a FLIR Boson 640 thermal camera; the object is then placed on a turntable for thermal scanning. Objects are 3D printed with recessed 3 mm hemispherical tracking markers. Because the hand is rigid relative to the object, noisy OpenPose 2D detections across all frames and cameras are aggregated by a robust (Huber + RANSAC) reprojection optimization into one set of 21 3D joints per hand in the object frame. Contact maps are thermal textures on object meshes normalized to [0, 1], thresholded at 0.4 for binary contact. The dataset also includes MANO fitting data (palm contact plates, 7 hand gestures), providing meshes fit to the 3D joints.

## Evaluation Protocol

The contact modeling task predicts a contact map on a known object given hand pose or RGB grasp images. Object shape is a sampled pointcloud or 64^3 voxel grid; hand pose features range from simple-joints (63-D) through relative-joints, skeleton (40-D), and mesh (23-D MANO-based) representations; image features come from a U-Net-style encoder-decoder in 1-view and 3-view settings. Learners are PointNet++, a VoxNet-style 3D CNN, and an MLP; a calibrated conic distance-field heuristic is the literature baseline. The metric is re-balanced area under the accuracy curve (AuC) over contact bins, on two splits: an object split holding out mug, pan, and wine glass, and a participant split holding out participants 5, 15, 25, 35, and 45.

## Findings and Analysis

Contact analysis shows thumb, index, and middle fingers are most contacted; all three index-finger phalanges have higher contact probability than the pinky tip, and proximal phalanges and palm contribute significant contact. 'Use' grasps average 35.87 cm^2 contact area versus 30.58 cm^2 for 'hand-off', but hand-off grasps are more pose-diverse (32.5% larger intra-cluster distance). Similar hand poses can produce different contact, showing pose alone inadequately represents grasping. In prediction, richer features win: mesh-PointNet++ reaches 81.29% AuC (participant split) and mesh-VoxNet 84.74% (object split), both beating the calibrated heuristic (78.31% / 81.11%), while 3-view image features (78.06%) clearly beat 1-view (72.89%) but trail pose-based methods. Ground-truth quality is validated: thermal contact agrees with Sensel Morph pressure data at 95.4%, and MANO-fit 3D joint error is 7.65 mm (10 pose parameters), lower than HO-3D's 7.7 mm and HANDS 2019's 11.39 mm.

## Contributions

The first dataset pairing ground-truth object contact maps with 3D hand pose, object pose, and multi-view RGB-D images of functional grasps; a markerless multi-view hand pose reconstruction pipeline; analyses of finger-level contact statistics and intent-dependent grasp diversity; and a systematic benchmark of representations, heuristics, and learners for contact prediction.

## Limitations

Objects have plain visual texture because they are 3D printed for consistent thermal properties, limiting RGB-based generalization; grasps are static since in-hand manipulation creates overlapping thermal imprints; the 25 objects are a subset chosen for both 'use' and 'hand-off' applicability; and image-based contact prediction suffers depth ambiguity, missing high-frequency detail compared to hand-pose-based prediction.
