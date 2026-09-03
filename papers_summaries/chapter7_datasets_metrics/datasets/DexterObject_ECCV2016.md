# Real-time Joint Tracking of a Hand Manipulating an Object from RGB-D Input (Dexter+Object)

**Authors:** Srinath Sridhar, Franziska Mueller, Michael Zöllhöfer, Dan Casas (with Antti Oulasvirta, Christian Theobalt)
**Date:** 2016 (ECCV 2016)
**Identifier:** [arXiv:1610.04889](https://arxiv.org/abs/1610.04889)
**Zotero item:** No record found in the Zotero library; identity verified against arXiv metadata.
**Evidence status:** No Zotero record; verified against full-text PDF extraction (arXiv 1610.04889).

## Summary
This paper addresses the lack of reliable evaluation resources for joint hand-object pose tracking: at the time, no public dataset provided ground truth for both hand (fingertip) and object pose, making quantitative evaluation of object tracking impossible. The authors propose a real-time method that simultaneously tracks hand articulation and rigid object motion from a single commodity RGB-D camera, and they introduce a new annotated hand-object dataset for benchmarking. The evaluation combines the new dataset with public datasets (IJCV, Dexter, and an in-hand scanning dataset), reporting average 3D Euclidean errors. Their tracker achieves 15.7 mm combined hand-object error on the new benchmark and runs at 25-30 Hz, over 60 times faster than the comparable multi-camera method of Tagliasacchi et al. while reaching comparable fingertip accuracy.

## Background and Motivation
Real-time joint tracking of hands manipulating objects is more challenging than tracking either alone because of mutual occlusions, fast motions, and the difficulty of disambiguating hand from object. Prior approaches resorted to multi-camera setups to mitigate occlusion, or used expensive segmentation and optimization steps that preclude real-time performance. Discriminative one-shot hand trackers suffer from temporal instability, and methods that handled hand-object interaction either did not track the object explicitly, ran far below real-time, or required controlled multi-camera rigs. The authors also note an evaluation gap: existing hand-object datasets provided ground truth joint annotations but no object pose annotations (and often none at all), so object tracking accuracy could not be measured.

## Dataset Construction
The paper introduces what the authors state is, to their knowledge, the first dataset containing ground truth for both fingertip positions and object pose.
- Source and sensor: RGB-D sequences captured with commodity depth sensors (the experiments cover Creative Senz3D, Intel RealSense F200, and Primesense Carmine); depth/color rescaled to 320x240 and 640x480 at 30 Hz.
- Scale: 6 sequences; in total 3014 frames with ground truth annotations.
- Objects: a cuboid in 2 different sizes, manipulated in different hand-object configurations and grasps.
- Sequences: Rigid, Rotate, Occlusion, Grasp1, Grasp2, and Pinch, covering varied grasps and deliberate occlusion of the hand by the object.
- Annotations: fingertip positions were obtained by manually annotating pixels on the depth image for 5 fingertips; 3 cuboid corners were likewise annotated to define object pose. Occluded fingertips (or cuboid corners) are excluded per frame.

## Evaluation Protocol
- Task: model-based, real-time tracking of hand articulation (26 DOF skeleton) and rigid object motion (6 DOF) from single-view depth plus color; output is per-frame hand pose and object pose.
- Method sketch: hand-object segmentation and hand part classification by multi-layer random forests guide a 3D articulated Gaussian mixture alignment energy with novel contact-point and occlusion regularizers, optimized with a two-proposal (winner-takes-all) strategy.
- Metrics: average 3D Euclidean distance (mm) between estimated and ground-truth positions, computed separately for fingertips, object (annotated cuboid corners), and combined; occluded keypoints are excluded on a per-frame basis.
- Baselines/comparisons: the slower generative method of Tagliasacchi et al. (IJCV dataset, 2D joint error in pixels), the in-hand scanning dataset of Tsoli et al. (qualitative), and an ablation against the 2.5D Gaussian mixture formulation of Qian et al. on the Dexter hand-only dataset; ablative analysis removes viewpoint selection, semantic alignment, occlusion handling, and contact terms.

## Findings and Analysis
- On the new benchmark, the method attains an overall average error of 15.7 mm (fingertips 15.6 mm, object 16.2 mm); per-sequence combined errors range from 10.9 mm (Pinch, best) to 18.0 mm (Rotate, worst), all below 20 mm with standard deviations under 12 mm. Hand and object errors are of the same order of magnitude.
- Tracking is consistent: almost all frames are tracked with less than 30 mm error; Pinch has almost all frames below 20 mm.
- On 5 rigid-object sequences of the IJCV dataset, fingertip pixel error is 8.6 px, comparable to the slower method of Tagliasacchi et al. (difference of only 2 px, within annotation and sensor noise uncertainty) while running over 60 times faster; the object is not annotated in that dataset.
- Hand-only ablation on Dexter: the 3D Gaussian mixture formulation reaches 17.2 mm average fingertip error versus 19.6 mm reported for the 2.5D formulation of Qian et al. (up to 5 mm improvement on 2 sequences), showing the benefit of the continuous 3D energy.
- Ablations show that removing viewpoint selection, semantic alignment, occlusion handling, or the contact term degrades accuracy and robustness; the data term alone produces large errors.
- Runtime is 25-30 Hz (about 4 ms preprocessing, 4 ms classification, 2 ms clustering, 20-30 ms pose optimization) on a desktop CPU/GPU.

## Contributions
- The first real-time method for simultaneous hand-object tracking from a single commodity RGB-D camera.
- 3D articulated Gaussian mixture alignment tailored to hand-object tracking, with novel analytic contact-point and occlusion objective terms motivated by grasp physics.
- A multi-layer classification architecture for hand-object segmentation and hand part labeling with viewpoint selection.
- A new, publicly released benchmark dataset with per-frame ground truth for both 5 fingertip positions and 3 cuboid corners (3014 annotated frames over 6 sequences), enabling joint hand-object accuracy evaluation.

## Limitations
The authors state that situations where a high fraction of the hand is occluded for a long period remain challenging because classification degrades under such occlusion. Misalignments appear when the occlusion heuristic assumption is violated, i.e., occluded parts do not move rigidly; they suggest learning grasp/interaction priors for occluded regions. Very fast motions are another error source, and the authors note that increasing object complexity (shape and color) affects runtime. The benchmark itself is small (6 sequences, one primitive object type, cuboid only) and annotations are sparse (fingertips plus 3 corners rather than full 21-joint hands or full 6D object pose meshes), which the paper does not explicitly discuss but which is evident from the construction.
