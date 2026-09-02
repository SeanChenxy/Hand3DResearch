# DexYCB: A Benchmark for Capturing Hand Grasping of Objects

**Authors:** Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, Dieter Fox  
**Date:** CVPR 2021 (June 2021, per Zotero record)  
**Identifier:** DOI `10.1109/CVPR46437.2021.00893`  
**Zotero item:** `27SZYANI` ([Zotero](zotero://select/library/items/27SZYANI))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

DexYCB is a markerless, multi-view RGB-D dataset of 582K frames (over 8 views) from 1,000 sequences in which 10 subjects grasp 20 physical YCB objects on a tabletop. Ground-truth 3D hand pose (MANO) and 6D object pose are obtained by multi-view optimization over crowdsourced 2D keypoint annotations and fused depth. The paper establishes the first benchmark that jointly evaluates 2D object and keypoint detection, 6D object pose estimation, and 3D hand pose estimation under hand-object interaction, plus a robotics-relevant safe human-to-robot object handover task. Results quantify how much grasping degrades each perception task.

## Background and Motivation

- 3D object pose estimation and 3D hand pose estimation have traditionally been studied separately, yet applications such as learning from human demonstration and human-robot interaction need both simultaneously. In interaction scenarios the difficulty "doubles but multiplies" due to object motion and mutual occlusion, so models trained on hand-only or object-only datasets do not generalize to interactions.
- Accurate real 3D poses are hard to acquire: gloves, magnetic sensors (e.g., FPHA), and marker-based mocap (e.g., ContactPose, GRAB) are intrusive, bias the naturalness of hand motion, and change hand appearance. Synthetic datasets work well for objects but synthesizing realistic grasp poses and natural hand motion remains open, and most synthetic sets offer static images rather than videos.
- The closest prior dataset, HO-3D, is markerless with automatic labeling, but captures fewer objects (10 vs. 20), fewer views (1-5 vs. 8), an order of magnitude fewer frames (78K vs. 582K) and sequences (27 vs. 1,000), labels only the held object, and often contains sequences where the hand is rigidly attached to the object with no finger articulation.

## Dataset Construction

- **Data gap addressed:** a markerless dataset with synchronized RGB-D, dynamic grasping motions, and accurate 3D pose for both hands and all on-table objects.
- **Hardware:** 8 Intel RealSense D415 RGB-D cameras, extrinsically calibrated and temporally synchronized, mounted to cover a tabletop workspace with minimal blind spots; color and depth recorded at 30 fps at 640 x 480 in all 8 views.
- **Collection protocol:** 20 objects from the YCB-Video dataset and 10 subjects. Each trial places one target object plus 2-4 distractor objects on a table; the subject starts relaxed, picks up the target, and holds it in the air (some subjects pretend to hand it over). Each recording lasts 3 seconds. Per target object, 5 trials are recorded (right hand for trials 1-2, left hand for 3-4, randomized for trial 5) with randomized distractor sets and placements, rotating the target across all 20 objects. This yields 100 trials per subject, 1,000 trials total, and 582K frames counted over all views.
- **Annotation:** 2D keypoints are labeled fully by crowdsourced human annotation (VATIC-based tool on Amazon Mechanical Turk) in every view. Hands use 21 predefined joints (3 joints plus 1 tip per finger, plus the wrist), with occluded keypoints marked invisible. For objects, rather than predefined keypoints, annotators select 2 distinctive, trackable landmark points per object per view and mark occlusions.
- **Pose solving:** an optimization over all views. Hands are represented with MANO (pose theta, shape beta; beta pre-calibrated per subject and fixed), implemented as a differentiable PyTorch layer; objects use the standard 6D pose (3x4 matrix) with YCB texture-mapped meshes. The energy E = E_depth + E_kpt + E_reg combines (1) a squared signed-distance term of the world-frame point cloud merged from all views against hand/object meshes (GPU point-parallel), (2) keypoint reprojection terms for the 21 hand joints and the 2 object landmarks (object keypoints are back-projected onto the object surface using an initial pose from PoseCNN, selected manually, then fixed), and (3) l2 regularization on the MANO pose embedding. Adam (learning rate 0.01, 100 iterations per frame) is run with temporal initialization from the previous frame.

## Evaluation Protocol

- **Splits (four setups):** S0 (default): all 10 subjects, 8 views, 20 objects; only sequences are disjoint between train and val/test. S1 (unseen subjects): split by subject, 7/1/2. S2 (unseen views): split by camera, 6/1/1. S3 (unseen grasping): split by grasped object, 15/2/3; test objects are never grasped in training but may appear static on the table so each still has training examples.
- **2D object and keypoint detection:** COCO protocol; 20 object classes plus 1 hand class, with ground truth from rendered segmentation masks and reprojected 3D joints. Reported as AP for bounding box and segmentation (object detection) and AP for 21-joint keypoint detection. Baselines: Mask R-CNN (Detectron2) and SOLOv2, both with ResNet50-FPN backbones.
- **6D object pose estimation:** single-view estimation under the BOP challenge protocol; average recall (AR) over VSD, MSSD, and MSPD error functions. Baselines: PoseCNN (RGB, and with depth post-refinement), DeepIM (RGB and RGB-D), DOPE (RGB, synthetic-only training), PoseRBPF (RGB and RGB-D), and CosyPose (RGB, single-view, initialized from PoseCNN).
- **3D hand pose estimation:** 3D positions of 21 joints from a single image; metrics are mean per-joint position error (MPJPE, mm) and PCK with AUC over the [0, 50 mm] range, each reported in absolute, root-relative, and Procrustes-aligned forms. Baselines: supervised Spurr et al. (HANDS 2019 winner) with ResNet50 and HRNet32 (RGB), and A2J (depth).
- **Safe human-to-robot handover:** given an RGB-D image of a person holding an object, generate diverse SE(3) grasps for a parallel-jaw Franka Panda gripper that take over the object without pinching the hand. A reference set R of 100 successful grasps per object (farthest-point-sampled from a physics-based grasp set) is filtered against ground-truth object and hand meshes. Predictions are scored by coverage (fraction of R matched within 0.05 m translation and 15 degrees rotation by non-colliding predicted grasps) and precision (fraction of predictions covering a successful grasp), yielding precision-coverage curves over a hand-collision threshold in [0, 0.07 m]. Baseline: grasps transformed by an estimated object pose and filtered against the hand point cloud from Mask R-CNN hand segmentation plus depth.

## Findings and Analysis

- **Cross-dataset vs. HO-3D (Spurr et al. + ResNet50/HRNet32, root-relative MPJPE):** a DexYCB-trained model degrades 1.4x-1.9x when tested on HO-3D (18.05 to 31.76 mm for ResNet50), while an HO-3D-trained model degrades 3.4x-3.7x on DexYCB (12.97 to 48.30 mm). Joint training improves HO-3D (18.05 to 15.79 mm) but slightly hurts DexYCB (12.97 to 13.36 mm). The authors conclude DexYCB complements HO-3D better than the reverse.
- **Grasping degrades object pose:** with PoseCNN (RGB) on S0, AR is 52.68% on the full test set, 56.53% on static objects, but 41.65% on grasped objects only.
- **6D pose on grasped objects:** AR drops under unseen subjects (S1: 38.26% vs. 41.65% on S0) and unseen grasping (S3, e.g., gelatin box 33.07% vs. 46.62% on S0), with small objects hit hardest; unseen views (S2) slightly improve AR (45.18% vs. 41.65%), suggesting the 8-view coverage is dense enough for cross-view transfer. On S1, depth helps (PoseCNN 38.26% to 43.27% with depth refinement), refinement helps more (DeepIM RGB-D 38.26% to 57.54%), and CosyPose is the best RGB method (57.43%).
- **2D detection:** Mask R-CNN and SOLOv2 are close (S0 mAP bbox 75.76 vs. 75.13; segm 69.58 vs. 71.56). Hand AP is lower than object AP (Mask R-CNN S0: 71.85 bbox for hand vs. 75.76 mAP), indicating hands are harder to detect; S1 (unseen subjects) lowers mAP. Keypoint detection AP (Mask R-CNN): 36.42 (S0), 26.85 (S1), 32.90 (S2), 35.18 (S3).
- **3D hand pose:** absolute pose from RGB alone is hard (Spurr et al. + ResNet50: 53.92 mm absolute MPJPE on S0; HRNet32 only marginally better at 52.26 mm). Errors rise for unseen subjects (HRNet32: 70.10 mm on S1) and, unlike objects, also rise for unseen views (80.63 mm on S2, the worst setup) because subjects always face the same direction from a fixed position, constraining per-view hand poses and hurting cross-view generalization. Depth-based A2J is far better on absolute error (27.53 mm on S0) but worse on articulation (Procrustes 12.07 mm vs. 6.83 mm for HRNet32).
- **Handover:** better object pose estimation yields better grasp generation (precision-coverage curves). Most failure grasps stem from inaccurate object pose; some collide with the hand when Mask R-CNN misses partially occluded hands, which the authors suggest could be addressed by model-based full-hand prediction.

## Contributions

- A new markerless dataset capturing full grasping processes (approach, finger opening, contact, stable hold) in a tabletop workspace, with human-verified 3D pose for hands and all on-table objects; empirically shown to be larger and more diverse than HO-3D.
- The first benchmark allowing joint evaluation of 2D object and keypoint detection, 6D object pose estimation, and 3D hand pose estimation on the same interaction data, with four generalization setups.
- A new robotics-relevant safe human-to-robot object handover task that demonstrates the importance of joint hand and object pose estimation for downstream robot grasping.

## Limitations

The paper does not include a dedicated limitations section; the following are evidenced by its own reported results and protocol.

- The captured behavior is narrow: short (3-second) single-object pick-up-and-hold grasps of 20 rigid YCB objects by one hand at a time, in a fixed tabletop scene; no bimanual, articulated-object, or long-horizon manipulation.
- Generalization remains poor in the reported baselines: unseen subjects, unseen grasped objects, and (for hand pose) unseen views all cause large error increases; absolute 3D hand pose from RGB alone is far from solved (about 52-54 mm MPJPE on S0).
- Object pose estimation degrades substantially on grasped objects (41.65% vs. 56.53% AR for static objects with PoseCNN), and hand detection fails under occlusion in the handover task, limiting end-to-end reliability.
- Ground truth depends on a costly pipeline of crowdsourced 2D keypoint annotation in all 8 views plus per-frame multi-view optimization, in contrast to fully automatic labeling pipelines such as HO-3D's.
