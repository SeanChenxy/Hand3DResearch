# ArtHOI Datasets (CVPR 2025)

> Wang, Zhang, Wang, Li, Zuo (Harbin Institute of Technology, Shanghai Jiao Tong University). *ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions.* CVPR 2025. Zotero Key: `U8CZZKUX`. Project: https://arthoi-reconstruction.github.io.

## Summary
ArtHOI simultaneously proposes an optimization-based framework and two new 4D articulated HOI evaluation datasets (ArtHOI-RGBD + ArtHOI-Wild), specifically designed to evaluate the still-unexplored sub-task of "monocular RGB video → 4D bi-manual + articulated object mesh reconstruction".

## 1. Dataset Purpose
- Addresses the fundamental gap in "monocular 4D hand-articulated-object reconstruction": most existing HOI reconstruction methods assume rigid objects, while existing articulated 4D reconstruction methods generally depend on pre-scanned object templates or multi-view video.
- Tasks: (1) reconstructing 4D hand + articulated object mesh + part-level SE(3) time-series trajectories from monocular RGB video; (2) evaluating the physical plausibility of monocular in wild scenarios.
- Anchors "4D articulated HOI + monocular" as an independent sub-task, forming a "monocular vs multi-view" contrast with ARCTIC (3D articulated + multi-view mocap).
- Provides two complementary evaluation sets: ArtHOI-RGBD (RealSense-shot controlled scenes) and ArtHOI-Wild (challenging videos collected from the Internet).

## 2. Data Composition
- Source: mixed — ArtHOI-RGBD is captured by RealSense RGBD cameras in controlled environments; ArtHOI-Wild comes from Internet videos.
- Viewpoint: ArtHOI-RGBD is first / third-person RGBD; ArtHOI-Wild is cluttered monocular RGB.
- Scale: the specific numbers are not given in the abstract (need to check the paper); at least includes bi-manual + articulated object (scissors, eyeglasses, laptops, etc.) in multiple scenes.
- Object and action: typical articulated everyday items (scissors opening, eyeglass temples, laptop opening, drawer, etc.); a mix of bi-manual and single-hand manipulation.
- ArtHOI-Wild contains naturally captured "uncontrolled" articulated interactions from Internet videos.

## 3. Annotation and Supervision
- Hand: 3D 21 joints, MANO β / θ (from hand-reconstruction model prediction + MLLM-guided alignment optimization).
- Object: canonical mesh (from image-to-3D models such as HunYuan3D) + metric scale + 6-DoF pose (estimated by Adaptive Sampling Refinement) + part-wise SE(3) time-series trajectories.
- Contact: vertex-level contact (inferred by MLLM prompt for frame-wise contact states + fingers, used as constraints for hand-object mesh composition optimization).
- Scene: RGB + metric depth (from monocular depth estimator) + camera intrinsics; ArtHOI-RGBD also provides RGBD.
- No language, no tactile, no robot teleoperation annotation.

## 4. Supported Evaluation
- Benchmark tasks: (1) hand 3D pose (MPJPE / PA-MPJPE / Mesh Error); (2) articulated object 4D mesh (vertex-level distance / F-score); (3) part-wise SE(3) trajectory error; (4) hand-object contact F-score.
- Key metrics: MPJPE, 4D mesh error, part SE(3) error, contact F-score.
- Head-to-head comparison with the existing RSRD dataset — ArtHOI does not need pre-scanned geometry, and its cross-scene robustness is superior.
- Cross-scene split: ArtHOI-RGBD vs ArtHOI-Wild evaluates controlled → wild generalization.

## 5. Why It Matters
- The first to clearly define "monocular 4D articulated HOI" as an independent benchmark direction, jointly releasing framework and dataset.
- ArtHOI-Wild is a rare "in-the-wild articulated HOI" evaluation — existing ARCTIC and others are all controlled-studio recordings.
- The joint release of framework and dataset provides a complete "method + evaluation" baseline for follow-up work.
- Introducing MLLM as the contact-reasoning constraint source is a new paradigm in Ch4 "language reasoning prior" for 4D HOI.
- Promotes "foundation model prior + physical / contact prior" joint optimization as the standard paradigm for articulated HOI 4D reconstruction.
- The core anchor shared by "shape completion / spatial geometry prior" in Ch3 and "language reasoning prior" in Ch4.

## 6. Limitations and Biases
- The specific dataset scale (number of sequences, frames, objects) needs to be confirmed in the paper.
- The object type is dominated by "everyday articulated items" (scissors, eyeglasses, laptops, etc.), not covering industrial articulated equipment.
- No language instruction, no robot teleoperation annotation, which limits direct application of VLA / imitation learning.
- The Wild subset comes from the Internet, and the annotation pipeline is difficult to unify due to video quality and subject identity diversity.
- Ground truth depends on MLLM inference + optimization, and artifacts (e.g., inconsistent depth, MLLM inference failure) will propagate.
- No tactile, no force, and the contact ground-truth accuracy is affected by the upper limit of MLLM capability.
- Released relatively recently (2025), community adoption is still being established.

## 7. Takeaway
ArtHOI Datasets is best for demonstrating the capability of "monocular 4D hand-articulated-object reconstruction + foundation-model prior improvement". **Not suitable** for evaluating rigid-only reconstruction, pure bi-manual tasks, simple grasping with abundant mocap in the studio, or language-conditioned VLA tasks. In this survey, it plays the role of the core anchor for "4D articulated HOI + monocular + foundation prior evaluation" and serves as the unified new benchmark for evaluating "shape completion / spatial geometry prior" in Ch3 and "language reasoning prior" in Ch4.
