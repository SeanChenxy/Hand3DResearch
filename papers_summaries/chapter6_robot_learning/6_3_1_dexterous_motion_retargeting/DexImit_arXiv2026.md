# DexImit: Learning Bimanual Dexterous Manipulation from Monocular Human Videos

**Authors:** Juncheng Mu, Sizhe Yang, Yiming Bao, Hojin Bae, Tianming Wei, Linning Xu, Boyi Li, Huazhe Xu, Jiangmiao Pang  
**Date:** 2026-02-10  
**Identifier:** [arXiv:2602.10105](https://arxiv.org/abs/2602.10105)  
**Zotero item:** `VYKZ9HIF` ([Zotero](zotero://select/library/items/VYKZ9HIF))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
DexImit addresses data scarcity in bimanual dexterous manipulation by converting monocular human manipulation videos—in-the-wild or generated—into physically plausible robot training data requiring no depth, camera intrinsics, or teleoperation. Its four-stage pipeline reconstructs near-metric 4D hand-object interactions, decomposes tasks with action-centric bimanual scheduling, synthesizes force-closure grasps with motion planning, and applies comprehensive augmentation. Policies trained on the generated data achieve near-perfect success on short-horizon simulation tasks, beat RigVid and DexMan baselines, and deploy zero-shot on a real dual-UR5e + XHand platform.

## Background and Problem
Collecting real-world data for bimanual dexterous hands is expensive because teleoperation is difficult and hardware is costly, so available datasets lag far behind those for simple grippers. Human videos are abundant and even text-to-video models can synthesize them, but direct pretraining on human hands suffers a severe embodiment gap, and video-to-trajectory methods typically need absolute depth, tolerate little reconstruction noise under reinforcement learning, and fail on fast motions, occlusions, or long-horizon interactions. The paper's task is: given only an RGB human video, produce training-ready bimanual dexterous robot demonstrations and policies that transfer zero-shot to real robots.

## Method
DexImit runs four stages. (1) Reconstruction: Qwen3-VL and Grounded Sam2 segment hands, objects, and table; SpatialTracker v2 depth is scaled to near-metric by aligning a Wilor hand mesh with the hand point cloud (the human hand size supplies the metric prior); SAM3D generates object meshes; FoundationPose++ tracks 6D object poses; trajectories are mapped from arbitrary camera views into a shared world frame defined by the table normal and the inter-hand direction. (2) Scheduling: a VLM decomposes the video into tasks with pregrasp/grasp/motion/release subactions, and an Action-Centric Scheduling algorithm assigns them across embodiments over arbitrary horizons. (3) Action generation: force-closure-based grasp synthesis (BODex-style optimization, candidates ranked by distance to the demonstrated human pose and checked by simulated stability rollouts) plus keyframe-based rigid-body motion planning. (4) Augmentation: object pose and scale (0.8–1.2, reusing original grasps), camera pose, and point-cloud noise randomization, followed by training a 3D Diffusion Policy (DP3).

## Contributions
- An automated pipeline that turns monocular RGB human videos into physically plausible bimanual dexterous manipulation data without additional sensors or annotations.
- A comprehensive augmentation system (object pose/scale, camera pose, observation noise) enabling zero-shot real-world policy deployment.
- Experiments showing high-fidelity data generation for long-horizon, tool-using, and fine-grained tasks, alleviating data scarcity.

## Experimental Setup
Reconstruction is evaluated on 100 short-horizon tasks across depth models (VGGT, Trace-Anything, Depth-Anything v3, SpatialTracker v2) and pose methods (RANSAC, Color-PCR, FoundationPose++). Data quality is compared against re-implemented RigVid and DexMan on six simulation tasks (Put Cup, Grapefruit, Fruits, Pour, Pot, Stack Cups) with DP3 policies trained on 100 augmented demonstrations per task. Real-world deployment uses two UR5e arms with XHands and an Azure Kinect, testing four meta-tasks: Place Apple, Place Potato&Pepper, Place Pot, and Pour Water.

## Results
In reconstruction, SpatialTracker v2 + FoundationPose++ reaches 82% success versus 11–45% for correspondence/registration alternatives. In simulation, DexImit scores 100% on Put Cup, Grapefruit, Fruits, and Pour, 78% on Pot, and 52% on Stack Cups, while RigVid fails on bimanual tasks and DexMan only succeeds on short-horizon ones. Zero-shot real-world success stays high across all four meta-tasks; removing scale augmentation, regenerating grasps per scale, or removing point-cloud noise each degrades success. Generation takes roughly four minutes per video.

## Limitations
The sequential pipeline propagates errors, sometimes producing unusable data and occasionally needing manual correction for long videos; complex in-hand manipulation cannot be handled due to occlusion; rigid-object assumptions exclude deformable and articulated objects; and the framework is restricted to tabletop settings without mobile manipulation.
