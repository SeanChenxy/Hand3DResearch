# HERMES: Human-to-Robot Embodied Learning from Multi-Source Motion Data for Mobile Dexterous Manipulation

**Authors:** Zhecheng Yuan, Tianming Wei, Langzhe Gu, Pu Hua, Tianhai Liang, Yuanpei Chen, Huazhe Xu  
**Date:** 2025-08-31  
**Identifier:** [arXiv:2508.20085](https://arxiv.org/abs/2508.20085)  
**Zotero item:** `WYIU4ECE` ([Zotero](zotero://select/library/items/WYIU4ECE))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
HERMES is a human-to-robot learning framework for mobile bimanual dexterous manipulation that converts heterogeneous human motion—one-shot teleoperation, motion capture, or poses extracted from raw videos—into physically plausible robot behaviors via a unified goal-conditioned reinforcement learning formulation. It distills state-based expert policies into depth-image student policies with DAgger plus a general depth augmentation and hybrid sim2real control, and couples the manipulation policy with the ViNT navigation foundation model refined by closed-loop Perspective-n-Point localization. The framework achieves an average 67.8% success across six real-world bimanual dexterous tasks (+54.5% over a raw-depth baseline) and precise autonomous navigation in indoor and outdoor scenes.

## Background and Problem
Human motion data is abundant, but existing video-to-robot approaches target parallel grippers or perform kinematic retargeting that ignores hand-object interactions and cannot produce feasible actions for high-dimensional multi-fingered hands. RL-based translation methods draw on limited data sources and often lack real-world transfer, while prior sim2real pipelines depend on explicit object state extraction, confining policies to fixed setups. The paper's task is: given a single human motion reference from any source, train a bimanual dexterous manipulation policy that transfers zero-shot to a real mobile robot operating autonomously in unstructured environments.

## Method
HERMES follows a four-stage pipeline. First, one-shot human motion is collected from simulation teleoperation (Apple Vision Pro at 75 Hz), the OakInk2 mocap dataset, or raw videos processed with WiLoR hand estimation, FoundationPose object tracking, and ARCode mesh scanning; trajectories are augmented by randomizing object poses. Second, DexPilot retargeting initializes robot references, and a state-based RL teacher is trained in MuJoCo (DrM) or GPU-parallel MJX (PPO) using a shared reward set—object-centric distance chain gated on hand-object contact, object trajectory tracking, and a power penalty—with residual actions refining coarse trajectory-following for arms and fully learned hand actions. Third, DAgger distills the expert into a 140x140 depth-image student (partial ResNet-18 encoder) with clipping, noise, blur, missing-pixel and NYU-depth mixup augmentation, and hybrid control runs policy actions through the simulator before mapping joints to the real robot. Fourth, ViNT provides image-goal navigation while closed-loop PnP localization (Efficient LoFTR matching, RANSAC PnP, PID control) aligns the robot to the manipulation pose. The platform pairs a Galaxea X1 base, two 6-DoF A1 arms, and two 6-DoF OYMotion hands.

## Contributions
- A unified RL formulation transforming multi-source one-shot human motions into feasible bimanual dexterous behaviors with reusable rewards.
- An end-to-end depth-based sim2real transfer method combining DAgger distillation, general depth augmentation, and hybrid sim2real control.
- A navigation-to-manipulation bridge that augments a navigation foundation model with closed-loop PnP localization for autonomous mobile dexterous manipulation.

## Experimental Setup
Training covers seven tasks (e.g., Bottle Handover, Clean Table, Scan Bottle, Place Drawer, Pour Teapot, Clean Plate, Putoff Burner) plus Flower Vase, evaluated over 3 seeds; baselines include re-implemented ObjDex rewards, kinematic retargeting, and trajectory replay. Real-world manipulation is evaluated over 15 trials on six tasks against a raw-depth sim2real baseline. Navigation is tested in two indoor and one outdoor scenario, and full mobile manipulation over 10 runs per task on the Galaxea platform.

## Results
HERMES outperforms ObjDex on all tasks and is the only method succeeding on multi-object interactions, learning within 3M training steps. Kinematic retargeting alone scores 0% on both video-derived (HERMES 78.1%) and mocap-derived tasks (HERMES 88.9%); direct replay reaches only 52.2% versus 91.9% for Bottle Handover and 49.9% versus 72.2% for Place Drawer. In the real world, HERMES averages 67.8% success (task range 60.0-73.3%) versus 13.3% for raw depth, and beats a Depth-Anything-based pipeline (e.g., 66.7% versus 40.0% on Bottle Handover). Closed-loop PnP achieves 1.3-3.2 cm and 0.57-1.79 degree localization errors versus 7.3-18 cm for ViNT alone, and works where RTAB-MAP fails; it lifts mobile manipulation success by 54.0% over ViNT-only deployment.

## Limitations
The authors state the tasks are quasi-static, so the hybrid control scheme would require complex system identification for highly dynamic, velocity-dependent tasks; physics collision parameters are manually tuned and objects approximated with primitives; and simulation-hardware assembly and calibration mismatches persist, reducing success rates despite closed-loop mitigation.
