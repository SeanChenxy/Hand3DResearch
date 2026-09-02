# DexMV: Imitation Learning for Dexterous Manipulation from Human Videos

**Authors:** Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, Xiaolong Wang  
**Date:** 2022-07-06  
**Identifier:** [arXiv:2108.05877](https://arxiv.org/abs/2108.05877)  
**Zotero item:** `ZUJVZIFH` ([Zotero](zotero://select/library/items/ZUJVZIFH))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
DexMV provides a paired platform and pipeline for learning complex dexterous manipulation from human videos: a computer-vision rig records people performing tasks that a multi-finger Adroit hand also performs in MuJoCo simulation, and a demonstration translation module converts estimated 3D hand-object poses into robot demonstrations. The translation combines Task-Space-Vector-based motion retargeting with minimum-jerk action estimation, and the resulting demonstrations are benchmarked with several imitation learning algorithms. Demonstrations improve success rates by a large margin over reinforcement learning alone, even enabling policies that solve tasks RL cannot solve and that generalize to unseen objects.

## Background and Problem
Although 3D hand-object pose estimation has advanced rapidly, robots still struggle with multi-finger dexterous manipulation: the 30-DoF Adroit hand with tendon-based actuation makes pure reinforcement learning sample-hungry and prone to unnatural behavior, and prior imitation work either uses costly VR/mocap demonstrations in small amounts (25 per task) or targets only simple gripper tasks. The paper's task is to translate human manipulation videos into robot state-action demonstrations and study how they augment RL for goal-conditioned relocation, pouring, and object placement.

## Method
The platform pairs a capture rig (a 35 cubic-inch frame with two RealSense D435 cameras) with a MuJoCo simulation system using the Adroit hand on three task types: Relocate (five YCB objects, goal-conditioned), Pour (pouring particles from a mug), and Place Inside (placing a banana into a mug). The pipeline first estimates object 6D poses with PVN3D and optimizes MANO hand parameters against 2D reprojection and depth across two cameras, with low-pass filtering for temporal consistency. Demonstration translation then (i) retargets human motion to robot joint angles by optimizing the matching of fingertip-to-palm and fingertip-to-middle-phalanx Task Space Vectors between the 51-DoF human hand model and the 30-DoF robot hand, and (ii) recovers actions by fitting a minimum-jerk joint trajectory and applying inverse dynamics, before resampling to the 120 Hz simulation rate. The demonstrations are used with TRPO-based RL through state-action methods (GAIL+, DAPG) and a state-only method (SOIL).

## Contributions
- The DexMV platform: paired computer-vision and simulation systems with multiple complex dexterous manipulation tasks, collecting roughly 100 demonstrations per hour.
- A demonstration translation method converting human hand-object pose sequences into executable robot demonstrations via TSV-based retargeting and minimum-jerk action estimation.
- A benchmark showing demonstrations greatly improve dexterous manipulation performance and generalization over RL alone.

## Experimental Setup
Experiments use MuJoCo with the Adroit hand and YCB objects, 100 recorded demonstrations per task, and evaluation over 100 trials with three random seeds. Baselines are TRPO RL alone versus SOIL, GAIL+, and DAPG imitation learning. Ablations cover retargeting variants, number of demonstrations (10/50/100), object size and friction changes, and hand pose estimation settings measured by MPJPE on DexYCB.

## Results
On Relocate, imitation beats RL on all five objects; DAPG reaches 1.00 on tomato soup can, clamp, and mug, and RL fails entirely on mustard bottle and sugar box (0.06 and 0.00 versus DAPG's 0.93 and SOIL's 0.67). More demonstrations help: SOIL with 100 demonstrations reaches 1.00 success at 800 iterations versus 0.36 with 10 and 0.13 for RL. Better hand pose estimation correlates with better policies (2-camera with post-processing, MPJPE 32.5, yields 93.3% success). On Pour, DAPG pours 27.2% of particles versus 1.3% for RL; on Place Inside, DAPG scores 31.3 versus 3.2 for RL. DAPG also generalizes best to ShapeNet instances (e.g., 83.6% relocating novel cans, 47.2% on the novel camera category).

## Limitations
The paper notes that retargeting from the higher-dimensional human hand to the robot hand inevitably loses information, and that state-only imitation learns less accurate actions than analytically computed ones, particularly struggling to fit an inverse model for water particles in Pour. Policies also depend on pose-estimation quality: lower-accuracy single-camera estimation degrades downstream success, and demonstration translation relies on a calibrated multi-camera rig and YCB-trained object pose estimation.
