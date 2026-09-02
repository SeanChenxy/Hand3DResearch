# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning

**Authors:** Kailin Li, Puhao Li, Tengyu Liu, Yuyang Li, Siyuan Huang  
**Date:** 2025-03-27 (CVPR 2025)  
**Identifier:** [arXiv:2503.21860](https://arxiv.org/abs/2503.21860)  
**Zotero item:** `P43M38E2` ([Zotero](zotero://select/library/items/P43M38E2))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
ManipTrans is a two-stage framework that transfers human bimanual manipulation skills to dexterous robotic hands in simulation. It first pre-trains a generalist trajectory imitator on hand-only motion data, then fine-tunes a residual module that refines the coarse imitation under interaction constraints (object tracking, contact forces, bimanual coordination), avoiding task-specific reward engineering. It surpasses prior methods in success rate, fidelity, and efficiency—training a new 60-frame trajectory in roughly 15 minutes versus about 40 hours of optimization for QuasiSim—and is used to build DEXMANIPNET, a 3.3K-episode dataset covering previously unexplored tasks such as pen capping and bottle unscrewing, with real-world replay on dual Realman arms.

## Background and Problem
Data-driven embodied AI requires precise, large-scale, human-like manipulation sequences that conventional RL (needing task-specific rewards) and teleoperation (labor-intensive, embodiment-specific, lacking haptics) cannot supply efficiently. Transferring human MoCap trajectories to robot hands is non-trivial: morphological differences make direct pose retargeting suboptimal, MoCap error accumulates critically in high-precision tasks, and bimanual manipulation introduces a high-dimensional action space. As a result, prior work mostly stops at single-hand grasp-and-lift, leaving complex bimanual activities like unscrewing a bottle or capping a pen largely unexplored. The task is formulated over a bimanual MDP where two dexterous hands replicate reference human hand and object trajectories under PD joint targets plus 6-DoF wrist force actions.

## Method
Stage one trains a hand trajectory imitation policy with PPO using hand-only datasets (plus mirrored and interpolated synthetic data), with wrist SE(3) tracking, weighted keypoint finger imitation (emphasizing thumb, index, middle), and a joint-power smoothness reward; reference state initialization, early termination, and a curriculum on the finger-error threshold handle noise. Stage two expands the state with object pose/velocities, mass, gravity, BPS shape encoding, hand-object distances, and simulated contact forces, then learns residual actions added element-wise to the imitation actions (zero-mean Gaussian initialization with warm-up prevents collapse). Rewards add object-trajectory following and a contact-force term gated on MoCap hand-object proximity. A physics-relaxation curriculum (zero gravity, high friction, relaxed thresholds, gradually restored) avoids local minima. All experiments run in Isaac Gym with 4096 parallel environments on a single RTX 4090 PC. DEXMANIPNET is generated from FAVOR and OakInk-V2 using a simulated 12-DoF Inspire Hand configuration.

## Contributions
- A two-stage transfer framework decoupling hand motion imitation from physics-based interaction via residual learning, enabling precise tracking of hand and object references without task-specific rewards.
- DEXMANIPNET: 3.3K episodes over 61 tasks and 1.2K objects (1.34 million frames, ~600 bimanual sequences), extensible for downstream policy training.
- Demonstration of SOTA accuracy and efficiency, cross-embodiment generalization, and real-world bimanual deployment.

## Experimental Setup
Evaluation uses ~80 filtered OakInk-V2 validation episodes (4-20 s, 60 fps) with per-frame rotation/translation/joint/fingertip errors and a strict success criterion (30 deg, 3 cm, 8 cm, 6 cm; both hands must succeed). Baselines: retarget-only, RL-only from scratch, retarget plus residual learning, and qualitative comparison with QuasiSim. Cross-embodiment tests cover Shadow, articulated MANO, Inspire, and Allegro hands.

## Results
ManipTrans reaches 58.1%/39.5% success on single/bimanual tasks versus 47.8%/13.9% for retarget-plus-residual, 34.3%/12.1% for RL-only, and 4.6%/0.0% for retarget-only, with lower errors (e.g., Et 0.49 cm). New trajectories train in ~15 minutes versus QuasiSim's ~40 hours; the approach tolerates Gaussian noise up to 1.5 cm. Tactile contact information as reward and observation improves convergence, and removing physics relaxation can prevent convergence entirely. Real-world replay on two 7-DoF Realman arms with tactile Inspire hands executes tasks like toothpaste opening. On DEXMANIPNET bottle rearrangement, diffusion policies reach at most 18.44% success, highlighting remaining difficulty.

## Limitations
Some MoCap sequences still fail to transfer, attributed to excessive noise in interaction poses and insufficiently accurate object models for simulation, particularly for articulated objects.
