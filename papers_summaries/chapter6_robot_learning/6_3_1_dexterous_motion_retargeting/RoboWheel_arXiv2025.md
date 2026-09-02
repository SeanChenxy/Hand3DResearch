# RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning

**Authors:** Yuhong Zhang, Zihan Gao, Shengpeng Li, Ling-Hao Chen, Kaisheng Liu, Runqing Cheng, Xiao Lin, Junjia Liu, Zhuoheng Li, Jingyi Feng, Ziyan He, Jintian Lin, Zheyan Huang, Zhifang Liu, Haoqian Wang  
**Date:** 2025-12-02  
**Identifier:** [arXiv:2512.02729](https://arxiv.org/abs/2512.02729); DOI `10.48550/arXiv.2512.02729`  
**Zotero item:** `T6KVH9YL` ([Zotero](zotero://select/library/items/T6KVH9YL))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
RoboWheel is a data engine that converts monocular RGB/RGB-D hand-object interaction (HOI) videos into training-ready supervision for robots with different morphologies. It reconstructs hand and object motion in a world frame, enforces physical plausibility through SDF-based collision optimization plus a residual RL policy, retargets the resulting trajectories to parallel-gripper arms, dexterous hands, and humanoids, and amplifies coverage with Isaac Sim domain randomization (five 6/7-DoF arms, object retrieval, textures, clutter, mirroring). The produced HORA dataset of roughly 150K trajectories yields policies comparable to teleoperation-trained ones—e.g., Pi0 pre-trained on 5K HORA trajectories reaches 76.3% average success on easy and 58.8% on hard real tasks—providing, to the authors' knowledge, the first quantitative evidence that HOI data can supervise robotic learning.

## Background and Problem
Teleoperation and studio motion capture demand specialized hardware and curation, limiting the diversity and transferability of robot demonstrations, while abundant HOI videos go unused because of reconstruction noise, interpenetration, unsmooth trajectories, and embodiment mismatch. The paper requires a pipeline that (i) acquires physically plausible robot-object trajectories at scale from real-world operational spaces, (ii) retargets them flexibly across embodiments while preserving interaction semantics, and (iii) supports scalable composition of augmentations—requirements that teleoperation and purely synthetic simulation each fail to meet jointly.

## Method
Reconstruction first classifies each clip as hand-only or whole-body, estimating MANO/SMPL-H states accordingly; the object mask and depth yield a metrically scaled mesh from a multi-view 3D generator, and a correspondence-driven tracker recovers the object pose stream, with camera intrinsics and camera-to-world transforms from visual SLAM placing everything in one world frame aligned to a canonical action space. Physical plausibility is then enforced in stages: TSDF penalties remove palm and hand-object interpenetration, after which a residual RL policy refines hand-object relative poses with rewards combining geometry tracking, dynamics smoothness, and contact force, inspired by ManipTrans. Retargeting maps hand poses to gripper end-effector poses using either a palm frame from MCP joints (whole-hand contact) or an index-thumb chord (finger-only contact), chosen by a kNN gesture classifier, with gripper open/close decided from CoTracker keypoint displacement; dexterous-hand and humanoid retargeting use kinematic similarity and IK over SMPL-H estimates. In Isaac Sim, five arms (UR5/UR5e, Franka Panda, KUKA iiwa 7, Kinova Gen3, Sawyer) replay trajectories via cuRobo GPU inverse kinematics, and augmentations include object retrieval by fused Chamfer/AABB/semantic similarity, trajectory re-anchoring, textures, clutter, and hand mirroring; Qwen2.5-VL automatically filters failed replays. HORA aggregates a tactile-instrumented multi-view mocap subset (RealSense D455 rig, glove with 29 magnetic encoders and 16 tactile sensors), RGB(D) recordings, and public HOI corpora.

## Contributions
- A physically plausible monocular HOI reconstruction and cross-embodiment retargeting framework outputting executable operational- and joint-space actions.
- A simulation-augmented data flywheel with domain randomization validated on mainstream VLA and imitation learning architectures.
- HORA, a roughly 150K-trajectory multimodal dataset (mocap, recordings, public HOI) including tactile signals in the mocap subset.

## Experimental Setup
HOI reconstruction is evaluated on HO-Cap against HORT, HOLD, and DiffHOI. Downstream evaluation benchmarks eight real household tasks split into Easy/Hard groups with ACT, Diffusion Policy, RDT, and Pi0 under three regimes: fine-tuning on 10 teleoperation demos versus 10 HORA trajectories, plus 5K-trajectory HORA pre-training for RDT and Pi0. Robustness tests compare augmented versus unaugmented HORA under unseen objects, clutter, and backgrounds, and retargeting is validated by direct replay on a UR5 with a two-finger gripper against GAT-Grasp and YOTO.

## Results
RoboWheel's reconstruction attains the best scores on all metrics (e.g., Chamfer 5.1 cm, F10 89.1%, hand jitter 0.92 cm/s2, and relative-pose consistency 0.26 cm/1.9 deg versus 2.44-4.51 cm for baselines). With equal episode counts, HORA-trained policies match or approach teleoperation (e.g., Pi0 easy 58.8% versus 68.8%), while 5K pre-training lifts averages to 75.0-76.3% easy and 47.5-58.8% hard. Augmented HORA raises unseen-background success from 1.5 to 4.0 of 10 trials with RDT. Direct replay reaches a 91.7% macro average versus 50.0% for GAT-Grasp and 66.7% for YOTO.

## Limitations
The authors state that real-world cross-embodiment experiments, particularly for dexterous hands and humanoids, remain limited, and expanding cross-domain validation in both real and simulated settings is left to future work.
