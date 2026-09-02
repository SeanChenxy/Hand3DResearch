# Human2Robot: Learning Robot Actions from Paired Human-Robot Videos

**Authors:** Sicheng Xie, Haidong Cao, Zejia Weng, Zhen Xing, Haoran Chen, Shiwei Shen, Jiaqi Leng, Zuxuan Wu, Yu-Gang Jiang  
**Date:** 2025-02-23  
**Identifier:** [arXiv:2502.16587](https://arxiv.org/abs/2502.16587); DOI `10.48550/arXiv.2502.16587`  
**Zotero item:** `N3BM47W7` ([Zotero](zotero://select/library/items/N3BM47W7))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Human2Robot turns precisely synchronized human-robot video pairs into a human-video-conditioned manipulation policy. The authors first build H&R, a 2,600-episode third-person dataset of frame-aligned human-hand and robot-arm videos collected with a coordinate-calibrated VR teleoperation system, then train a Stable Diffusion-based Video Prediction Model (VPM) that hallucinates the robot's video from a human demonstration, and finally decode actions from the VPM's one-step denoising features with a Diffusion Policy head. The resulting policy reaches a 95% average success rate on seen pick-and-place style tasks and generalizes one-shot to novel positions, appearances, instances, backgrounds, task combinations, and even brand-new writing tasks, where self-supervised and language-conditioned baselines collapse to 0%.

## Background and Problem
Learning from human demonstrations is attractive for scaling robot skills, but existing pipelines rely on coarsely aligned human-robot video pairs and self-supervised or contrastive objectives, so models capture only global or task-level features rather than fine-grained frame-level dynamics. The authors diagnose a vicious circle: the lack of densely paired datasets produces methods that cannot exploit fine-grained supervision, and the dominance of such methods discourages collecting those datasets. The paper therefore asks how to (i) collect truly synchronized human-robot video pairs at scale and (ii) convert that fine-grained correspondence into a policy that imitates unseen tasks from a single human video.

## Method
The data engine is a VR teleoperation rig (Meta Quest 3 headset, 7-DoF xArm, two Intel RealSense D435 cameras at 240x424 and 30 Hz, built on OpenTeach) in which three anchor points per side establish a shared, scale-consistent coordinate system between the operator's hand and the robot arm, yielding perfectly aligned third-person videos from identical viewpoints; the residual embodiment gap (e.g., for screwing) is left as future work, so collection focuses on pick-and-place style tasks. The H&R dataset stores 2,600 episodes (300-600 frames each) in RT-X format with paired videos, robot states, joint velocities, human hand transforms and keypoints, and retargeted robot actions. The model has two stages: a VPM built from a Stable Diffusion-initialized Spatial UNet plus a Spatial-Temporal UNet and a convolutional Behavior Extractor first learns image-level and then video-level generation of robot footage conditioned on human videos; the frozen VPM is then used as a visual encoder whose first-upsampling-layer features after a single denoising step condition a Video Former plus Diffusion Policy action decoder. A KNN retrieval over DINOv2/CLIP features of the first robot frame additionally supplies the conditioning demonstration, enabling execution on seen tasks without an explicit human video.

## Contributions
- H&R, claimed as the first dataset with perfectly aligned human-hand and robot-arm videos, collected via a coordinate-alignment protocol for VR teleoperation.
- A two-stage generative framework that treats fine-grained human-robot alignment as conditional video generation and repurposes one-step VPM features for action decoding.
- A KNN-based inference mode that executes seen tasks with high precision without any test-time human demonstration.

## Experimental Setup
All evaluation is on a real xArm workstation with 20 trials per task, covering seen push/pull, pick-and-place, and rotation tasks plus six generalization axes (appearance, position, instance, background, task combination, brand-new writing of "H"/"R" from random play data). Baselines are Diffusion Policy (CLIP-language-conditioned), XSkill (self-supervised cross-embodiment alignment), Video Prediction Policy (language-conditioned VPP), and ablations that feed human videos directly to the action decoder or skip VPM pretraining. Stage-1 VPM training on 2,600 episodes took 3 days on 4 NVIDIA A100 GPUs; stage-2 policy training took about 6 hours on 8 A100 GPUs, with a separate 6-hour training run on writing play data.

## Results
On seen basic tasks HUMAN2ROBOT achieves 100% (push and pull), 90% (pick and place), and 95% (rotation) for a 95% average, versus 80% for VPP, 53% for XSkill, and 28% for Diffusion Policy; the KNN variant retains an 82% average. Under generalization it scores 100% on appearance, 80% on position, 70% on novel instances (ping-pong balls, bananas), 80% on unseen backgrounds, 50% on combined tasks, and 70% on brand-new writing, while XSkill and VPP score 0% on most of these axes. Ablations show that decoding actions directly from human videos yields only 23% average success with jittery executions, and removing VPM pretraining collapses the average to 10%, confirming that the generative alignment stage supplies the critical motion prior.

## Limitations
The teleoperation data engine cannot yet collect paired data for tasks with a strong embodiment gap (e.g., screwing), so training is restricted to relatively simple pick-and-place style behaviors and generalization must be demonstrated from them. The KNN demo-free mode loses 10-20 percentage points of success rate relative to explicit human-video conditioning, and performance on composed long-horizon tasks remains at 50%. No quantitative comparison on dexterous hands is reported in the available evidence.
