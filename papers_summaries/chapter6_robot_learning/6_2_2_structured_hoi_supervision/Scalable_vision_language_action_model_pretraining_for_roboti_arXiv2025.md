# Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos

**Authors:** Qixiu Li, Yu Deng, Yaobo Liang, Lin Luo, Lei Zhou, Chengtang Yao, Lingqi Zeng, Zhiyuan Feng, Huizhi Liang, Sicheng Xu, Yizhong Zhang, Xi Chen, Hao Chen, Lily Sun, Dong Chen, Jiaolong Yang, Baining Guo  
**Date:** 2025-10-24  
**Identifier:** [arXiv:2510.21571](https://arxiv.org/abs/2510.21571); DOI `10.48550/arXiv.2510.21571`  
**Zotero item:** `9GNQRWJ2` ([Zotero](zotero://select/library/items/9GNQRWJ2))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

This paper shows that unscripted, unannotated real-life egocentric human videos can be transformed into Vision-Language-Action (VLA) training data fully aligned with robotic data in task granularity and labels, treating human hands as dexterous robot end-effectors. A fully automatic pipeline performs 3D motion labeling, atomic action segmentation, and instruction labeling, yielding a hand-VLA dataset of 1 million episodes and 26 million frames. A dexterous hand VLA model built on a PaliGemma-2 backbone with a diffusion action expert shows strong zero-shot hand action prediction in unseen environments and, after fine-tuning on 1.2K teleoperated trajectories, substantially improves real-world dexterous manipulation with clear data-scaling behavior.

## Background and Problem

Robot V-L-A data is collected by teleoperation in laboratory settings, making it costly and limited in skills, objects, and scene variation, and no large-scale dexterous hand action dataset exists for pretraining. The paper asks whether in-the-wild human videos can become robot-aligned VLA data, requiring two alignments: task alignment, meaning segmentation of atomic short-horizon actions matching robot data recipes, and label alignment, meaning recovery of metric 3D hand motion from single, uncalibrated, often moving cameras plus precise language instructions. Given a visual observation, hand state, and instruction, the model predicts future hand actions; after fine-tuning it controls a dexterous robot hand.

## Method

The pipeline has three automatic stages. First, 3D motion labeling: camera intrinsics are estimated with DroidCalib or MoGe-2 with DeepCalib, HaWoR reconstructs per-frame camera-space MANO hands, and a modified MegaSAM tracks camera pose, producing metric world-space hand trajectories transformable into any frame's camera coordinates to simulate a static camera. Second, atomic action segmentation places cutting points at speed minima of the 3D wrist trajectories, detected independently per hand with no extra model inference. Third, instruction labeling: eight sampled frames per clip are overlaid with projected hand trajectories and captioned by GPT-4.1 in imperative form, with non-manipulation clips labeled N/A. The dataset draws on Ego4D (77%), Epic-Kitchen (12%), EgoExo4D (6%), and Something-Something-V2 (5%), ignoring their original annotations. The VLA model pairs a 3B PaliGemma-2 VLM (with a camera-FoV token and a learnable cognition token) with a DiT-Base diffusion action expert that denoises 102-dimensional bimanual actions (relative wrist translation and rotation plus 15 MANO joint angles per hand) using causal attention and per-dimension action masks unifying single- and dual-hand episodes. Trajectory-aware augmentation transforms images and actions consistently, and each robot joint is mapped to its closest human joint for fine-tuning.

## Contributions

- The first approach to pretrain manipulation VLA models from large-scale unstructured, unannotated real-life human videos aligned with robot V-L-A data.
- A fully automatic framework combining monocular 3D hand and camera reconstruction, speed-minima-based atomic segmentation, and trajectory-overlay GPT captioning.
- A 1M-episode, 26M-frame hand-VLA dataset with measured visual and linguistic diversity exceeding EgoDex, Open X-Embodiment, DROID, and AgiBot World.
- A dexterous hand VLA architecture demonstrating zero-shot generalization, strong fine-tuned robot performance, and favorable data scaling.

## Experimental Setup

Hand action prediction is evaluated on a new benchmark: grasping over 47 unseen environments with 396 annotated objects (metric: minimum distance between predicted finger trajectories and object points, d_hand-obj) and a general-action user study across 117 unseen environments with 23 participants. Robot experiments use a Realman3 arm with 12-DoF XHand dexterous hands, fine-tuned on 1.2K teleoperated trajectories for pick and place, functional grasping, pouring, and sweeping, on seen and unseen objects, categories, and backgrounds, against VPP, Pi0, no VLA pretraining, latent-action pretraining, and OXE pretraining with identical fine-tuning data.

## Results

On grasping, the pretrained model achieves average/median d_hand-obj of 8.8/6.2 cm versus 19.1/18.4 cm for Being-H0 (8B) and 17.6/18.3 cm for EgoDex pretraining, with initial hand-object distance at 20.0 cm; the user-study score is 1.91 versus 0.15 for Being-H0. Fixed-interval segmentation (10.5/8.8 cm) and removing trajectory overlays (11.7/10.7 cm) both degrade performance, and accuracy improves approximately linearly with data scale on the log axis. On real robots, the fine-tuned model averages 71.0% success on seen tasks versus 46.9% for Pi0, 24.8% for VPP, and 41.3% for OXE pretraining, and 64.6% on unseen objects, categories, and backgrounds versus 16.1% for Pi0 and 0.0% for latent-action pretraining. EgoDex pretraining underperforms a model trained on only 10% of this paper's data despite more frames (130M versus 2.6M).

## Limitations

The authors note that pretraining data still contains inaccuracies from current 3D reconstruction and VLM captioning, with better reconstruction and filtering planned. The framework targets short-horizon atomic skills and does not organize data for long-horizon planning. Real-robot experiments focus mainly on single-handed tasks, with bimanual capability shown only through a simple hand-over demonstration; multi-view inputs and tactile feedback are left for future work.
