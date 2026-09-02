# EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos

**Authors:** Ruihan Yang, Qinxi Yu, Yecheng Wu, Rui Yan, Borui Li, An-Chieh Cheng, Xueyan Zou, Yunhao Fang, Xuxin Cheng, Ri-Zhao Qiu, Hongxu Yin, Sifei Liu, Song Han, Yao Lu, Xiaolong Wang  
**Date:** 2025-07-18  
**Identifier:** [arXiv:2507.12440](https://arxiv.org/abs/2507.12440); DOI `10.48550/arXiv.2507.12440`  
**Zotero item:** `MWHAIMSV` ([Zotero](zotero://select/library/items/MWHAIMSV))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

EgoVLA trains a Vision-Language-Action (VLA) model on egocentric human videos and transfers it to a bimanual humanoid robot. The VLA predicts human wrist and hand actions, converted to robot actions via inverse kinematics and retargeting, then fine-tuned on a few robot demonstrations. A new simulation benchmark with 12 bimanual tasks shows EgoVLA outperforming specialist and generalist baselines, with human pretraining improving both in-domain success and generalization.

## Background and Problem

Imitation learning from real robot data avoids the sim-to-real gap but is constrained in scale and task diversity by the need for robots and teleoperation. The paper instead trains a Human Egocentric VLA: given egocentric observations, a language instruction, and current hand pose, it predicts future human wrist and hand actions, which convert to robot actions through inverse kinematics and retargeting; small-scale robot fine-tuning then corrects embodiment mismatch. A further problem is the lack of scalable, reproducible evaluation for humanoid bimanual manipulation.

## Method

EgoVLA is built on the NVILA-2B vision-language model. Inputs are six RGB frames, a language instruction, action query tokens, and proprioception including wrist pose and MANO hand pose (top 15 PCA components). A 300M transformer action head predicts a one-second action chunk for both hands, trained with weighted L2 losses. Pretraining uses about 500,000 image-action pairs from HOI4D, HOT3D, HoloAssist (subsampled one-tenth), and TACO, with world-frame camera poses projecting future wrist positions into the current camera frame. For transfer, robot demonstrations are retargeted by optimizing MANO parameters to match robot fingertips; at deployment, inverse kinematics converts wrist poses to arm commands and a lightweight MLP maps MANO parameters to hand joints, with 5e-5 m mean fingertip error.

## Contributions

- A Human Egocentric VLA pretrained on about 500K image-action pairs predicting wrist and hand actions convertible to robot actions.
- A unified action space aligning human and robot hands via MANO retargeting, enabling fine-tuning without architectural changes.
- The Ego Humanoid Manipulation Benchmark: a Unitree H1 humanoid with two Inspire hands, 12 bimanual tasks, 100 demonstrations per task, and 25 visual backgrounds.
- Evidence that human pretraining improves success rates and generalization.

## Experimental Setup

The benchmark provides 7 short-horizon tasks (e.g., Push-Box, Flip-Mug) and 5 long-horizon tasks (e.g., Sort-Cans), evaluated with Success Rate (SR) and Progress Rate (PSR) under Seen and Unseen background splits. Baselines are EgoVLA-NoPretrain, fine-tuned on robot data only, and ACT, a per-task specialist transformer; a 50%-demonstration variant probes robot-data scale. Human motion modeling is evaluated by wrist translation error and instruction-following on HOI4D with modified prompts.

## Results

After training on human videos, EgoVLA predicts wrist translation with about 8 cm average error and shifts predicted trajectories when instructions change. Zero-shot deployment without robot fine-tuning yields 0% success. With post-training, EgoVLA reaches mean SR of 77.78 on seen short-horizon tasks versus 64.55 for EgoVLA-NoPretrain and 24.87 for ACT, and 45.93 versus 26.67 and 2.22 on seen long-horizon tasks. On unseen backgrounds, EgoVLA attains 69.11 SR on short-horizon tasks while the NoPretrain baseline drops 23%, and about 30% SR on long-horizon tasks. Halving robot demonstrations degrades long-horizon seen SR from 45.93% to 7.41%.

## Limitations

The framework requires human videos with hand and wrist pose annotations, limiting data availability, though AR/VR devices may ease this. EgoVLA cannot be deployed without fine-tuning on a moderate amount of robot data; more embodiment-agnostic pretraining is left for future work.
