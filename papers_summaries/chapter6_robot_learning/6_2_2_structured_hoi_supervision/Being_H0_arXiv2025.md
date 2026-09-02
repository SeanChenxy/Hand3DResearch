# Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos

**Authors:** Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, Zongqing Lu  
**Date:** 2025-07-21  
**Identifier:** [arXiv:2507.15597](https://arxiv.org/abs/2507.15597)  
**Zotero item:** `7PWYMKYP` ([Zotero](zotero://select/library/items/7PWYMKYP))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Being-H0 is a dexterous Vision-Language-Action (VLA) model pretrained on large-scale human videos, treating the human hand as a foundation manipulator. Since teleoperated demonstrations lack the scale and diversity dexterous hands need, the paper proposes physical instruction tuning: VLA pretraining from human videos, physical space alignment, and post-training adaptation, supported by a part-level motion tokenizer and the UniHand dataset. Being-H0 beats GR00T N1.5 on hand motion generation and real-robot manipulation.

## Background and Problem

Teleoperated robot datasets are far smaller than internet-scale multimodal data, and dexterous-hand demonstrations are scarcer still due to hardware cost, restricting most VLAs to simple grippers. The paper asks whether a dexterous VLA can be pretrained from human videos to imitate human actions and adapt to robots via post-training. Given a scene image and instruction, the model generates MANO-parameterized 3D hand motion, or robot action chunks after post-training; challenges span data heterogeneity, precision-preserving quantization, cross-modal reasoning, and kinematic mismatch.

## Method

Physical instruction tuning has three stages. Pretraining uses InternVL3 backbones (1B/8B/14B) that process vision, text, and hand motion as one autoregressive sequence with shared attention. Continuous MANO-D162 features (6D rotations plus auxiliary joints) are discretized by separate wrist and finger tokenizers built on Grouped Residual Quantization, yielding 128 tokens per hand per second. Physical space alignment unifies camera intrinsics via weak-perspective projection and balances viewpoints through depth scaling and in-plane rotation. Post-training projects proprioception into the embedding space, and learnable action queries regress robot action chunks under an L1 loss.

## Contributions

- Physical instruction tuning, establishing human hands as the foundational manipulator for robot hand transfer.
- Part-level motion tokenization preserving millimeter-level precision while compatible with autoregressive language models.
- UniHand, aggregating 11 sources (motion capture, VR, and pseudo-annotated RGB videos) into over 444K trajectories, 130M frames, 1,155 hours, and 166.5 million instruction samples.
- Being-H0, the first dexterous VLA trained on explicit motion modeling from large-scale human videos.

## Experimental Setup

Hand motion evaluation uses a held-out 5% of UniHand, split into a head split (EgoDex) and a tail split (TACO, HOI4D, H2O, OakInk2), over three generation and translation tasks measured by MPJPE, MWTE, PA-MPJPE, retrieval metrics, FID, and valid rate. Real-robot experiments use a 7-DoF Franka Research 3 arm, a 6-DoF Inspire hand, and a RealSense L515; each task is post-trained on 50–100 teleoperated trajectories and evaluated over 20 randomized trials against GR00T N1.5 and InternVL3.

## Results

On visual-grounded generation, Being-H0-14B reaches a 100% valid generation rate and MPJPE of 6.87/8.11 cm (head/tail) versus 9.82/15.35 cm for GR00T N1.5, with T2M R@3 of 19.0/22.1. In real-world manipulation, Being-H0 attains the highest success rate on all seven tasks: 0.75/0.65/0.60 for seen/unseen/cluttered pick-and-place, 0.85 Close-Toolbox, 0.60 Close-Lid, 1.00 Pour-Cup, and 0.75 Unfold-Clothes. With 25% of demonstrations it matches the InternVL3 baseline trained on 50–100% of data, and on Close-Lid it reaches 15% success where the baseline scores 0%. Ablations confirm part-level tokenization, view balancing, and data scaling each help.

## Limitations

Long-horizon generation degrades with error accumulation. Robot transfer relies on a simple MLP projection, with adaptive strategies deferred to future work; the current version excludes object pose modeling, leaves occlusion-heavy or dynamic-camera datasets unprocessed, and omits depth and tactile cues.
