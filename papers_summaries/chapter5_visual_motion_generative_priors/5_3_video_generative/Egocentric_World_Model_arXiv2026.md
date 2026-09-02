# Egocentric World Model for Photorealistic Hand-Object Interaction Synthesis

**Authors:** Dayou Li, Lulin Liu, Bangya Liu, Shijie Zhou, Jiu Feng, Ziqi Lu, Minghui Zheng, Chenyu You, Zhiwen Fan  
**Date:** 2026-03-13  
**Identifier:** [arXiv:2603.13615](https://arxiv.org/abs/2603.13615)  
**Zotero item:** `EGEGRU6E` ([Zotero](zotero://select/library/items/EGEGRU6E))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

EgoHOI is an egocentric hand-object interaction (HOI) world model that simulates photorealistic, contact-consistent first-person rollouts purely from action signals - reconstructed 3D hand kinematics and metric head motion - deliberately avoiding the common shortcut of conditioning on privileged future object trajectories. It injects physics-informed embeddings distilled from 3D geometry and kinematics estimates (hand kinematics, Plücker-ray ego motion, and a first-frame object entity anchor) into a frozen Wan-DiT backbone via lightweight adapters, and on a HOT3D-derived benchmark it beats Wan, Cosmos 2B/14B, and Uni3C on frame prediction, ego-motion consistency, and kinematic fidelity (for example, hand missing ratio drops from 14.61% for Cosmos 14B to 5.84%).

## Background and Problem

World models intended as scalable data sources for embodied AI should act as true simulators that infer interaction dynamics strictly from user actions rather than conditional video generators that rely on privileged future object states. Egocentric HOI world modeling is profoundly difficult because rapid head motions, severe occlusions, and high-DoF hand articulation abruptly alter contact topologies; existing HOI video generation is mostly static and exocentric, and many methods condition on ground-trutured future object trajectories or waypoints, which bypasses the core challenge of contact-driven dynamics reasoning. Once those future-state shortcuts are removed, simply scaling RGB data is insufficient for physical accuracy, motivating explicit metric and kinematic structure - now extractable from unstructured video by 3D foundation models for geometry and hand pose.

## Method

- EgoHOI instantiates a latent world model: the VAE-encoded first frame is the initial state, actions are a sequence of hand kinematics renders and metric head poses, and a DiT transition model evolves the latent state over 81-frame rollouts, with the first frame also anchoring the rollout through CLIP image embeddings supplied as fixed cross-attention keys and values.
- Hand Kinematic Embeddings (HKE) encode dense per-frame renders of reconstructed hand meshes through a temporal 3D convolutional stack into the 5120-dim backbone token space, fused by a learnable gated residual, while a reference-hand branch encodes the first-frame hand pose to stabilize hand identity and reduce temporal drift.
- Ego-Motion Embeddings (EME) convert calibrated head poses (relative to the first frame) into per-pixel 6D Plücker ray fields, processed by causal 3D convolutions and hybrid 2D-convolution/temporal-attention blocks, then injected into the first DiT blocks through zero-initialized linear adapters that preserve backbone stability at initialization.
- Object Entity Embeddings (OEE) encode a first-frame object segmentation (from an off-the-shelf segmenter) with the frozen VAE and a 3D convolutional patchifier, and append these tokens with shifted rotary positions as key/value-only context during self-attention, providing a persistent entity reference that curbs object drift.
- Implementation builds on Wan 2.1 14B with LoRA rank 128, trained 8,000 steps (about one day) on 16 H100 GPUs at 480x480, batch size 1, and runs at 16 FPS during inference; the 3D priors come from off-the-shelf reconstruction and hand-pose foundation models.

## Contributions

- An egocentric HOI world model that performs causal simulation of physically plausible hand-object interactions under user-specified actions, without privileged future object states, in contrast to trajectory-conditioned HOI video generation.
- Physics-informed embeddings - hand kinematics, metric Plücker-ray ego motion, and first-frame object entity anchoring - that regularize action-driven latent dynamics toward physically consistent interaction while preserving the pretrained generative prior through lightweight adapters.
- A HOT3D-derived evaluation protocol covering visual prediction, ego-trajectory consistency (ATE/RPE/RRE), object integrity (Object-CLIP plus custom object position and orientation errors), and kinematic fidelity (hand missing ratio, MPJPE, segmentation RMSE via HaMeR).
- Evidence that backbone scaling and motion control alone do not solve egocentric HOI simulation: Cosmos 2B to 14B scaling yields only modest gains and Uni3C-style motion control degrades under rollout, whereas explicit 3D physics priors deliver the largest improvements.

## Experimental Setup

- Since no public benchmark existed for egocentric HOI world modeling, the authors curate a HOT3D-Clips benchmark (Project Aria forward-facing RGB stream, 150-frame clips at 16 FPS with metric camera, hand, and 6-DoF object annotations), split 90/10 at the clip level into 1,364 training and 152 test samples, and evaluate on 100 clips from scenes excluded from training using sliding 81-frame windows.
- Baselines comprise the Wan 2.1 backbone, Cosmos 2B and 14B, and Uni3C, all fine-tuned from released checkpoints with the same data and protocol; several recent egocentric world models were omitted because their implementations were not publicly available.
- Metrics: PSNR, SSIM, LPIPS, and Object-CLIP for frame prediction; six VBench dimensions (subject/background consistency, motion smoothness, dynamic degree, aesthetic and imaging quality); ATE, RPE, and RRE computed on MapAnything-estimated trajectories; and HaMeR-based missing ratio, MPJPE, and hand-segmentation RMSE for kinematic fidelity.

## Results

- EgoHOI achieves the best score in every metric group: PSNR 21.05 versus 15.89 for Cosmos 14B, SSIM 0.65, LPIPS 0.27, Object-CLIP 0.92, ATE 0.084, RRE 5.192, RPE 0.021, missing ratio 5.84%, MPJPE 0.014, and hand-segmentation RMSE 0.044.
- VBench confirms the gains: highest subject consistency (95.51%), background consistency (94.91%), aesthetic quality (52.03%), and imaging quality (64.48%), while its lower dynamic degree (53.29%) reflects a deliberate emphasis on stable hand-object interaction over highly dynamic camera motion.
- Ablations isolate each embedding: HKE cuts the missing ratio and MPJPE from 28.77% and 0.576 (base backbone) to 6.47% and 0.015; EME lowers ATE/RRE/RPE from 0.133/15.249/0.039 to 0.096/6.525/0.023 alone and 0.084/5.192/0.021 in the full model; OEE raises Object-CLIP from 0.81 to 0.83 and slashes object position/orientation errors, which reach 0.015 and 9.412 in the full model versus 0.141 and 27.739 for the base.
- Qualitatively, rollouts generated from the same first frame but different user inputs diverge into distinct, plausible interaction trajectories, demonstrating genuine action-driven simulation rather than fixed continuation.

## Limitations

- The paper itself notes that fully evaluating physical plausibility remains an open problem: the benchmark measures appearance, trajectory, and kinematic proxies, but more direct contact-aware and dynamics-aware evaluation of first-person rollouts is left to future work.
- Rollouts are capped at 81 frames per generation pass, and the comparatively low dynamic degree indicates the model favors stable interaction over expressive motion.
- Comparison coverage is limited by reproducibility: several recent egocentric world models, including PlayerOne and Generated Reality, could not be evaluated because their implementations were not publicly available at submission time.
