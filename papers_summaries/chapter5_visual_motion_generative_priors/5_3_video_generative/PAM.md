# PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation

**Authors:** Mingju Gao, Kaisen Yang, Huan-ang Gao, Bohan Li, Ao Ding, Wenyi Li, Yangcheng Yu, Jinkun Liu, Shaocong Xu, Yike Niu, Haohan Chi, Hao Chen, Hao Tang, Yu Zhang, Li Yi, Hao Zhao  
**Date:** 2026-03-23  
**Identifier:** [arXiv:2603.22193](https://arxiv.org/abs/2603.22193)  
**Zotero item:** `G5VSU2HW` ([Zotero](zotero://select/library/items/G5VSU2HW))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

PAM is a decoupled three-stage engine for controllable sim-to-real HOI video generation requiring only sparse inputs: initial and target MANO hand poses, an initial 6-DoF object pose, and an appearance-less object mesh. A pretrained pose model (GraspXL) interpolates a hand-object trajectory; a ControlNet-tuned Flux synthesizes a diverse first frame; a multi-condition CogVideoX animates it into a video. On DexYCB it reaches FVD 29.13 and MPJPE 19.37 mm at 480x720, and 3,400 synthetic videos (207k frames) let a hand pose estimator trained on 50% of real data match the 100%-real baseline. The paper reports acceptance to CVPR 2026.

## Background and Problem

HOI reconstruction and synthesis are central to embodied AI and AR/VR, but generation research is fragmented across three disjoint tracks: (1) pose-only synthesis predicts MANO trajectories without pixels (e.g., GraspXL); (2) single-image HOI generation hallucinates appearance from masks or 2D cues but lacks dynamics; (3) video generation methods (InterDyn, ManiVideo) require both the entire pose sequence and the ground-truth first frame, unavailable from a simulator, preventing true sim-to-real deployment. The motivation is data: annotated real HOI sequences are costly, limiting scalability. The problem: generate photorealistic, temporally coherent grasp-to-place videos that begin at the initial pose, end at the target pose, and need no ground-truth first frame or complete pose sequence, enabling sim-to-real transfer.

## Method

PAM decomposes generation into three stages. Stage I (Pose Generation): pretrained GraspXL takes the initial MANO pose (51x3), 6-DoF object pose, and object mesh, producing temporally coherent hand-object pose trajectories. Stage II (Appearance Generation): Flux is fine-tuned with a ControlNet fork conditioned on depth, segmentation, and hand-keypoint images (VAE-encoded, channel-concatenated), injected into the first two DiT blocks through zero-convolutions; only ControlNet parameters train, synthesizing the first frame with appearance diversity. Stage III (Motion Generation): the Stage-I trajectory is rasterized into depth, instance segmentation, and 2D keypoint sequences, encoded by a pretrained video VAE, and injected into CogVideoX through 12 duplicated DiT blocks using the same multimodal conditions as Stage II; each cue is randomly masked with probability 0.2 during training to prevent over-reliance on any modality, also improving robustness to noise (FVD 30.45 vs. 34.58).

## Contributions

(1) Minimal-conditioning generation: an engine requiring only sparse pose keyframes and object geometry, overcoming the first-frame bottleneck of prior methods. (2) A decoupled architecture separating pose, appearance, and motion synthesis, leveraging multi-modal conditions (depth, segmentation, keypoints) for realism, controllability, and diversity. (3) State-of-the-art results on DexYCB and OAKINK2 plus demonstrated downstream utility: synthetic videos as data augmentation yield measurable gains for hand pose estimation.

## Experimental Setup

Evaluation uses DexYCB (s0-split: 6,400 training / 1,600 validation videos) and a curated OAKINK2 subset of 8,000 49-frame clips (6,400/1,600). Depth conditions are estimated with DepthCrafter; segmentation and keypoints come from dataset annotations. Training: 8x NVIDIA 800 GPUs (as stated in the paper), batch 4x8, lr 1e-4, 8,000 steps, AdamW with DeepSpeed. Evaluation samples 1,600 49-frame videos per test. Metrics: SSIM/LPIPS/PSNR, FVD (StyleGAN-V implementation), Motion Fidelity (MF) from 100 CoTracker3-tracked foreground points, and MPJPE (mm) via Hamer-estimated joints. Baselines: CosHand (fine-tuned on the same splits), InterDyn, ManiVideo (from original papers; evaluated with ground-truth first frames as they require).

## Results

On DexYCB, PAM achieves FVD 29.13 (vs. InterDyn 38.83, CosHand 58.51), MF 0.712, LPIPS 0.069, SSIM 0.914, PSNR 30.17, and MPJPE 19.37 mm (vs. CosHand 30.05 mm) at 480x720 resolution versus 256x256/256x384 baselines. On OAKINK2, the full model improves FVD from 68.76 (CosHand) to 46.31 and MPJPE from 14.49 to 7.01 mm. Condition ablations on DexYCB show monotonic gains: single conditions give FVD 30.00-33.41, pairs 29.32-29.62, depth+hand+segmentation the best 29.13/19.37. Downstream, training SimpleHand with PAM's 3,400 synthetic videos (207,400 frames, bottom 25% filtered by Hamer pose accuracy) plus 50% real data matches the 100%-real baseline (PA-MPJPE 5.5 vs. 5.5 mm; F-Score@05 0.8001 vs. 0.7953). Zero-shot transfer from DexYCB (single-hand) to OAKINK2 (bimanual) remains plausible. The full pipeline takes 301.1 s for 40 frames (Stage III: 245.7 s on an H20).

## Limitations

The conclusion defers more complex object interactions and unifying the motion and appearance stages end-to-end to future work, indicating the current pipeline is staged rather than end-to-end and covers relatively simple interactions. The error-propagation analysis notes that Stage-I geometric errors (interpenetration, missing contact) propagate into physically implausible interactions despite photorealistic output, and that Stage-III quality depends heavily on the Stage-II reference frame, with low-quality first frames degrading textures and worsening temporal flickering.
