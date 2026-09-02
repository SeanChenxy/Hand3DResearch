# HVG-3D: Bridging Real and Simulation Domains for 3D-Conditional Hand-Object Interaction Video Synthesis

**Authors:** Mingjin Chen, Junhao Chen, Zhaoxin Fan, Yujian Lee, Zichen Dang, Lili Wang, Yawen Cui, Lap-Pui Chau, Yi Wang  
**Date:** 2026-03-31  
**Identifier:** [arXiv:2604.03305](https://arxiv.org/abs/2604.03305)  
**Zotero item:** `HICSV6DG` ([Zotero](zotero://select/library/items/HICSV6DG))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HVG-3D is a diffusion framework for 3D-aware hand-object interaction (HOI) video synthesis: given a real image plus an explicit 3D condition (hand-object point cloud and tracking sequences from real-video reconstruction or a simulator), a 3D ControlNet on a frozen CogVideoX-5B-I2V backbone injects geometric and motion cues at every denoising step. A hybrid pipeline pairs real images with 3D conditions from simulation or other real videos, bridging real and simulated domains and achieving state-of-the-art spatial fidelity, temporal coherence, and controllability on TASTE-Rob.

## Background and Problem

Recent HOI video generation methods rely on 2D control signals (trajectories, optical flow, boxes, masks) that lack spatial expressiveness, causing (1) imperfect 3D understanding — partial cues yield unrealistic deformations and implausible interactions — and (2) high data cost, since 2D conditions must come from real videos, blocking cheap simulator data. Even 3D-aware efforts such as Diffusion as Shader (DaS) project 3D tracking into 2D sequences, losing spatial structure and depth relations. The task: from an input image, a T-frame hand-object point cloud sequence, and an optional 3D tracking sequence, generate a realistic, temporally coherent video faithful to the 3D constraints, accepting real or simulated conditions.

## Method

The architecture has two parts. (1) 3D-aware HOI diffusion: a pretrained CogVideoX-5B-I2V backbone (frozen) generates the video; the image latent is temporally zero-padded and concatenated with noised video latents. A 3D Point Cloud ControlNet — a trainable copy of all DiT blocks — encodes the conditions: a point cloud encoder (3DShape2VecSet) maps the T x N x 3 clouds to T x L x 768 features, aligned with the tracking latents via a learnable projection, concatenated, and injected through zero-initialized convolutions at every layer. (2) Hybrid condition pipeline: training data are recovered from monocular egocentric RGB via inter-frame differences plus YOLOv8-X (boxes), SAMURAI (masks), VGGT (per-frame point clouds), and SpatialTracker (3D tracking); training adds a mask-weighted diffusion loss ((1 + M_i)/2, after StableAnimator) emphasizing hand-object regions. At inference, the 3D condition can come from the same real-video pipeline, Blender-edited mesh sequences, or 3D HOI datasets such as ARCTIC and HOT3D.

## Contributions

(1) A paradigm bridging real and simulated domains for HOI video generation, synthesizing from a real image and a 3D condition from either simulation or another real video. (2) HVG-3D, a unified framework combining a 3D-aware diffusion architecture (3D point-cloud ControlNet over a frozen video DiT) with a hybrid input/condition construction pipeline for flexible, precise control. (3) Experiments showing state-of-the-art spatial fidelity, temporal coherence, and controllability with effective use of real and simulated data.

## Experimental Setup

Training uses the Taste-Rob Single Hand subset (office, dining, bedroom, kitchen, dressing-table scenes), center-cropped to 720 x 480 and cut into 49-frame clips; only the ControlNet blocks are optimized (AdamW, lr 1e-4, 20 epochs, effective batch size 4) on 8 x H20 GPUs. Evaluation uses 100 test videos (2% per-scene sample), split into 49-frame clips. Metrics: image quality (L1, PSNR, SSIM, LPIPS, CLIP Score, FID, CLIP-FID) and spatio-temporal similarity (FVD, ST-SSIM, GMSD-T), reported full-frame and within the hand-object mask region. Baselines: Kling, Wan2.2, CogVideoX, InterDyn, and DaS; Sora2 is compared qualitatively only (resolution mismatch).

## Results

Full-frame, HVG-3D achieves the lowest FID (58.2 vs. DaS 75.5, Kling 98.9), best CLIP Score (0.96), best ST-SSIM (0.97), best GMSD-T (0.40), and lowest FVD (13.8), though some low-level full-frame metrics trail DaS (L1 9.50 vs. 7.77; PSNR 24.15 vs. 24.83). Within the hand-object masked region — the primary interaction area — HVG-3D is best on all metrics: L1 20.90, PSNR 19.08, SSIM 0.97, LPIPS 0.032, CLIP 0.97, ST-SSIM 0.93, GMSD-T 0.15, FID 88.5, C-FID 13.1, with FVD reduced from 13.8 to 9.6. Qualitatively, only HVG-3D executes the specified manipulations (unrolling a folded sheet, moving plates) without hand or object deformation; DaS and InterDyn deform objects under folding or out-of-plane motion, and the general models fail the manipulations. Ablations (same training budget): removing the 3D point cloud drops PSNR to 18.44 and SSIM to 0.37; removing 3D tracking gives 22.76/0.75; removing the mask loss gives 22.09/0.80 — versus 24.15/0.81 for the full model — and the mask loss also accelerates convergence.

## Limitations

The paper states future work will extend to more diverse interaction scenarios, longer sequences, and closed-loop robotic integration, indicating limited current coverage there. It also reports some full-frame low-level metrics (L1, PSNR, SSIM, LPIPS) trail DaS, with advantages concentrated in the hand-object region, and that quality degrades substantially on tasks demanding depth awareness (folding or vertical motion) without 3D point-cloud conditioning.
