# SViMo: Synchronized Diffusion for Video and Motion Generation in Hand-object Interaction Scenarios

**Authors:** Lingwei Dang, Ruizhi Shao, Hongwen Zhang, Wei Min, Yebin Liu, Qingyao Wu  
**Date:** 2025-06-03  
**Identifier:** [arXiv:2506.02444](https://arxiv.org/abs/2506.02444)  
**Zotero item:** `UKTK3P5Z` ([Zotero](zotero://select/library/items/UKTK3P5Z))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

SViMo jointly denoises hand-object interaction (HOI) video and 3D motion in one synchronized diffusion process built on a pretrained image-to-video DiT foundation model, while a vision-aware 3D interaction diffusion model (VID) generates explicit hand trajectories and object point clouds in a closed loop — removing the need for predefined object models or pose guidance and yielding the best VBench overall score and large motion-quality gains over MDM and EMDM.

## Background and Problem

The paper targets HOI generation where, given a reference image and a text prompt, the model must simultaneously produce the future video and a 3D motion sequence of hand joint trajectories and object point clouds. Existing 3D HOI motion generators rely on predefined object meshes and lab-captured motion data, hurting generalization and causing ambiguous boundaries and implausible actions; HOI video generators built on large video foundation models or pose-guided image animation achieve pixel fidelity but lack explicit motion dynamics, require pose sequences as input, and suffer from flickering. The authors argue that visual appearance and motion patterns share the same physical dynamics and can be denoised together.

## Method

SViMo extends a pretrained image-to-video DiT (CogVideoX-based) into joint video-motion generation. Since 3D motion data differs greatly from 2D video representations, 3D interactions are projected onto the image plane as "rendered motion videos". Text (frozen T5), video, and motion tokens are fused through tri-modal adaptive modulation (per-modality scaling, shifting, and gating learned from the timestep embedding) plus a unified 3D full-attention capturing intra- and inter-modal dependencies. The Vision-aware 3D Interaction Diffusion (VID) maps SViMo's video and motion latents, via a dual-stream 3D convolutional encoder and cross-attention, into explicit 3D hand poses (MSE loss) and object point clouds (Chamfer loss). A closed-loop feedback cycle completes the design: VID's output is projected back to a rendered motion video and injected as interaction guidance, while VID gradients backpropagate into SViMo as a gradient constraint; training warms up VID for 5K steps, then jointly optimizes both with L = w1*L_SViMo + w2*L_VID (w1=1, w2=0.05).

## Contributions

1. A synchronized diffusion model jointly denoising HOI video and motion, integrating large-scale visual priors with motion dynamic constraints.
2. A vision-aware 3D interaction diffusion generating explicit 3D interaction sequences in a closed-loop pipeline that enhances video-motion consistency.
3. End-to-end HOI video and motion generation without predefined poses or object models, with zero-shot generalization to unseen real-world data.

## Experimental Setup

Training uses TACO (2.5K bimanual interaction sequences, 20 object categories, 196 3D models, 14 participants, 15 interaction types, 5.2M frames at 30 Hz), split 1:9 for training:test; hand-object regions are cropped to 3:2 aspect ratio and downsampled to 49 frames at 8 FPS (416x624x49). The DiT backbone has 42 blocks, 48 attention heads, and 6.02B parameters, producing 26,590 multimodal tokens per sequence. Training ran on 4 A800-80G GPUs with DeepSpeed ZeRO-3, gradient checkpointing, and BF16 (per-GPU batch 4; 30K steps at 240x368 then 5K steps at full resolution; 50 sampling steps at inference). Video baselines: Hunyuan-13B, Wan-14B (zero-shot), Animate Anyone, Easy Animate, CogVideoX-5B; motion baselines: MDM and EMDM. Metrics: VBench content/dynamic quality and an Overall product score; MPJPE and motion smoothness for hands; Chamfer distance and FID (via a pretrained interaction autoencoder) for objects.

## Results

SViMo achieves the best VBench Overall score of 0.8785 (subject consistency 0.9500, background 0.9533, temporal smoothness 0.9898, dynamic degree 0.9801), ahead of CogVideoX-5B (0.8727), Easy Animate (0.8297), Animate Anyone (0.8172), Wan-14B (0.7675), and near-static Hunyuan-13B (0.4493). For motion, it records MPJPE 0.1087, motion smoothness 0.0255, Chamfer 0.1577, and FID 0.1050, versus EMDM (MPJPE 0.3255, Chamfer 0.7788, FID 0.3681) and MDM (0.3382, 0.7915, 0.4056). User studies (41 participants, 1,066 valid responses over 26 prompts) give a 78.42% video preference rate; motion results are preferred in 97.56% of 410 responses. Ablations show that decoupled video/motion modeling drops the Overall score from 0.8719 to 0.8381 and worsens motion FID (0.0546 to 0.0575), while the full VID with both interaction guidance and gradient constraints performs best. Zero-shot tests on household objects (rollers, spatulas, spoons, bowls) show real-world generalization.

## Limitations

The appendix identifies three limitations: (1) the method relies on a large-scale pretrained video foundation model fine-tuned on a comparatively small video-3D motion pair dataset, whose expansion remains essential; (2) generated 3D object point clouds are restricted to rigid, simple objects and struggle with structurally complex geometries; (3) the pretrained foundation model's capabilities directly affect training efficiency and final performance — LoRA fine-tuning of CogVideoX gives suboptimal results, and even full-parameter fine-tuning can produce blurring artifacts when sampling at reduced resolution.
