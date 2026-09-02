# Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control

**Authors:** Linxi Xie, Lisong C. Sun, Ashley Neall, Tong Wu, Shengqu Cai, Gordon Wetzstein  
**Date:** 2026-02-20  
**Identifier:** [arXiv:2602.18422](https://arxiv.org/abs/2602.18422)  
**Zotero item:** `UPMHQI33` ([Zotero](zotero://select/library/items/UPMHQI33))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Generated Reality is a human-centric video world model for extended reality that is conditioned on both tracked head pose and joint-level hand poses, enabling users to drive dexterous hand-object interactions in zero-shot generated virtual environments. The paper systematically compares diffusion transformer conditioning strategies, finds that a hybrid 2D-3D scheme (ControlNet-style skeleton video plus 3D hand pose parameters injected by token addition) works best, distills the bidirectional teacher into a causal real-time system running at 11 FPS on a Meta Quest 3 pipeline, and shows in a user study that task accuracy jumps from 3.0% to 71.2% and perceived control from 1.74 to 4.21 on a 7-point Likert scale relative to a head-pose-only baseline.

## Background and Problem

XR content creation is expensive because it requires specialized expertise and laboriously designed 3D assets, and video world models promise zero-shot generation of immersive environments, yet current systems accept only coarse control signals such as keyboard input or text. Camera-motion or full-body-pose conditioning treats hands merely as part of the body and lacks the granularity needed for the wrist and finger articulation that dexterous hand-object interaction requires. The paper poses two questions: which conditioning strategies best preserve hand fidelity, realism, and temporal coherence when injecting joint-level hand poses into video diffusion transformers, and how to combine them with tracked head pose into a closed-loop, interactive egocentric generator.

## Method

- Hands are represented in two complementary forms: a 2D skeleton video rendered from the user's viewpoint (spatially aligned but depth-ambiguous and self-occluding) and 3D hand pose parameters from the UmeTrack model (wrist 6-DoF transform plus 20 joint angles), which provide metric depth and articulation.
- Four injection strategies for the 3D parameters are compared: token concatenation, token addition, adaptive layer normalization, and cross-attention; the winning hybrid design encodes the raw video and its skeleton video with a shared VAE, concatenates the latents channel-wise, and adds the embedded hand pose parameters to the patch tokens.
- Head pose is captured as a 6-DoF camera pose from headset sensors, converted into per-frame Plücker embeddings, encoded, and fused with hand embeddings via token addition; because both signals enter through the same operation, encoders are first trained independently (camera encoder initialized from the Wan FUN model) and then merged in a final joint fine-tuning stage.
- The study builds on Wan2.2 14B image-to-video (mixture-of-experts with high-noise and low-noise experts) trained with LoRA rank 32 on both experts; the deployed system distills a bidirectional Wan2.2 5B teacher into a causal 5B student via self-forcing, generating 12-frame chunks with per-frame head and hand conditioning.

## Contributions

- The first systematic study of joint-level hand pose conditioning strategies in video diffusion transformers, identifying a hybrid 2D-3D mechanism that outperforms 2D-only, 3D-only, and alternative injection schemes on video quality, hand pose accuracy, and camera pose accuracy.
- A joint camera-and-hand conditioning framework that coordinates head dynamics with hand actions, avoiding the failure mode where hand-only models reach for incorrect objects and camera-only models cannot manipulate anything.
- A complete closed-loop generated-reality system: a self-forcing-distilled causal model streaming bidirectionally between a Meta Quest 3 headset (Unity client tracking head and hands) and a server-side H100, achieving 11 FPS with 1.4 seconds of latency, of which the added conditioning contributes only 0.002 seconds.
- Human-subject evidence that tracked hand conditioning, rather than text prompting, is what makes fine-grained embodied tasks completable in generated environments.

## Experimental Setup

- The conditioning study uses HOT3D, segmented into 5-second clips yielding 5,824 training samples and 45 held-out evaluation clips, with motion-captured 3D hand annotations and synchronized camera poses.
- Metrics: PSNR, LPIPS, SSIM, and FVD for video quality; WiLoR-based Procrustes-aligned MPJPE (20 joints), MPVPE (778 vertices), and 2D landmark L2 error for hand accuracy (WiLoR fitting to ground-truth frames gives lower bounds of 9.42, 7.74, and 9.08); GLOMAP-estimated trajectories give translation and rotation errors for camera accuracy.
- Conditioning ablations train LoRA modules for over 1K steps at 480x480 with learning rate 1e-5 and batch size 16; generalization is additionally checked on GigaHands (8x larger than HOT3D) with a Wan2.2 5B model.
- The user study recruited 11 subjects (ages 22-30, 4 female, 7 male) who performed three tasks ("push the green button", "open the jar", "turn the steering wheel") under two conditions with 8-second time limits, four runs each in random order, rating perceived control on a 7-point Likert scale.

## Results

- In the conditioning ablation, the hybrid strategy achieves the best hand accuracy (MPJPE 12.23 mm, MPVPE 9.10 mm, 2D L2 error 11.50) versus the 2D skeleton baseline (12.38/9.25/11.72) and clearly beats pose-parameter injection schemes, whose FVD (560-677) shows cross-attention and AdaLN underperform even the unconditioned Wan2.2 baseline (601.55); the hybrid model approaches the estimator lower bounds.
- For joint control, the combined model reaches PSNR 18.60, LPIPS 0.2800, SSIM 0.6173, and FVD 396.93 while balancing hand accuracy (MPJPE 12.81) and camera accuracy (TransErr 0.25 m, RotErr 2.79 degrees), whereas CameraCtrl is camera-accurate but hand-inaccurate and the hand-only model has camera rotation error of 13.40 degrees.
- On GigaHands, hybrid conditioning reduces MPJPE by 10%, MPVPE by 11%, and 2D error by 34% relative to 2D-only conditioning, indicating the benefit grows with data scale and motion richness.
- In the user study, the hand-conditioned system achieved 71.2% average task accuracy versus 3.0% for the head-pose-only, text-prompted baseline, and mean perceived control of 4.21 versus 1.74.

## Limitations

- The system lags modern VR hardware in resolution, latency, stereo rendering, image quality, and compute efficiency, and 1.4 seconds of latency is not sufficient for fully immersive XR, partly because generation runs on a remote rather than local GPU.
- As with autoregressive video models generally, drift significantly degrades image quality after a few seconds of rollout.
- The distilled causal model exhibits typical distribution-matching-distillation drawbacks, including mode-seeking behavior and oversaturation over long horizons, and the system struggles with longer-range hand-object-object dependencies.
