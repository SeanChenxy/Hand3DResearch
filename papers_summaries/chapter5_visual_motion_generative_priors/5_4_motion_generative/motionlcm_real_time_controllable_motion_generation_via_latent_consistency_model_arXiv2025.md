# MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model

**Authors:** Wenxun Dai, Ling-Hao Chen, Jingbo Wang, Jinpeng Liu, Bo Dai, Yansong Tang  
**Date:** 2024-04-30  
**Identifier:** [arXiv:2404.19759](https://arxiv.org/abs/2404.19759) ; DOI `10.1007/978-3-031-72640-8_22`  
**Zotero item:** `GNJKKTSF` ([Zotero](zotero://select/library/items/GNJKKTSF))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

MotionLCM extends controllable text-to-motion generation to real time by applying latent consistency distillation to a motion latent diffusion model (MLD): one-step inference takes about 30 ms per motion (AITS 0.030 s) versus 0.217 s for MLD and 24.74 s for MDM, while matching or exceeding MLD's quality on HumanML3D (one-step FID 0.467, R-precision top-3 0.803). A motion ControlNet operating in the latent space, supervised both by latent reconstruction and by an explicit control loss in the decoded motion space, enables trajectory-conditioned generation that outperforms OmniControl in quality, control accuracy, and speed (1929x faster).

## Background and Problem

Diffusion-based text-to-motion models (MDM, MLD, MotionDiffuse) produce high-quality motion but require many denoising steps, blocking real-time applications; spatial-temporally controlled variants are even slower (OmniControl takes about 81 s per sequence). The paper targets real-time controllable motion generation: distilling a motion latent diffusion model into a consistency model for one- or few-step sampling, and injecting spatial-temporal control signals (initial joint trajectories) into a latent space that has no explicit motion semantics — a non-trivial problem because latent codes cannot be directly manipulated like raw motion.

## Method

MotionLCM distills MLD (reproduced with improved performance) via latent consistency distillation following Latent Consistency Models: a frozen pre-trained VAE encoder compresses motion into latents, noise is added over n+k steps, and an online network (initialized from the teacher) is trained with an EMA-updated target network to enforce the self-consistency property between time steps n+k and n, using a k-step DDIM ODE solver estimate and classifier-free guidance integrated into the distillation (training guidance scale sampled from [5,15], Huber loss). For control, a motion ControlNet (trainable copy of MotionLCM with zero-initialized linear layers) and a transformer-based Trajectory Encoder process the initial tau poses (trajectories of K control joints, tau=0.25, K=6: pelvis, both feet, head, both wrists) and steer one-step denoising in latent space. Because latent reconstruction alone under-constrains control, the predicted latent is decoded through the frozen VAE decoder into the motion space, where a masked control loss on global joint locations provides explicit supervision (overall loss L_recon + lambda*L_control, lambda=1.0). The same mechanism supports autoregressive long-motion generation conditioned on the last frames of the previous clip.

## Contributions

- First introduction of consistency distillation into motion generation, extending text-conditioned motion synthesis to real-time (about 30 ms one-step) with quality on par with or better than the 50-step MLD teacher.
- A motion ControlNet in the latent space, made effective by explicit control supervision decoded into the vanilla motion space, enabling high-quality trajectory/multi-joint control at real-time speed.
- Extensive experiments and ablations showing a favorable balance of generation quality, controllability, and runtime efficiency on HumanML3D.

## Experimental Setup

Dataset: HumanML3D (14,616 motion sequences, 44,970 textual descriptions) with the redundant motion representation (root velocity/height, joint positions, velocities, rotations, foot contacts). Metrics: Average Inference Time per Sentence (AITS), FID, R-precision (top-1/2/3), MM Dist, Diversity, MultiModality, plus control errors — trajectory error, location error (50 cm threshold), and average error of control joints. Training: MotionLCM distilled for 96K iterations (AdamW, batch 256, lr 2e-4, cosine decay with 1K warm-up, EMA 0.95, DDIM skipping interval k=20, Huber loss); ControlNet trained 192K iterations (batch 128, lr 1e-4, L2 loss, lambda=1.0). Evaluation uses 20 repeated runs with 95% confidence intervals. Trained on an NVIDIA RTX 4090, tested on a Tesla V100.

## Results

Text-to-motion on HumanML3D: MotionLCM achieves AITS 0.030 s (1-step) with FID 0.467, R-precision top-3 0.803, MM Dist 3.022, Diversity 9.631; two-step inference gives the best R-precision (0.805) and MM Dist (2.986), and four-step gives the best FID (0.304), versus the reproduced MLD at 0.225 s and FID 0.450 (original MLD 0.217 s, FID 0.473). This is roughly an order of magnitude faster than MLD, versus MDM's 24.74 s and MotionDiffuse's 14.74 s AITS. Controllable generation: with both latent and motion-space supervision (LC&MC), 2-step MotionLCM reaches FID 0.397, trajectory error 0.1960, location error 0.0143, and average error 0.1092, versus OmniControl's FID 2.328, trajectory error 0.3362, and 81.00 s AITS — 1929x faster and 13x faster than MLD. Ablations show dynamic guidance scales [5,15] beat static 7.5 (FID 0.467 vs. 0.479), Huber beats L2 (0.467 vs. 0.622), k=20/10 outperform k=1 (0.467/0.449 vs. 0.635), and whole-body control with K=22 joints further lowers average error to 0.0881 (2-step).

## Limitations

The paper states that because MLD's VAE lacks explicit temporal modeling, MotionLCM cannot achieve good temporal interpretability, and the authors name developing a more explainable compression architecture for efficient motion control as future work. The ablations also expose an inherent trade-off in the control stage: increasing the control loss weight lambda improves control metrics (trajectory error 0.1988 to 0.1465 from lambda=1 to 10) but degrades generation quality (FID 0.419 to 0.636), and the method is evaluated only on HumanML3D.
