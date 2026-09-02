# Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction

**Authors:** Hao Tian, Chenyangguang Zhang, Rui Liu, Wen Shen, Xiaolin Qin  
**Date:** 2025-11-18  
**Identifier:** [arXiv:2511.14540](https://arxiv.org/abs/2511.14540)  
**Zotero item:** `9VF7LAXK` ([Zotero](zotero://select/library/items/9VF7LAXK))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

This paper reconstructs dynamic hand-object interaction (HOI) scenes — geometry and appearance together — from RGB egocentric video without any object priors, using a 4D Gaussian Splatting framework. It decomposes the scene into hand, object, and background Gaussian sets with separate implicit deformation fields, introduces interaction-aware Gaussians with two new learnable parameters (a weight w for occlusion/smoothing and a radius o for edge sharpness), feeds hand positions into the object deformation field, and applies explicit hand/object/interaction 3D losses plus a five-phase progressive optimization. On HOI4D it reaches 30.32 dB PSNR in translation scenes (+9% over the best dynamic 3D-GS baseline) and 24.16 dB in translation-rotation scenes, and on HO3D it attains 25.19 dB PSNR, outperforming dynamic 3D-GS baselines and HOI-specific methods HOLD and BIGS.

## Background and Problem

Accurate HOI reconstruction matters for VR and robotics, but grasping and manipulation involve complex contact dynamics and severe mutual occlusions. Previous approaches either require costly object priors such as poses or templates (multi-view fitting methods, EgoGaussian, BIGS), or reconstruct geometry without appearance (SDF-based methods). NeRF-style implicit fields provide both geometry and appearance but are slow due to backward-mapping ray rendering. Dynamic 3D-GS methods (4DGS, Deform3DGS, SC-GS) are fast and high-fidelity, yet they model the whole scene with a single unified field and 2D supervision, which in HOI settings leads to collapsed clearances, blurred contact boundaries, and non-physical merging of hand and object surfaces under drastic motion, irregular rotation, and heavy occlusion. The target setting is: given a monocular/egocentric RGB video, reconstruct the complete HOI scene at arbitrary timestamps from arbitrary views, with no object shape, category, or bounding-box prior.

## Method

The scene is decomposed into three Gaussian groups with dedicated deformation fields. Hand Gaussians are warped by a hand implicit field taking canonical position and timestamp with positional encoding (plus a noise-smooth term to avoid over-smoothing). Object Gaussians are warped by an object field that additionally takes key-frame hand positions — concatenated with object positions at the timestamp just before hand-object contact — so the deformation is explicitly conditioned on the hand that drives it; the background uses Deform3DGS-style deformation. The key representation is the interaction-aware Gaussian, which extends standard 3D-GS with two optimizable parameters: weight w, which balances motion smoothness and noise reduction (small w marks weak structure or occlusion), and radius o, which controls edge sharpness (small o yields sharper contours); their combination suppresses edge blur between interacting hand, object, and background. Supervision goes beyond 2D photometric losses with three explicit 3D regularizations derived from a lightweight off-the-shelf hand tracker's MANO predictions (< 3% of total runtime): a hand loss (Chamfer distance pulling hand Gaussians onto MANO vertices), a hand-guided object rotation loss (aligning object Gaussian rotations with the SVD-averaged MANO joint rotation prior, gated by a contact-aware sigmoid of w to act only during contact, in the SO(3) logarithmic map), and an interaction loss (bidirectional Chamfer distance between hand and object Gaussians to enforce proper grasping proximity, with a separate penetration loss penalizing overlapping or overly close hand-object Gaussians). Optimization is progressive in five phases: initialization (MANO vertices for hands; object Gaussians randomly sampled within an expanded AABB of MANO vertices; background from SfM sparse points), warm-up with the 3D losses and periodic density adjustment, HOI refinement (each Gaussian learns global importance w and local radius o, and its pose is updated via an LBS-inspired linear blend over K nearest neighbors' field-predicted rigid transformations), background pretraining, and final collaborative reconstruction of all fields into a shared target space.

## Contributions

- An interaction-aware hand-object Gaussian representation with new learnable parameters (weight and radius) adopting a piecewise-linear hypothesis, addressing mutual occlusion and edge blur without object priors.
- Interaction-aware dynamic fields: incorporating key-frame hand information into the object deformation field to model the flexible, hand-driven motions of grasped objects, where single-field or hand-independent formulations lose fine-grained motion.
- A progressive optimization strategy that handles dynamic regions and static background step by step, together with explicit 3D interaction losses requiring only a lightweight hand tracker.
- State-of-the-art results against dynamic 3D-GS baselines (4DGS, Deform3DGS, SC-GS) and HOI-specialized methods (HOLD, BIGS) on HOI4D and HO3D, with demonstrated robustness to noisy object initialization.

## Experimental Setup

Evaluation uses HOI4D (RGB-D egocentric videos with frame-level hand-object poses and masks; two purely translational and two translation-rotation scenes at official resolution) and HO3D (camera 4, four translation-rotation sequences at half resolution, providing real-world 3D pose annotations for actions like pickup and rotation). Comparisons follow the EgoGaussian protocol with alternate-frame testing to assess extrapolation, reporting PSNR, SSIM, and LPIPS for pure translation and translation-rotation settings, plus full-frame evaluation variants; on HO3D, HOLD and BIGS are additionally compared, with foreground-only metrics marked separately. Baselines run with official code. All experiments use an NVIDIA RTX 3090 with 21,000 iterations (1h20m training).

## Results

- HOI4D, translation scenes: 30.32 PSNR / 0.93 SSIM / 0.29 LPIPS (full-frame: 33.03 / 0.95 / 0.27), versus 4DGS 24.86 / 0.80 / 0.47, Deform3DGS 26.33 / 0.87 / 0.29, SC-GS 25.08 / 0.84 / 0.46 — a stated +9% PSNR gain.
- HOI4D, translation-rotation scenes: 24.16 / 0.86 / 0.37 (full-frame: 24.02 / 0.85 / 0.39) versus 4DGS 23.68, Deform3DGS 23.57, SC-GS 17.32 PSNR.
- HO3D, translation-rotation: 25.19 / 0.89 / 0.15, beating SC-GS (20.37), 4DGS (19.44), HOLD (18.03), Deform3DGS (9.68, non-convergent under pose noise), and BIGS (3.85, which reconstructs only foreground without background); in the foreground-only comparison, the method reaches 28.16 PSNR versus BIGS's 24.51.
- Ablations on HOI4D-Scene 1 (full model 32.96 / 0.95 / 0.35): removing the interaction-aware module costs 4.20 dB; removing HOI refinement, object loss, hand loss, or interaction loss costs 0.51-1.51 dB PSNR (stated relative degradations of 2.2%/1.1%/11.4% for refinement; 4.6%/1.1%/8.6% for object loss; 1.5%/-/5.7% for hand loss; 3.5%/1.1%/14.3% for interaction loss). Adding Gaussian initialization noise (sigma = 0.01 and 0.05) costs only 0.16-0.24 dB, indicating robustness to imperfect object initialization.

## Limitations

The authors note that the progressive optimization workflow consists of multiple stages that could ideally be unified as stronger optimizers emerge. The method struggles with extreme cases such as exceedingly rapid motion and complex trajectories, which the authors suggest could be addressed by integrating more interaction priors. The evaluation covers a small number of scenes per dataset (four on HOI4D and four on HO3D), and identity/metric comparisons on HO3D are affected by annotation pose errors that degrade all methods; quantitative geometry accuracy (e.g., mesh or depth errors) is not reported in the paper — evaluation is by rendering metrics and qualitative comparisons.
