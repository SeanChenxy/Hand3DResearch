# Reconstructing Hands in 3D with Transformers

**Authors:** Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, Jitendra Malik  
**Date:** 2024-06-16  
**Identifier:** [arXiv:2312.05251](https://arxiv.org/abs/2312.05251); DOI `10.1109/CVPR52733.2024.00938`  
**Zotero item:** `XYAK3PBM` ([Zotero](zotero://select/library/items/XYAK3PBM))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HaMeR (Hand Mesh Recovery) applies the "simple model plus scale" recipe that proved successful in human mesh recovery to monocular 3D hand reconstruction: a fully transformer-based architecture (a ViT-Huge backbone with a transformer-decoder head) that regresses MANO pose, shape, and camera parameters, trained on 2.7M consolidated 2D/3D hand annotations from ten datasets, four times larger than FrankMocap's training set. HaMeR achieves state-of-the-art results on FreiHAND and HO3Dv2 and, more strikingly, 2-3x improvements in PCK@0.05 over previous methods on in-the-wild data. The paper also contributes HInt, a benchmark of 40.4K in-the-wild hands from YouTube and egocentric video annotated with 21 2D keypoints and per-joint occlusion labels, the first hand keypoint dataset with explicit occlusion annotations, which reveals that HaMeR's robustness advantage is far larger than controlled benchmarks suggest.

## Background and Problem

The paper starts from the observation that progress in vision and NLP increasingly comes from simple, high-capacity models trained on large data (GPT-3/4, CLIP, SAM, and HMR2.0 for human mesh recovery), and asks whether the same holds for 3D hand pose estimation. Prior hand methods either regress MANO parameters parametrically (e.g., FrankMocap), regress mesh vertices directly (e.g., METRO, MeshGraphormer, which align better with image evidence but are more prone to failure under occlusion and truncation), or design specialized components for specific subproblems (MobRecon for speed, HandOccNet for occlusion, AMVUR for probabilistic estimation, BlurHand for motion blur). Benchmarking is skewed: standard 3D benchmarks (FreiHAND, HO-3D, DexYCB, InterHand2.6M) are captured in controlled multi-camera settings, so they under-represent viewpoint, appearance, and interaction diversity. HaMeR defines its goal as accurate and robust reconstruction of the MANO hand surface (778 vertices, 21 joints) from monocular RGB, studying scaling of both data and architecture, and introduces in-the-wild evaluation to measure robustness that controlled benchmarks can no longer discriminate.

## Method

The formulation follows parametric hand recovery: given an RGB image I, a network learns f(I) = {theta, beta, pi}, where theta in R^48 and beta in R^10 are MANO pose and shape and pi is a camera translation t in R^3 used with fixed intrinsics K to project joints and mesh to the image, x = Pi_K(X + t). The architecture mirrors HMR2.0: the image is split into patches fed to a ViT-Huge backbone; a transformer-decoder head processes a single learned token while cross-attending to the ViT output tokens and outputs the MANO and camera parameters. Training combines three losses: a 3D loss on parameters and joints when 3D ground truth is available (squared errors on theta and beta plus L1 on 3D joints), an L1 reprojection loss against 2D keypoints applied to all data, and adversarial losses from discriminators trained on hand shape, hand pose, and each hand joint angle separately, which prevent unnatural poses that reproject well when only 2D supervision is available. The training set consolidates FreiHAND, HO3D, MTC, RHD, InterHand2.6M, H2O3D, DexYCB, COCO WholeBody, Halpe, and MPII NZSL into 2.7M examples, of which only 5% (the in-the-wild sets) carry 2D-only annotations. Although single-frame, HaMeR yields temporally smooth reconstructions on video without explicit temporal modeling.

## Contributions

- HaMeR, a fully transformer-based (ViT-H + transformer decoder) parametric hand mesh recovery model demonstrating that scaling training data (4x more examples than FrankMocap) and model capacity together yields large, consistent accuracy and robustness gains, with the two factors shown to be complementary in ablation.
- State-of-the-art 3D results on FreiHAND and HO3Dv2 and 2-3x PCK@0.05 improvements over previous methods on in-the-wild images, including heavily occluded and interacting hands, different skin tones, gloves, artwork, and even mechanical hands.
- HInt (Hand Interactions in the wild), a dataset of 2D keypoint annotations for 40.4K hands from Hands23/New Days (12.0K), Epic-Kitchens VISOR (5.3K), and Ego4D FHO critical frames (23.2K), with 86.7% of hands in natural contact, and the first hand keypoint dataset with per-joint occlusion labels, enabling fine-grained analysis of occlusion robustness.
- Public release of code, data, and models to support future hand reconstruction work.

## Experimental Setup

3D evaluation follows standard protocols on FreiHAND and HO3Dv2, reporting PA-MPJPE and AUCJ for joints and PA-MPVPE, AUCV, F@5mm, and F@15mm for meshes, against parametric and non-parametric baselines including I2L-MeshNet, Pose2Mesh, I2UV-HandNet, METRO, Tang et al., MeshGraphormer, MobRecon, AMVUR, HandOccNet, ArtiBoost, Keypoint Transformer, and Liu et al. In-the-wild 2D evaluation on HInt reports PCK at thresholds 0.05/0.1/0.15 on the projected 3D joints, separately for the New Days, VISOR, and Ego4D subsets and separately for all, visible-only, and occluded-only joints, against FrankMocap, METRO, MeshGraphormer, and HandOccNet. Ablations isolate training-data scale (ResNet-50 base design with a quarter of the data, versus the full 2.7M set) and architecture (ResNet-50 versus ViT-H), and a separate experiment adds the HInt training split to training. HInt annotation quality was checked by double-annotating 90 images: 90.5% occlusion-label and 100% existence-label agreement, with 94.6% of visible keypoints within 0.25 palm lengths.

## Results

- FreiHAND: PA-MPJPE 6.0 mm, PA-MPVPE 5.7 mm, F@5 0.785, F@15 0.990, best PA-MPVPE and F-scores among compared methods (MobRecon retains a slightly better PA-MPJPE of 5.7 mm).
- HO3Dv2: AUCJ 0.846, PA-MPJPE 7.7 mm, AUCV 0.841, PA-MPVPE 7.9 mm, F@5 0.635, F@15 0.980, ahead of AMVUR (0.835/8.3 mm) and all other baselines.
- HInt (all joints, PCK@0.05): HaMeR reaches 48.0 on New Days, 43.0 on VISOR, and 38.9 on Ego4D, versus 16.8, 19.1, and 14.6 for MeshGraphormer and roughly 9-17 for the other baselines; on occluded joints HaMeR scores 27.2/25.9/23.0 versus about 7-11 for all baselines, i.e., a roughly 2-3x improvement.
- Scaling ablation on HInt (all joints, New Days PCK@0.05): a FrankMocap-style ResNet-50 base design scores 16.9; adding the 4x larger training set raises it to 31.3; switching to ViT-H with the small set yields 25.9; combining both (full HaMeR) reaches 48.0, showing both ingredients are needed and complementary.
- Training with HInt's own training split (Ours*) further improves PCK, especially on egocentric subsets (VISOR all-joints 43.0 to 56.5; Ego4D occluded 23.0 to 33.1), reflecting the previous scarcity of annotated egocentric hand data.

## Limitations

The paper does not include a dedicated limitations section, but the evaluation scope makes several constraints explicit. HInt provides only 2D keypoint supervision, so in-the-wild evaluation measures reprojection accuracy rather than true 3D accuracy, and the authors present it as complementary to 3D benchmarks rather than a replacement. On the controlled FreiHAND and HO3Dv2 benchmarks, margins over prior work are small and the authors themselves describe performance there as saturated, meaning the headline gains come mainly from the in-the-wild setting. HaMeR regresses MANO parameters, so its reconstruction fidelity is bounded by the parametric hand model, which the related-work discussion notes recovers less image-aligned detail than non-parametric vertex regression. The model is single-frame with no temporal component, so video stability is an emergent property rather than a designed guarantee. Finally, the ViT-Huge backbone entails a large computational footprint that the paper does not optimize or quantify, in contrast to efficiency-focused methods such as MobRecon.
