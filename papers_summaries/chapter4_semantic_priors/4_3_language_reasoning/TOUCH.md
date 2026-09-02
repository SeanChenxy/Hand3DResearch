# TOUCH: Text-guided Controllable Generation of Free-Form Hand-Object Interactions

**Authors:** Guangyi Han, Wei Zhai, Yuhang Yang, Yang Cao, Zheng-Jun Zha  
**Date:** 2025-10-16  
**Identifier:** [arXiv:2510.14874](https://arxiv.org/abs/2510.14874)  
**Zotero item:** `9YRBFP7U` ([Zotero](zotero://select/library/items/9YRBFP7U))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

TOUCH introduces Free-Form HOI Generation: producing controllable, diverse, physically plausible hand-object interactions conditioned on fine-grained text intent, going beyond the grasp-centric paradigm to non-grasping actions like pushing, poking, and rotating. The method is a three-stage pipeline — contact-map prediction with two CVAEs, a multi-level coarse-to-fine conditioned Transformer diffusion model, and a physical-constraints refinement module with self-supervised contact cycle-consistency. It is trained on WildO2, a new in-the-wild 3D HOI dataset built from internet videos with 4.4k interactions across 92 intents and 610 object categories. On WildO2 it surpasses ContactGen and Text2HOI on contact accuracy (P-IoU 0.776), penetration (PV 2.67), diversity, and semantic consistency.

## Background and Problem

Existing HOI generation is confined to fixed grasping patterns: control is tied to physical priors such as force closure or coarse verb-noun instructions, and even LLM-based language conditioning inherits designs whose inductive biases favor stable grasps. This sacrifices the diversity of daily interactions — varied hand poses, contact details, and nuanced semantic intent (tipping a bottle, poking it, rotating a tap handle). The paper defines Free-Form HOI Generation with two core challenges: spatial plausibility (escaping restrictive grasping priors on palm position, orientation, and contact regions while remaining physically valid) and semantic controllability (mapping fine-grained text to specific hand configurations and contact regions). A further obstacle is data: existing 3D HOI datasets (HOI4D, OakInk, GRAB) are lab-collected with limited diversity, while in-the-wild videos lack 3D annotations and suffer severe hand occlusion of objects during reconstruction.

## Method

TOUCH has three stages. (1) Contact Map Prediction: two CVAEs generate binary contact maps for object and hand surfaces; the object branch encodes a 3000-point cloud with PointNet plus scale, and the hand branch encodes a 778-point MANO zero-pose cloud combined with a hand-part mask derived from fine-grained text; both are conditioned on Qwen-7B text features and trained with focal, dice, and KL losses. (2) Multi-Level Conditioned Diffusion: a Transformer-based DDPM directly predicts denoised hand pose parameters. Conditions are injected hierarchically across 8 Transformer blocks: early blocks (i < 4) receive global context (global geometry, scale, coarse SSC text) via FiLM, while later blocks switch to fine-grained conditions (DSC text plus local features of 128 object and 64 hand contact-region points) via cross-attention; each condition is dropped with 10% probability during training. Auxiliary losses supervise global rotation/translation and a joint-to-object distance map. (3) Physical Constraints Refinement: a refiner network first corrects global pose drift in one forward pass, then test-time optimization fine-tunes local contact using physical losses (contact, penetration, anatomy) plus a self-supervised cycle-consistency loss enforcing that hand-to-object and object-to-hand contact mappings compose to identity.

## Contributions

1) The Free-Form HOI Generation task, extending HOI synthesis from constrained grasping to diverse daily interactions. 2) TOUCH, a framework generating natural, physically plausible, diverse free-form HOI under fine-grained text guidance via explicit contact conditioning and multi-level diffusion. 3) An automated pipeline and WildO2, an in-the-wild 3D daily-HOI dataset with multi-level annotations, enabling research beyond laboratory grasp settings.

## Experimental Setup

WildO2 is built by filtering Something-Something V2 into 8k single-hand, single-object clips; an O2HOI pairing strategy extracts an unoccluded object-only frame and an interaction frame per clip; image-to-3D reconstruction, camera alignment via differentiable rendering, and hand-object refinement yield 4,414 high-quality 3D samples after manual inspection (about 55% reconstruction success rate). Annotations exceed 44k, including meshes, contact maps, template-based SSCs, VLM-generated DSCs (manually verified), and 17-part hand segmentation covering dorsal-side contact. The split is roughly 3.7k train / 677 test (4:1 per hand-part contact category, with long-tail aggregation and resampling). Training: 1000 epochs, Adam, learning rate 1e-4, batch size 128; the refiner trains with the diffusion model frozen. Evaluation covers contact accuracy (P-IoU, P-F1), physical plausibility (MPVPE, penetration depth PD, penetration volume PV), diversity, and semantic consistency (point-cloud FID, VLM-assisted scoring, 10-user perceptual score). Baselines: ContactGen and a temporally adapted Text2HOI, both augmented with optimization-based post-processing for fairness.

## Results

TOUCH achieves P-IoU 0.776, P-F1 0.844, MPVPE 2.97, PD 0.932, PV 2.67, entropy 2.93, cluster size 5.40, P-FID 4.13, VLM score 7.1, and perceptual score 8.8, versus ContactGen (P-IoU 0.620, P-FID 6.08, PS 6.3) and Text2HOI (P-IoU 0.711, P-FID 15.72, PS 7.5). Ablations show removing hand-object contact guidance drops P-IoU to 0.492, removing the multi-level structure to 0.525, and removing the refiner to 0.513 (its low PD/PV is deceptively favorable because the hand drifts away from the object); removing the cycle-consistency loss lowers P-IoU to 0.702, and removing either text level also degrades contact metrics. Text-encoder comparisons indicate Qwen-7B captures fine-grained semantics better than CLIP, BERT, or MPNet. Qualitatively, the same object yields distinct plausible poses under different contact/intent specifications.

## Limitations

The authors state the framework currently focuses on static HOI snapshots, which inherently limits capturing the temporal dynamics of interaction processes, and that the dataset scale remains an area for future growth; they plan to extend to dynamic sequences using large-scale video datasets and 6-DoF object pose estimation. The data pipeline also has a practical bound: reconstruction from in-the-wild images succeeds only about 55% of the time, with geometric reconstruction failure as the primary obstacle.
