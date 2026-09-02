# AttentionHand: Text-driven Controllable Hand Image Generation for 3D Hand Reconstruction in the Wild

**Authors:** Junho Park, Kyeongbo Kong, Suk-Ju Kang  
**Date:** 2024-07-25 (ECCV 2024)  
**Identifier:** [arXiv:2407.18034](https://arxiv.org/abs/2407.18034)  
**Zotero item:** `I7E47EKW` ([Zotero](zotero://select/library/items/I7E47EKW))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; the summary was written without full-text extraction, and unavailable details are marked as not reported.  
## Summary
AttentionHand addresses the shortage of in-the-wild images that carry reliable 3D hand supervision. It uses text-driven controllable image generation to produce diverse hand images together with aligned 3D hand labels, and filters the generated samples before using them for reconstruction training. The resulting synthetic data is intended to reduce the domain gap between indoor training imagery and outdoor or in-the-wild observations. The paper reports improved in-the-wild 3D hand reconstruction, but the available evidence does not provide a complete numerical result table.

## Background and Problem
Single-image 3D hand reconstruction is difficult in the wild because self-occlusion, viewpoint variation, appearance changes, and ambiguous depth are not adequately covered by many existing labeled datasets. The paper considers a text prompt describing a hand pose or scene as input and generates a hand image with corresponding 3D hand annotations as output. The downstream task explicitly evaluated is 3D hand reconstruction from images; object interaction is outside the reported problem definition.

## Method
AttentionHand uses a text-conditioned image generator to control the pose and scene content of synthetic hand images. A quality-control or filtering procedure removes samples whose visual content and 3D labels are not sufficiently aligned. The retained image-label pairs are then added to training data for a 3D hand reconstruction model, transferring the generator's visual diversity to the reconstruction task.

## Contributions
- A text-driven pipeline for generating controllable hand images with 3D labels.
- A filtering and validation process for improving the reliability of generated training pairs.
- A synthetic-data training strategy targeting in-the-wild generalization of 3D hand reconstruction.

## Experimental Setup
The paper evaluates generated hand imagery and its use for 3D hand reconstruction on standard hand-reconstruction benchmarks. The available evidence does not specify all benchmark names, data splits, baseline configurations, or metric values. It identifies image quality, diversity, and 3D reconstruction accuracy as evaluation concerns, but the exact protocol is not reported in the paper evidence available here.

## Results
The paper reports that the generated images are controllable and aligned with 3D hand labels, and that incorporating them improves reconstruction in the wild while reducing the indoor-to-outdoor domain gap. Representative numerical comparisons and ablation values are not reported in the available evidence.

## Limitations
The approach is limited to hand-image generation and does not establish an object-conditioned hand-object generator. Its effectiveness depends on the quality of the underlying text-to-image model and on filtering generated samples. Coverage of rare poses, appearances, or severe occlusions beyond the evaluated setting is not reported in the paper.
