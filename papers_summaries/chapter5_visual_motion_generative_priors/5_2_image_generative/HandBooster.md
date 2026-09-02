# HandBooster: Boosting 3D Hand-Mesh Reconstruction by Conditional Synthesis and Sampling of Hand-Object Interactions

**Authors:** Hao Xu, Haipeng Li, Yinqiao Wang, Shuaicheng Liu, Chi-Wing Fu  
**Date:** 2024-03  
**Identifier:** [arXiv:2403.18575](https://arxiv.org/abs/2403.18575)  
**Zotero item:** `FCATUUFP` ([Zotero](zotero://select/library/items/FCATUUFP)), `U6N5AX2Z` ([Zotero](zotero://select/library/items/U6N5AX2Z))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; the summary was written without full-text extraction, and unavailable details are marked as not reported.  
## Summary
HandBooster addresses the limited pose, appearance, viewpoint, and background diversity of real hand-object datasets for single-image 3D hand-mesh reconstruction. It learns a conditional diffusion space for hand-object images and uses similarity-aware distribution sampling to seek novel but realistic interactions rather than drawing indiscriminately from the training distribution. The generated images retain known 3D hand supervision and are used to train reconstruction models. On HO3D and DexYCB, the paper reports improvements over prior reconstruction systems, including gains beyond the previous state of the art.

## Background and Problem
Single-image hand-mesh reconstruction must infer a 3D MANO hand from visual evidence that can be occluded or out of distribution. Existing real datasets do not cover enough interaction configurations, while synthetic data can exhibit a synthetic-to-real gap. HandBooster takes hand-object interaction conditions for image synthesis and uses the resulting labeled images to improve a downstream model that maps an image to a 3D hand mesh.

## Method
The method first trains a conditional generative model on hand-object interactions. Its conditions encode content factors such as hand appearance, pose, viewpoint, and background. A similarity-aware sampling strategy then constructs or selects conditions that are sufficiently different from existing samples while remaining plausible. The synthesized image-label pairs are mixed into downstream reconstruction training.

## Contributions
- A conditional generative space for diverse hand-object interaction images with usable 3D hand annotations.
- Similarity-aware sampling that targets novel and realistic interaction conditions.
- A data-augmentation pipeline that improves single-image 3D hand-mesh reconstruction.

## Experimental Setup
The downstream evaluation uses HO3D and DexYCB. The evaluation uses PA-MPJPE, PA-MPVPE, and F-score as representative hand-mesh metrics and compares several reconstruction baselines. The available evidence does not provide the exact split names, sample counts, or complete numerical table.

## Results
HandBooster reports significant improvements for multiple reconstruction baselines on HO3D and DexYCB, including performance beyond the previous state of the art. The reported analysis attributes the gains to increased data diversity and sampling that reduces the synthetic-to-real mismatch. Exact metric values and ablation magnitudes are not available in the paper evidence used here.

## Limitations
The pipeline couples a generative stage with downstream reconstruction training and therefore inherits errors from both stages. The sampling strategy requires task-specific design, and a residual synthetic-to-real gap may remain. Performance outside the evaluated hand-object datasets is not reported in the paper.
