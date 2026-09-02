# RHanDS: Refining Malformed Hands for Generated Images with Decoupled Structure and Style Guidance

**Authors:** Chengrui Wang, Pengfei Liu, Min Zhou, Ming Zeng, Xubin Li, Tiezheng Ge, Bo Zheng  
**Date:** 2025-04-14  
**Identifier:** [arXiv:2404.13984](https://arxiv.org/abs/2404.13984)  
**Zotero item:** `35U82QNE` ([Zotero](zotero://select/library/items/35U82QNE))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; the summary was written without full-text extraction, and unavailable details are marked as not reported.  
## Summary
RHanDS targets malformed hands produced by otherwise visually plausible image generators. Given an image with a malformed hand, it uses a reconstructed hand mesh as structural guidance and the original hand appearance as style guidance, then generates a refined image. A two-stage training strategy and multi-style hand data are used to reduce interference between correcting anatomy and preserving appearance. The paper reports improved hand structure while retaining the input style, but the available evidence does not contain representative numerical scores.

## Background and Problem
Text-to-image diffusion models frequently fail on finger count, articulation, and self-occlusion even when the surrounding scene is satisfactory. The task takes a generated image containing a malformed hand and outputs an image with a more plausible hand structure while preserving the original visual style. The method is a hand-refinement system; an object-conditioned interaction output is not part of the reported task.

## Method
RHanDS extracts two complementary signals from the input. A hand mesh reconstructed from the malformed hand supplies structural information, while the original malformed hand supplies appearance and style information. A latent diffusion model conditions on both signals. The first training stage uses paired hand images to establish style consistency, and the second expands the training distribution with multiple hand styles to reduce structure–style interference.

## Contributions
- Decoupled structural and stylistic guidance for hand correction.
- A conditional diffusion refinement model that preserves appearance while changing hand structure.
- A two-stage, multi-style training strategy for reducing interference between the two guidance signals.

## Experimental Setup
The reported evaluation uses hand-image data, including paired training examples and multi-style data, and assesses structural accuracy, style preservation, and image quality. Exact dataset names, splits, baseline configurations, and numerical metric values are not reported in the available evidence.

## Results
RHanDS reports effective correction of malformed hands with preservation of the original style. The paper also reports that the two-stage training and multi-style data are important to the final behavior. Quantitative comparisons and ablation magnitudes are not available in the evidence used for this rewrite.

## Limitations
The method depends on a hand-mesh reconstruction of the input, so errors in that intermediate representation can affect refinement. Its reported scope is hand-only correction, and behavior for severe or atypical malformations is not established in the paper. The two-stage training also adds complexity relative to direct image generation.
