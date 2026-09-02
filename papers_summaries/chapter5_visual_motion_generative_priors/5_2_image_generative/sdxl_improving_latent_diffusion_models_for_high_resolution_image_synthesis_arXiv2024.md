# SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

**Authors:** Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, Robin Rombach  
**Date:** 2024 (ICLR)  
**Identifier:** [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)  
**Zotero item:** `JEWCR9NG` ([Zotero](zotero://select/library/items/JEWCR9NG))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; summary content is derived from the paper with in-text caveats where detail is unavailable.  
## Summary
SDXL improves open latent diffusion for high-resolution text-to-image synthesis while retaining an openly inspectable model. It scales the U-Net, adds resolution and crop micro-conditioning, uses two text encoders, and separates base generation from high-resolution refinement. On ImageNet class-conditional evaluation, size conditioning improves the reported FID and IS over an otherwise matched condition-free setup, and user studies favor SDXL over earlier Stable Diffusion versions. The paper also reports competitive qualitative quality with proprietary systems.

## Background and Problem
Earlier open latent diffusion models had quality gaps, cropping artifacts, and weaker text or resolution control compared with closed systems. SDXL takes a text prompt and generates a high-resolution image through a latent diffusion pipeline; the refinement stage can further improve the result. The paper also evaluates class-conditional synthesis to study architectural and conditioning choices with conventional metrics.

## Method
The main denoiser has a 2.6-billion-parameter U-Net, approximately three times the size of earlier Stable Diffusion models, with heterogeneous transformer-block placement. Micro-conditioning embeds the original training image size and crop coordinates so the model can learn resolution-aware behavior without additional labels. Dual CLIP text encoders provide token and pooled text representations. A base model generates latent images and a dedicated refinement model applies a noising–denoising step for higher fidelity.

## Contributions
- A substantially scaled latent-diffusion backbone for high-resolution synthesis.
- Size and crop micro-conditioning that improves use of varied-resolution training data and reduces cropping artifacts.
- A dual-text-encoder base-plus-refiner pipeline for open high-resolution generation.

## Experimental Setup
The paper reports ImageNet class-conditional evaluation at 512² with FID and IS, as well as user studies comparing SDXL with Stable Diffusion 1.5 and 2.1. It examines size conditioning, crop conditioning, transformer-block placement, and the refinement stage. Under the reported ImageNet settings, size conditioning gives FID 36.53 and IS 215.34, compared with FID 39.76 and IS 211.50 for the no-conditioning baseline and FID 43.84 and IS 110.64 for 512-only training.

## Results
User studies consistently favor SDXL over Stable Diffusion 1.5 and 2.1, with additional gains from refinement. The ImageNet results show that size conditioning improves both FID and IS over the no-conditioning baseline under the stated protocol. Qualitative comparisons report fewer cropping artifacts and competitive quality with Midjourney.

## Limitations
FID and IS do not fully capture the quality of foundation text-to-image models, as acknowledged by the paper. The larger 2.6-billion-parameter model increases computational cost, and the highest quality requires a two-stage pipeline. The reported scope emphasizes architectural and conditioning changes, with limited evidence about alternative training methodologies.
