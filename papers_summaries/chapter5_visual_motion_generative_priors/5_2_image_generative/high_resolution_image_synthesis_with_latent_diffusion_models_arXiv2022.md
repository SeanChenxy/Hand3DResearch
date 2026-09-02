# High-Resolution Image Synthesis with Latent Diffusion Models

**Authors:** Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer  
**Date:** 2021-12-20  
**Identifier:** [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)  
**Zotero item:** not in the Zotero snapshot (repository-only prior-source card)  
**Evidence status:** Identity verified against Zotero/arXiv metadata; summary content is derived from the paper with in-text caveats where detail is unavailable.  
## Summary
Latent Diffusion Models (LDMs) reduce the cost of high-resolution diffusion by moving denoising from pixels into the latent space of a perceptual autoencoder. A U-Net learns the semantic distribution in that compact spatial representation, while cross-attention supports text and other token-based conditions. The paper reports competitive or state-of-the-art image synthesis across class-conditional, text-to-image, inpainting, super-resolution, and layout-conditioned settings at substantially lower compute than pixel-space diffusion. It also reports that mild compression preserves more detail than highly compressed latent pipelines.

## Background and Problem
Pixel-space diffusion spends computation on high-frequency details that are not equally important for perceptual quality and can require hundreds of GPU-days. More aggressively compressed autoregressive or latent systems may lose spatial detail, while older conditioning mechanisms do not naturally support free-form text or layouts. During inference, LDM takes noise and an optional condition such as text, a class label, a mask, or a low-resolution image and generates a high-resolution RGB image.

## Method
The first stage trains a perceptual autoencoder that maps an RGB image to a mildly compressed spatial latent and reconstructs it with perceptual, adversarial, and regularization losses. The second stage trains a denoising U-Net directly on those latents with the standard diffusion objective. Cross-attention injects token sequences such as text or layouts, while concatenation handles tensor-valued conditions such as masks or low-resolution images. The decoder converts the final latent back to pixels.

## Contributions
- A two-stage perceptual-compression and latent-denoising formulation for efficient high-resolution diffusion.
- Cross-attention as a general interface for text and spatial or token-based conditioning.
- Evidence that mild spatial compression offers a favorable compute–fidelity trade-off across several synthesis tasks.

## Experimental Setup
Experiments cover ImageNet class-conditional synthesis, LAION-based text-to-image generation, LSUN unconditional generation, DIV2K or ImageNet super-resolution, LAION-Mask inpainting, and COCO layout-to-image settings. The reported metrics include FID, IS, PSNR, SSIM, and LPIPS, together with human preference studies for text-to-image. At compression factor 4, the paper reports reconstruction PSNR 27.4 and R-FID 0.58, versus PSNR 19.9 and R-FID 4.98 for a VQGAN comparison at factor 16.

## Results
The paper reports new state-of-the-art results for class-conditional ImageNet synthesis and image inpainting, competitive results for LSUN, super-resolution, and layout-to-image, and 2.5–10× lower training compute than comparable pixel-space diffusion models. The factor-4 reconstruction comparison supports the use of mild compression for retaining detail. Ablations examine compression, KL versus VQ regularization, conditioning type, and sampling steps.

## Limitations
The autoencoder is fixed before diffusion training, so information it discards cannot be recovered by the denoiser. Text-to-image quality is bounded by the text encoder, and sampling remains sequential. The released training data and checkpoints inherit biases and unsafe content from web-scale data, and pixel-perfect reconstruction may still favor pixel-space models.
