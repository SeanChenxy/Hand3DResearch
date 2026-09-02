# FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space

**Authors:** Black Forest Labs (Stephen Batifol, Andreas Blattmann, Frederic Boesel, et al.)  
**Date:** 2025-06-24  
**Identifier:** [arXiv:2506.15742](https://arxiv.org/abs/2506.15742)  
**Zotero item:** `SNLD3SRE` ([Zotero](zotero://select/library/items/SNLD3SRE))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; summary content is derived from the paper with in-text caveats where detail is unavailable.  
## Summary
FLUX.1 Kontext unifies text-to-image generation and image editing in one flow-matching model. It concatenates image context and text instructions as sequences in latent space, allowing the same network to perform local and global edits, reference-based generation, and text-only generation. The paper introduces KontextBench and reports strong single-turn quality, multi-turn consistency, and character preservation, with 1024 × 1024 generation reported at 3–5 seconds. The available evidence does not establish performance beyond the evaluated resolution or enumerate all failure cases.

## Background and Problem
Existing editing systems can lose the identity of a referenced character or object across repeated edits, while separate generation and editing models complicate deployment. The task accepts an optional context image and a text instruction and outputs either an edited image or, when no context image is supplied, a generated image. The evaluated settings include local editing, global editing, character reference, style reference, and text editing.

## Method
The model predicts a velocity field with flow matching. Image latents from the optional context image and text or instruction tokens are concatenated so that self-attention can use both visual and linguistic context. A convolutional autoencoder maps images to a 16-channel latent representation. The architecture uses double-stream blocks for image and text processing followed by single-stream blocks, while the same network handles context-conditioned editing and text-only generation.

## Contributions
- A unified flow-matching architecture for image generation and in-context editing.
- Sequence-level conditioning that carries visual context and text instructions through the same latent model.
- KontextBench, a benchmark covering five image-generation and editing task categories.

## Experimental Setup
KontextBench contains 1,026 image–prompt pairs across the five task categories. The paper compares FLUX.1 Kontext with state-of-the-art image generation and editing systems and examines both single-turn quality and multi-turn consistency. The custom autoencoder is reported with PDist = 0.332 and SSIM = 0.896, compared with PDist = 0.452 and SSIM = 0.858 for SD3-VAE and PDist = 0.890 and SSIM = 0.748 for SDXL-VAE.

## Results
For 1024 × 1024 outputs, the paper reports generation times of 3–5 seconds. FLUX.1 Kontext achieves competitive state-of-the-art quality and is particularly strong at preserving characters across iterative edits. The evidence available here does not provide the complete per-task benchmark scores.

## Limitations
The reported evaluation centers on 1024 × 1024 images and does not establish behavior at substantially higher resolutions. The paper evidence available here does not provide a systematic failure-mode analysis or detailed training-resource requirements. Multi-turn consistency remains an evaluated capability rather than a guarantee for arbitrary edits.
