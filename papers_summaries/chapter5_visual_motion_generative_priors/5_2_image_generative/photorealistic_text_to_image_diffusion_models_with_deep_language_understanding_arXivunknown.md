# Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

**Authors:** Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, et al.  
**Date:** 2022-05-23  
**Identifier:** [arXiv:2205.11487](https://arxiv.org/abs/2205.11487)  
**Zotero item:** not in the Zotero snapshot (repository-only prior-source card)  
**Evidence status:** Identity verified against Zotero/arXiv metadata; summary content is derived from the paper with in-text caveats where detail is unavailable.  
## Summary
Imagen studies whether stronger language understanding can improve text-to-image generation more effectively than simply enlarging the image model. It uses a frozen T5-XXL text encoder with a cascade of diffusion and super-resolution models, together with classifier-free guidance and dynamic thresholding. The paper reports a COCO FID of 7.27 without training on COCO and favorable human comparisons on image quality and text alignment. DrawBench is introduced to test challenging compositional prompts, where the large language encoder is especially useful.

## Background and Problem
Text-to-image generation requires both detailed language interpretation and high-fidelity image synthesis. Text encoders trained only on paired image–text data may have weaker linguistic knowledge than large language models trained on text-only corpora. Imagen takes a text prompt as input and outputs a photorealistic image, with a cascade that progressively increases resolution.

## Method
A frozen T5-XXL encoder maps the prompt to conditioning embeddings. A 64 × 64 base diffusion model generates a coarse image, followed by two super-resolution diffusion models. Classifier-free guidance strengthens prompt alignment, while dynamic thresholding limits saturation artifacts that can occur at high guidance weights. The design keeps the language encoder separate from image-generation training.

## Contributions
- Demonstration that a large text-only pretrained language model can provide an effective text-to-image conditioning signal.
- A cascaded base-plus-super-resolution diffusion system for photorealistic synthesis.
- DrawBench, a prompt suite for evaluating compositional text-to-image behavior.

## Experimental Setup
The model is evaluated on COCO without training on the COCO images used for evaluation and on the DrawBench prompt suite. Human raters compare Imagen with VQ-GAN+CLIP, Latent Diffusion Models, GLIDE, and DALL-E 2 for sample quality and text alignment. Ablations compare T5 with CLIP and examine dynamic thresholding and guidance strength.

## Results
Imagen reports FID 7.27 on COCO under the stated no-COCO-training protocol. Human evaluation favors Imagen over the listed alternatives for both image quality and alignment, and the T5-XXL encoder is reported to outperform CLIP on compositional prompts. The available evidence does not include all per-prompt or per-model scores.

## Limitations
High guidance weights can still cause saturation without thresholding. The model is computationally large, but complete training-resource details are not reported in the available evidence. The main evaluation emphasizes photorealistic content; coverage of other artistic domains is limited in the paper's central experiments.
