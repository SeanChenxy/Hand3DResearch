# GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

**Authors:** Alex Nichol*, Prafulla Dhariwal*, Aditya Ramesh*, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, Mark Chen (OpenAI)  
**Date:** 2021-12 (arXiv v1); 2022-03-08 (arXiv v3); published at ICML 2022  
**Identifier:** [arXiv:2112.10741](https://arxiv.org/abs/2112.10741)  
**Zotero item:** `YU2NKL42` ([Zotero](zotero://select/library/items/YU2NKL42))  
**Evidence status:** Zotero metadata and the full paper PDF (main sections through safety considerations, with result tables) were verified.

## Summary

GLIDE (Guided Language-to-Image Diffusion for Generation and Editing) is a 3.5-billion-parameter text-conditional diffusion model operating at 64x64 resolution, paired with a 1.5-billion-parameter text-conditional upsampling diffusion model that raises outputs to 256x256, trained on the same dataset as DALL-E. The paper's central contribution is a systematic comparison of two guidance strategies for steering diffusion models toward text prompts: CLIP guidance, which perturbs the diffusion sampling mean with gradients from a noise-aware CLIP model, and classifier-free guidance, which interpolates between conditional and unconditional predictions of the same model. Human evaluators strongly prefer classifier-free guidance for both photorealism and caption similarity, and prefer GLIDE samples over DALL-E outputs 87% of the time for photorealism and 69% for caption similarity, even against CLIP-re-ranked DALL-E — despite GLIDE requiring no expensive CLIP re-ranking. The model is also fine-tuned to perform image inpainting, enabling natural-language-driven image editing that matches the style and lighting of the surrounding context. A smaller 300M-parameter model trained on filtered data is released with code and weights.

## Background and Problem

Text-conditional image generation models (DALL-E, CMA3, GAN-based systems) synthesize plausible images from free-form prompts but fall short of photorealism and fail to capture all aspects of their text prompts. Class-conditional diffusion models with classifier guidance achieve state-of-the-art sample quality, and an open question is how best to transfer this guidance machinery to free-form text conditioning. Two candidate strategies exist — classifier guidance with a separately trained CLIP model, and classifier-free guidance trained into the diffusion model itself — but they had not been rigorously compared for text-to-image generation. Additionally, a model capable of both zero-shot generation and iterative editing (inpainting with text prompts) was needed to support real-world creative workflows, where humans iteratively refine images.

## Method

Architecture: the base model follows the ADM (guided diffusion) architecture from ImageNet 64x64 with 512 channels (~2.3B visual parameters). Text conditioning uses a 24-residual-block Transformer encoder with width 2048 (~1.2B parameters) that encodes the caption into K tokens; the final token embedding serves as the class embedding, and the last layer's token embeddings are projected and concatenated to the attention context at each layer of the diffusion U-Net. The text encoder is width 1024 (versus 2048 in the upsampling model), and the base upsample architecture matches Dhariwal & Nichol's ImageNet upsampler with base channels increased to 384. The 1.5B upsampling diffusion model is conditioned like the base and additionally receives the full low-resolution image.

Training: the base model trains 2.5M iterations at batch size 2048, the upsampler 1.6M iterations at batch size 512, in 16-bit precision on the DALL-E dataset (~250M text-image pairs), with total compute roughly equal to DALL-E's.

Guidance: CLIP guidance perturbs the reverse-process mean with the gradient of the dot product between image and caption encodings, using a CLIP model explicitly trained on noised images (public CLIP models degrade sample quality because noised intermediates are out-of-distribution). Classifier-free guidance fine-tunes the model by randomly replacing 20% of text token sequences with an empty caption, then guiding toward the caption via the extrapolation between unconditional and conditional noise predictions with scale s >= 1.

Inpainting: the model is explicitly fine-tuned for inpainting — random regions of training images are erased, and the architecture gains four additional input channels (a second RGB set plus a mask channel) with zero-initialized weights. The upsampler always receives the full low-resolution image but only the unmasked high-resolution region, preventing edge artifacts that occur when re-noising completed samples each step.

Safety: to mitigate misuse (disinformation, deepfakes, harmful biases), training data gathered from the internet (several hundred million images) is filtered to reduce people-centric and violent/hateful content, and a smaller 300M-parameter GLIDE (filtered) is released with code and weights.

## Contributions

1. A 3.5B-parameter text-conditional diffusion model plus 1.5B upsampler achieving photorealistic text-to-image generation with compute on par with DALL-E.
2. A controlled comparison of CLIP guidance versus classifier-free guidance for text conditioning, showing classifier-free guidance is preferred by human evaluators for photorealism and caption similarity and produces a nearly Pareto-optimal FID/IS trade-off.
3. A fine-tuning recipe that turns the generator into a text-driven image editor via inpainting, supporting iterative scene creation and style-matched edits (including combinations with SDEdit).
4. Noised CLIP models that make CLIP guidance effective on diffusion sampling trajectories, plus a responsible release: a filtered-data 300M model with public code and weights.

## Experimental Setup

Zero-shot generation is evaluated on MS-COCO 256x256 (30K captions), with human evaluators judging photorealism and caption similarity via Elo scores and win probabilities against DALL-E (at temperatures 1.0 and 0.85, with and without CLIP re-ranking, and with dVAE blurring to isolate sample-quality effects). Automated metrics include FID and zero-shot FID against the full validation set and a DALL-E-filtered validation subset (reducing batch by 21%), Inception Score, Precision/Recall, and CLIP score, sweeping guidance scales to trace quality-diversity Pareto frontiers. Editing is evaluated qualitatively: object insertion and replacement in natural images and paintings, style and lighting preservation, iterative scene construction from a single prompt, and combined SDEdit workflows. Guidance scales for the human study are separately optimized per method (3.0 for classifier-free, 2.0 for CLIP guidance).

## Results

- Human evaluation (Elo, MS-COCO 256x256): classifier-free guidance 82.7 photorealism / 110.9 caption similarity versus CLIP guidance -73.2 / 29.3 and unguided -88.6 / -106.2 — humans disagree with the CLIP score and favor classifier-free guidance.
- Versus DALL-E: GLIDE is preferred 91% (photorealism) / 83% (caption similarity) at temperature 1.0 without re-ranking, 84% / 80% at 0.85; against CLIP-re-ranked DALL-E still 89% / 71% and 87% / 69%; even when DALL-E samples are blurred via dVAE, GLIDE wins 72% / 63% — although heavy DALL-E blurring narrows the photorealism gap.
- FID (MS-COCO 256x256): GLIDE 12.24 on the DALL-E-filtered validation subset and 12.89 on the full validation set (the increase largely explained by the 21% batch reduction), versus LAFITE 8.12 (trained on MS-COCO), XMC-GAN 9.33, DM-GAN+CL 20.79, and DALL-E ~28.
- Quality-diversity trade-off: increasing guidance scale cleanly trades FID vs IS, Precision vs Recall, and CLIP score vs FID; classifier-free guidance is nearly Pareto optimal in FID vs IS, while CLIP guidance boosts CLIP score further — attributed to adversarial examples for the evaluation CLIP model rather than genuine prompt matching.
- Editing: the inpainting fine-tune inserts, removes, and replaces objects with matching style, shadows, and lighting; supports iterative zero-shot scene construction (e.g., generating a living room then adding a painting, a table, and a vase through successive prompts) and combination with SDEdit for sketch-conditioned edits.

## Limitations

GLIDE can struggle with complex prompts, particularly binding multiple objects and attributes correctly, since zero-shot generation must produce realistic images for arbitrary compositions. Human evaluation of editing is qualitative — there is no standardized metric for edit fidelity, and edits are limited to inpainting-style operations rather than arbitrary semantic transformations. CLIP guidance requires an expensive separately trained noise-aware CLIP model, and even then its gains on automated metrics partly reflect adversarial examples rather than true alignment, complicating metric-based comparison. The model operates at 64x64 with a learned upsampler, so fine detail is bounded by the two-stage pipeline, and total training compute equals DALL-E's (~hundreds of petaflop-days), which is out of reach for most labs. Finally, the authors note substantial misuse risk — convincing fake imagery and perpetuation of dataset biases — which they address only through data filtering and the release of a smaller filtered model rather than a full solution.
