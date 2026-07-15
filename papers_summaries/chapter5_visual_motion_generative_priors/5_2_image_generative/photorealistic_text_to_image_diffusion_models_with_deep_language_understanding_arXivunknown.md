# Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

## Summary
Imagen is a text-to-image diffusion model that achieves unprecedented photorealism by leveraging large pretrained language models (T5) as text encoders, demonstrating that scaling the language model improves image generation quality more than scaling the image diffusion model itself.

## 1. Problem and Setting
Text-to-image synthesis requires models to capture both complex natural language understanding and high-fidelity image generation. Existing approaches rely heavily on paired image-text data for text encoder training, which limits the semantic richness and scale of text understanding compared to large language models trained on text-only corpora.

## 2. Core Method
- **Architecture**: Frozen T5-XXL encoder maps text to embeddings, followed by a cascade of conditional diffusion models (64×64 base + two super-resolution models) for progressive resolution upscaling
- **Key Innovation**: Using generic large language models pretrained on text-only corpora as text encoders, rather than training on paired image-text data
- **Classifier-Free Guidance**: Jointly trains conditional and unconditional objectives by randomly dropping conditioning during training, enabling strong text-to-image alignment
- **Dynamic Thresholding**: Novel sampling technique that prevents saturation artifacts at high guidance weights by adaptively thresholding pixel values at each sampling step based on percentile statistics

## 3. Knowledge, Supervision, and Assumptions
- **Text Encoder**: T5-XXL (frozen, pretrained on text-only corpora), compared against BERT and CLIP
- **Training Data**: Internal large-scale image-text dataset (not COCO), with no COCO training used for evaluation
- **Architecture Assumptions**: Freezing text encoder weights enables offline embedding computation; scaling language model size is more impactful than scaling diffusion model size

## 4. Experiments and Findings
- **COCO Dataset**: Achieved state-of-the-art FID of 7.27 without COCO training; human raters found samples on par with real COCO data in image-text alignment
- **DrawBench**: Introduced comprehensive benchmark for text-to-image models with challenging compositional prompts
- **Human Evaluation**: Imagen preferred over VQ-GAN+CLIP, Latent Diffusion Models, GLIDE, and DALL-E 2 in side-by-side comparisons for both sample quality and alignment
- **Ablation**: T5-XXL outperformed CLIP on compositional prompts despite similar performance on simple COCO benchmark; dynamic thresholding enabled effective high guidance weight sampling

## 5. Strengths and Limitations
**Strengths**:
- Demonstrates that text-only pretrained LMs are highly effective for text-to-image synthesis
- Achieves SOTA photorealism without training on target evaluation dataset
- Dynamic thresholding enables stable sampling at high guidance weights

**Limitations**:
- Large guidance weights can still produce over-saturated images without thresholding techniques
- Model size and computational requirements not extensively discussed
- Evaluation limited to photorealistic generation in main paper; artistic content deferred to appendix

## 6. Takeaway
The key insight is that text encoders derived from large language models trained on text-only data transfer exceptionally well to text-to-image generation, with scaling the language model yielding greater gains than scaling the image diffusion model—suggesting that deep language understanding is more critical than architectural scale for photorealistic synthesis.
