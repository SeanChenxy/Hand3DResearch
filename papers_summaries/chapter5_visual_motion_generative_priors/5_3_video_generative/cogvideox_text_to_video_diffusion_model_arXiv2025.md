# CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer

**Authors:** Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Yuxuan Zhang, Weihan Wang, Yean Cheng, Bin Xu, Xiaotao Gu, Yuxiao Dong, Jie Tang  
**Date:** 2024-08-12  
**Identifier:** [arXiv:2408.06072](https://arxiv.org/abs/2408.06072)  
**Zotero item:** `QZY5PR2M` ([Zotero](zotero://select/library/items/QZY5PR2M))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
CogVideoX is a diffusion-transformer text-to-video model from Zhipu AI and Tsinghua University, released in 2B and 5B parameter sizes, that generates 10-second, 16 fps videos at up to 768x1360 resolution with coherent, high-motion content. Its three core designs are a 3D causal VAE that compresses video jointly in space and time (8x8x4), an expert transformer with modality-specific adaptive LayerNorm plus 3D full attention for deep text-video fusion, and progressive training with multi-resolution frame packing. CogVideoX-5B reports state-of-the-art automated and human-evaluation results, beating the closed-source Kling in human preference.

## Background and Problem
DiT-based text-to-video models (e.g., following Sora) still struggle with limited motion, short durations, and incoherent long-duration narratives. Two technical bottlenecks are how to efficiently consume high-dimensional video data and how to fuse text and video modalities within one transformer. Prior approaches that fine-tune 2D VAEs into video VAEs suffer from flickering and long latent sequences, and naive concatenation of text and video embeddings mixes feature spaces of very different scales.

## Method
(1) A 3D causal VAE with temporally causal convolution compresses videos 8x8x4 (channels x height x width x time), trained with L1 reconstruction, LPIPS, KL, and a GAN loss from a 3D discriminator; an ablation over compression/latent-channel variants shows reduced inter-frame flickering versus an SDXL 2D VAE baseline (variant B, 8x8x4 with 16 latent channels, flickering 86.3 and PSNR 28.7 vs baseline 93.2/28.4). (2) The expert transformer patchifies video latents, applies 3D-RoPE positional encoding, concatenates text (T5 embeddings) and video tokens along the sequence, and uses 3D full attention over the joint sequence; Expert Adaptive LayerNorm applies separate vision and text expert modulation (timestep-conditioned, DiT-style) so each modality is normalized independently. (3) Progressive training with multi-resolution frame packing and Explicit Uniform Sampling (different timestep sampling intervals per data-parallel rank) stabilize and speed up training. Data: filtering yields about 35M single-shot clips averaging 6 seconds plus 2B aesthetic-filtered images from LAION-5B and COYO-700M; dense captions are produced via a Panda70M-style short-caption model, per-frame dense image captions, GPT-4 summaries, and a fine-tuned Llama 2 recaptioner.

## Contributions
A simple, scalable text-to-video architecture combining 3D causal VAE and expert transformer for coherent, long, high-action videos with multiple aspect ratios (up to 768x1360, 10 s, 16 fps); an effective text-video data preprocessing and recaptioning pipeline; and state-of-the-art automated and human evaluation with released code, checkpoints, VAE, and captioning model (github.com/THUDM/CogVideo). The paper was published at ICLR 2025.

## Experimental Setup
Two model sizes (2B and 5B) are trained. Automated evaluation uses VBench-derived metrics (Action Degree, Human Motion, Multiple Objects, Dynamic Objects, Appearance Style), plus Dynamic Quality and a GPT4o-MT metamorphic-amplitude score, compared against T2V-Turbo, AnimateDiff, VideoCrafter-2.0, OpenSora V1.2, Show-1, Gen-2, Pika, and LaVie-2. Human evaluation scores Sensory Quality, Instruction Following, Physics Simulation, and Cover Quality on 0/0.5/1 levels, comparing CogVideoX-5B with Kling (2024.7).

## Results
CogVideoX-5B is best in five of seven automated metrics: Action Degree 96.8, Human Motion 55.44, Multiple Objects 62.22, Dynamic Objects 70.95, Appearance Style 24.44, Dynamic Quality 69.5, and GPT4o-MT 3.36 (versus 2.68 for VideoCrafter-2.0 and 2.52 for OpenSora V1.2). CogVideoX-2B is competitive (e.g., Multiple Objects 66.39, GPT4o-MT 3.09). In human evaluation, CogVideoX-5B totals 2.74 versus Kling's 2.17 (Sensory 0.722 vs 0.638; Instruction Following 0.495 vs 0.367; Physics Simulation 0.667 vs 0.561; Cover Quality 0.712 vs 0.668), winning across all aspects.

## Limitations
The paper contains no dedicated limitations section. The conclusion frames current scale as a starting point, stating the authors are exploring scaling laws to train larger models generating longer, higher-quality videos, implying that the released 2B/5B models do not exhaust the achievable scale or duration. Detailed failure modes are not reported in the paper.

