# Open-Sora Plan: Open-Source Large Video Generation Model

**Authors:** Bin Lin, Yunyang Ge, Xinhua Cheng, Zongjian Li, Bin Zhu, Shaodong Wang, Xianyi He, Yang Ye, Shenghai Yuan, Liuhan Chen, Tanghui Jia, Junwu Zhang, Zhenyu Tang, Yatian Pang, Bin She, Cen Yan, Zhiheng Hu, Xiaoyi Dong, Lin Chen, Zhang Pan, Xing Zhou, Shaoling Dong, Yonghong Tian, Li Yuan  
**Date:** 2024-11-28  
**Identifier:** [arXiv:2412.00131](https://arxiv.org/abs/2412.00131)  
**Zotero item:** `2DCSE7QL` ([Zotero](zotero://select/library/items/2DCSE7QL))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Open-Sora Plan is an open-source project (Peking University Yuan Group and collaborators) providing a full text-to-video stack: a Wavelet-Flow VAE (WF-VAE) that compresses video via multi-level wavelet transforms with a lossless Causal Cache tiling-inference scheme, a 3D full-attention "Skiparse" denoiser trained jointly on images and videos, condition controllers for image-to-video/transition/continuation and structure-conditioned generation, plus a multi-dimensional data curation pipeline. Version 1.3 uses a 2.7B-parameter model and reports competitive VBench and ChronoMagic-Bench-150 results against Open-Sora v1.2, CogVideoX, and Mochi-1.

## Background and Problem
Open-source video generation lacks complete, reproducible frameworks covering the entire pipeline from data to deployment. Key engineering bottlenecks include memory- and compute-heavy video VAEs, tiling inference that disrupts latent continuity, the choice between 2+1D factorized and full 3D attention denoisers, fixed-sequence-length training constraints, and the absence of clean, well-captioned video training data.

## Method
WF-VAE extracts multi-scale features in the frequency domain via multi-level wavelet transforms, feeding them into a pyramid convolutional backbone; Causal Cache caches tail frames between chunks during block-wise inference so reconstruction is numerically identical to direct inference (lossless). The denoiser replaces the 2+1D Sora-like design with 3D full attention and adds Skiparse Attention, which skips tokens along spatial-temporal axes to cut computation. Frame-level image condition controllers support image-to-video, video transition, and video continuation in one framework, and a structure controller injects canny edges, depth maps, and sketches for controllable generation. Assistant strategies include the min-max token strategy (bucketing mixed resolutions/durations), adaptive gradient clipping, and DropFlow. Data curation slices videos to 16 s, detects jump cuts with LPIPS, filters motion speed, crops subtitles with EasyOCR, applies a Laion aesthetic threshold of 4.75 and DOVER technical-quality filtering; these steps progressively reduce the pool to 42%. The v1.3 data card lists image sets (SAM-LLaVA 11.1M, Anytext 1.8M, LAION-human 0.1M, internal 5.0M) and video sets (Panda70M 21.2M, VIDAL 2.8M, ShareGPT4Video 0.8M), and an LLM prompt refiner (trained from GPT-4o pairs) expands short user prompts.

## Contributions
(1) WF-VAE with Causal Cache, achieving high throughput and lossless long-video encoding; (2) a joint image-video 3D full-attention denoiser with Skiparse Attention for efficiency; (3) unified condition controllers for multiple video tasks and structure-conditioned control; (4) a practical data curation and training recipe, all released open-source (github.com/PKU-YuanGroup/Open-Sora-Plan).

## Experimental Setup
VAE evaluation compares WF-VAE-S against Allegro, OD-VAE, and CogVideoX VAE on Panda70M and WebVid-10M using PSNR, LPIPS, SSIM, and reconstruction FVD on 33-frame videos (256x256 for reconstruction; 512x512 for throughput/memory). Text-to-video evaluation uses selected VBench dimensions (Object Class, Multiple Objects, Human Action, Aesthetic Quality, Spatial Relationships, Scene, and consistency metrics) plus ChronoMagic-Bench-150 (CH Score) and GPT4o-MTScore, comparing Open-Sora v1.2 (1.2B), CogVideoX-2B/5B, and Mochi-1 (10.0B).

## Results
WF-VAE-S encodes 33-frame 512x512 videos at 11.11 videos per second, roughly 6x faster than CV-VAE and 4x faster than OD-VAE, with about 5x and 7x lower memory respectively while achieving superior reconstruction quality. In Table 8, Open-Sora Plan v1.3 with the prompt refiner (2.7B) scores Aesthetic 60.70, Action Class 86.4, Object Class 84.72, Spatial Objects 49.63, Scene 52.92, Multiple Objects 44.57, CH Score 68.39, and GPT4o-MTScore 2.95, versus Open-Sora v1.2 (GPT4o-MT 2.50), CogVideoX-2B (3.09), and CogVideoX-5B (3.36); Mochi-1 leads GPT4o-MT at 3.76. The prompt refiner ablation improves most VBench dimensions.

## Limitations
The paper's dedicated limitations section states: (1) the WF-VAE decoder, modeled after Rombach et al., carries redundant parameters relative to the encoder; (2) the 2B v1.3.0 denoiser saturates late in training and poorly understands physical laws (e.g., a cup overflowing with milk, a car moving forward, a person walking), hypothesized to stem from joint image-video training with only about 10M-level image data, insufficient model scale, and the flow-matching loss; (3) future work will scale to 10-15B (DeepSpeed/FSDP with ZeRO-3) or up to 30B parameters (MindSpeed/Megatron-LM) and refine the training recipe.

