# Open-Sora Plan: Open-Source Large Video Generation Model

## Summary
Open-Sora Plan is an open-source project for generating high-resolution, long-duration videos using a Wavelet-Flow VAE for compression, a 3D transformer-based denoiser with joint image-video capabilities, and multi-modal condition controllers, supported by efficient training strategies and automated data curation.

## 1. Problem and Setting
High-quality, long-duration video generation requires immense computational and data costs. Existing methods produce low-resolution, short videos. This work aims to enable generation of high-resolution videos with long durations (up to 16 seconds at 720p) conditioned on text prompts, images, or structural controls (canny, depth, sketch).

## 2. Core Method
**Pipeline:** Three-stage architecture:
- **WF-VAE (Wavelet-Flow Variational Autoencoder):** Uses multi-level Haar wavelet transform for frequency-domain decomposition, achieving compression rates of 4×8×8 (temporal×height×width) with pyramid backbone injection
- **Joint Image-Video Skiparse Denoiser:** 3D full-attention transformer (changed from 2+1D) with Skiparse Attention for computation reduction, handling both image and video generation
- **Condition Controllers:** Frame-level image controller for Image-to-Video/Video Transition/Video Continuation; structure control network for controllable generation

**Innovations:**
- Causal Cache method to fix latent space disruption from tiling inference
- Min-Max Token Strategy for efficient mixed-resolution/duration training
- Adaptive Gradient Clipping for outlier detection
- Prompt Refiner for semantic consistency

## 3. Knowledge, Supervision, and Assumptions
**Data:** Multi-dimensional curation pipeline from uncleaned video datasets with LPIPS-based jump cut detection, motion filtering, aesthetic scoring, and caption annotation. Uses large-scale video datasets (not specified in excerpt).

**Pretrained Models:** Builds on diffusion models (Ho et al., 2020; Song et al., 2020) and transformer architectures (Vaswani, 2017; Peebles and Xie, 2023). Wavelet transform theory (Haar filters).

**Assumptions:** Wavelet domain decomposition preserves essential visual information; 3D attention better captures temporal coherence than 2+1D structures; curated data with consistent prompts improves motion stability.

## 4. Experiments and Findings
**Datasets:** Curated from uncleaned video sources with multi-dimensional processing pipeline (specific dataset sizes not in excerpt).

**Metrics:** Qualitative and quantitative evaluations mentioned but specific metrics not detailed in excerpt.

**Results:** Claims impressive video generation results in high-resolution and long-duration scenarios. All code and model weights publicly available.

## 5. Strengths and Limitations
**Strengths:** Fully open-source with public code/weights; comprehensive pipeline covering data curation to inference; multi-modal condition support; efficient training strategies for memory/computation reduction.

**Limitations:** Ongoing project (continuous efforts); specific quantitative results not detailed in excerpt; computational requirements still significant despite optimizations; relies on automated data curation quality.

## 6. Takeaway
Open-Sora Plan demonstrates that systematic architecture design (wavelet compression + 3D attention + efficient training) enables practical high-quality long video generation. The modular design (WF-VAE, denoiser, controllers) and open-source approach provide a foundation for community advancement in video generation research.
