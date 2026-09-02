# Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets

**Authors:** Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach  
**Date:** 2023-11-25  
**Identifier:** [arXiv:2311.15127](https://arxiv.org/abs/2311.15127)  
**Zotero item:** `5WW9KT7D` ([Zotero](zotero://select/library/items/5WW9KT7D))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Stable Video Diffusion (SVD) is a latent video diffusion model created by inserting temporal layers into a pretrained text-to-image model, showing that systematic training-data curation—not just architecture—is the decisive factor for video generation quality. Starting from a 580M-clip collection, filtering yields a 152M-clip pretraining set (LVD-F), and a three-stage schedule (text-to-image pretraining, video pretraining, high-quality finetuning) produces base models that reach state-of-the-art zero-shot text-to-video FVD of 242.02 on UCF-101 and beat commercial image-to-video systems in human preference studies. The same base model also serves as a strong multi-view 3D prior.

## Background and Problem
Latent diffusion models (LDMs) trained for 2D image synthesis have been turned into video generators by inserting temporal layers and finetuning on small, high-quality video datasets. However, training recipes in the literature vary widely, and the field lacks a unified strategy for curating video data; the influence of training-data selection on video generation had not been investigated despite its undisputed importance. Commonly used public datasets such as WebVid-10M are watermarked and suboptimal in size, which motivates building and curating a large proprietary dataset.

## Method
The paper defines three training stages: (1) text-to-image pretraining, (2) video pretraining, and (3) high-quality video finetuning. For data curation, cuts are detected with PySceneDetect, and each clip receives three synthetic captions: CoCa captions the mid-frame, V-BLIP provides a video-based caption, and an LLM summarizes the two. Filtering uses CLIP scores, aesthetic scores, OCR detection rates, and optical flow scores; the resulting Large Video Dataset (LVD, 580M annotated clips, 212 years of content) is filtered to LVD-F with 152M clips. The base model starts from Stable Diffusion 2.1 weights at 256x384 resolution, trains on 14 frames (150k iterations, batch size 1536), then at 320x576 (100k iterations, batch size 768) with EDM preconditioning, shifting the noise schedule toward more noise for higher-resolution stages. Finetuning variants include SVD (image-to-video, 14 frames at 576x1024) and SVD-XT (25 frames), plus a text-to-video model finetuned on about 1M high-quality samples.

## Contributions
(i) A systematic data curation workflow converting a large uncurated video collection into a quality pretraining dataset, with evidence that curation improves results at fixed scale. (ii) State-of-the-art text-to-video and image-to-video models built on the three-stage recipe. (iii) Probing the motion and 3D understanding of the pretrained base model: image-to-video generation, camera motion-specific LoRA modules, and a multi-view diffusion model finetuned from the video base that outperforms specialized novel-view-synthesis methods (Zero123XL, SyncDreamer) in multi-view consistency at a fraction of their compute. Code and model weights are released.

## Experimental Setup
Ablations train smaller models on a 9.8M subset (LVD-10M) and its filtered version (LVD-10M-F, 2.3M clips), comparing against WebVid and InternVid subsets via human preference studies. The full base model is trained on LVD-F as described above. Evaluations cover zero-shot text-to-video FVD on UCF-101, human preference comparisons of the 25-frame image-to-video model against GEN-2 and PikaLabs, and multi-view generation comparisons against Zero123XL and SyncDreamer.

## Results
The base model achieves zero-shot UCF-101 FVD of 242.02, versus 355.20 for PYOCO, 367.23 for Make-A-Video, 550.61 for Video LDM, and 701.59 for CogVideo (EN). Filtering LVD-10M to 2.3M curated clips improves human preference over the uncurated set, confirming curation matters more than raw scale. The 25-frame image-to-video model is preferred by human voters over GEN-2 and PikaLabs. The multi-view model finetuned from the video base outperforms Zero123XL and SyncDreamer in multi-view consistency.

## Limitations
The paper states that although the approach excels at short video generation, it has fundamental shortcomings for long video synthesis: generating multiple keyframes at once is expensive during both training and inference, and future work should consider coarse-to-fine frame cascades or dedicated video tokenizers. Generated videos sometimes suffer from too little motion. Video diffusion models are typically slow to sample and have high VRAM requirements, and this model is no exception; diffusion distillation is named as a promising direction for faster synthesis.

