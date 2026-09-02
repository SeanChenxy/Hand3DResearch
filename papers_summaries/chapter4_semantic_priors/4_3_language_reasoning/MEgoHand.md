# MEgoHand: Multimodal Egocentric Hand-Object Interaction Motion Generation

**Authors:** Bohan Zhou, Yi Zhan, Zhongbin Zhang, Zongqing Lu  
**Date:** 2025-05-22  
**Identifier:** [arXiv:2505.16602](https://arxiv.org/abs/2505.16602)  
**Zotero item:** `QMN4H8US` ([Zotero](zotero://select/library/items/QMN4H8US))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

MEgoHand generates physically plausible hand-object interaction motion from egocentric RGB, a text instruction, and an initial MANO hand pose. It uses a bi-level architecture: a high-level "cerebrum" combining a vision-language model (Eagle-2) with monocular metric depth estimation for object-agnostic spatial reasoning, and a low-level DiT-based flow-matching policy with a training-free Temporal Orthogonal Filtering decoder for stable trajectories. Supported by a curated 3.35M-frame, 24K-interaction, 1.2K-object dataset, it outperforms LatentAct on five in-domain and two cross-domain benchmarks, cutting mean rotation error from 0.937 to 0.123 rad (86.9%) and wrist translation error from 7.221 to 4.756 cm (34.1%).

## Background and Problem

Generating hand-object motions from first-person views underpins AR/VR virtual-real alignment and robot imitation from human demonstrations, but egocentric settings suffer from unstable head-mounted viewpoints, frequent self-occlusions, close-range perspective distortion, and noisy ego-motion. Existing methods typically rely on predefined 3D object attributes such as mass and geometry (GEARS, MACS, DiffH2O, Text2HOI), which degrades on novel objects; image-based approaches like SIGHT-Fusion lack textual guidance and produce ambiguous actions; multimodal LatentAct requires an intricate contact-map pipeline; and open-loop prediction from the first frame accumulates errors into cascading failures. MEgoHand predicts future MANO parameter sequences (6D rotations for finger joints and wrist) of length l given task text T, visual observation Vk, and initial hand parameters hk, computed in the camera frame.

## Method

The high-level "cerebrum" builds on Eagle-2, a VLM with a SmolLM2 language backbone and SigLIP-2 vision encoder (text tokenizer and transformer frozen; vision encoder fine-tuned). An RGB frame is processed by the UniDepthV2 monocular metric depth estimator; the depth map is encoded by an ImageNet-pretrained ResNet-50 (single channel replicated to three) with MSE depth supervision, and fused with visual features before cross-modal attention with text embeddings. The low-level "cerebellum" is a Diffusion Transformer (DiT) flow-matching generator conditioned on the multimodal latent and initial MANO parameters; it predicts relative wrist transformations and repeats the initial shape beta, trained with a conditional flow-matching loss whose timestep is sampled from a beta distribution biased toward noisier values. At inference, Temporal Orthogonal Filtering (TOF) queries overlapping motion chunks, aggregates per-timestep estimates with a uniform temporal convolution, and projects the averaged rotation onto SO(3) via SVD — a training-free decode that suppresses jitter and enables closed-loop chunked prediction.

## Contributions

1. The first framework to leverage a VLM for motion-prior inference in egocentric hand-object interaction, augmented with a monocular metric depth module for object-agnostic 3D spatial reasoning. 2. A dataset curation pipeline — an Inverse MANO Retargeting Network (two-stage shape-then-wrist optimization with self-reconstruction loss, pretrained on 10K pairs from TACO and OakInk2) plus a Virtual RGB-D Renderer synthesizing depth aligned to RGB — yielding 3.35M RGB-D frames, 24K interaction trajectories, and 1.2K objects. 3. State-of-the-art generation with strong cross-domain generalization and fine-grained articulation.

## Experimental Setup

Training uses TACO, FPHA, HOI4D, H2O, HOT3D, and OakInk2 (FPHA re-annotated via the retargeting network and used exclusively for evaluation); in-domain evaluation holds out 10% of five datasets with no action/object overlap, and cross-domain zero-shot evaluation uses the full ARCTIC dataset and a 10% HOLO partition. Metrics: MPJPE, Procrustes-aligned MPJPE, MPVE, MPVE-PA, wrist translation error MWTE (cm), and joint rotation error MRE (radians, averaged over 16 joints). Baselines are LatentAct and its diffusion variant LatentAct-Diff (with and without contact maps), plus five MEgoHand modality variants (text-only, RGB-only, RGB+depth, text+RGB, full).

## Results

In-domain, MEgoHand achieves MPJPE 5.425 cm, MPJPE-PA 0.424 cm, MPVE-PA 0.409 cm, MWTE 4.756 cm, and MRE 0.123 rad (about 7 degrees), versus LatentAct's 7.726/1.478/7.696/7.221/0.937 — an 86.9% MRE reduction, a 34.1% wrist translation reduction, and 71.2%/71.9% relative MPJPE-PA/MPVE-PA improvements. Text-only input is weakest (MPJPE 8.328, +61% translation error over the full model) and RGB-only converges to ambiguous average behaviors; removing LatentAct's contact map raises its MPJPE by 10.3%, while MEgoHand-TI still halves LatentAct's error. Zero-shot, MEgoHand reaches MPJPE 7.358 cm on ARCTIC and 5.775 cm on HOLO, improvements of 33.9% and 29.8% over the strongest baselines. Metric depth beats relative depth in-domain (5.425 vs 5.610 MPJPE) while relative depth is more robust to drastic camera changes cross-domain; removing depth supervision hurts (5.725 in-domain).

## Limitations

The paper states that training currently covers only right-hand motion. Its rigid-object training data limits performance on articulated objects — ARCTIC's dynamic hand-object coupling (e.g., scissor-cutting) exposes this weakness, and HOI4D's 800 object instances across 610 scenes make it the hardest in-domain set. The authors note that data scale remains bounded by curated datasets and propose using the pretrained inverse MANO retargeting network to annotate more HOI datasets, or modern hand pose detectors to label in-the-wild videos, as future work toward better results.
