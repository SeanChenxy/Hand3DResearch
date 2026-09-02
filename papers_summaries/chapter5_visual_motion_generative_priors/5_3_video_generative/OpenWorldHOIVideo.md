# Open-world Hand-Object Interaction Video Generation Based on Structure and Contact-aware Representation

**Authors:** Haodong Yan, Hang Yu, Zhide Zhong, Weilin Yuan, Xin Gong, Zehang Luo, Chengxi Heyu, Junfeng Li, Wenxuan Song, Shunbo Zhou, Haoang Li  
**Date:** 2025-12-01  
**Identifier:** [arXiv:2512.01677](https://arxiv.org/abs/2512.01677)  
**Zotero item:** `DBNBVBJ2` ([Zotero](zotero://select/library/items/DBNBVBJ2))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

The paper proposes SCAR, a 3D-annotation-free HOI representation pairing contact-augmented hand-object contours with video depth maps to capture contact, occlusion, and holistic structure. A joint-generation paradigm with a hierarchical share-and-specialization denoiser generates representation and RGB video simultaneously, avoiding multi-stage error accumulation. Instantiated on CogVideoX-I2V-5B (SCAR_C) and Wan2.1-I2V-14B (SCAR_W), it outperforms state-of-the-art baselines on two real-world datasets and generalizes to 200 open-world samples with unseen objects.

## Background and Problem

Conditioned on an observed image and a task description, HOI video generation must synthesize hands manipulating objects with physics-realistic contact/occlusion and temporally coherent motion. General video diffusion models lack inductive biases for interaction physics and often produce distorted hands and implausible contact. Existing HOI-representation-guided methods face a scalability-fidelity dilemma: scalable 2D signals (optical flow, segmentation, 2D keypoints) lack structure and contact cues, while 3D mesh/MANO sequences carry full structure but need costly 3D annotations that hinder scale-up. Multi-stage methods (MaskI2V, Taste-Rob, FLOVD) train each stage on ground truth but condition on prior-stage predictions at inference, accumulating errors. The problem: a scalable, interaction-oriented supervision signal and a training paradigm avoiding error propagation, enabling open-world generalization.

## Method

Representation curation: a Chain-of-Thought-prompted VLM (Qwen2.5-VL) grounds hands and objects; SAM2 segments and propagates masks, followed by manual verification. The contact proxy is the intersection of dilated hand and object contours, with a scale-adaptive object dilation radius r_o = min(r_max, max(r_min, beta * L)) proportional to the object bounding-box diagonal L. Sparse contours (rather than dense masks) preserve the depth map during alpha-blending; depth comes from Video Depth Anything, giving reliable relative structure without absolute scale. Joint generation: a 3D VAE encodes the RGB video and the video-like HOI representation into one latent space; visual and interaction tokens are concatenated and co-denoised by a hierarchical joint denoiser on a DiT. The Shared Semantics module (layers 1 to k*) enforces cross-modal alignment via a cosine-similarity loss on hidden states; the Specialized Details module adds a learnable interaction embedding to interaction tokens only. Identical positional encodings link corresponding spatio-temporal tokens; LoRA adapters (dimension 128 for SCAR_C, 256 for SCAR_W) apply selectively to interaction tokens via a binary mask, preserving pre-trained visual knowledge. The loss combines the diffusion losses of both streams (weight 1.0) with the alignment loss (weight 0.1).

## Contributions

(1) A structure and contact-aware representation as a scalable, interaction-oriented supervisory signal requiring no 3D annotations, curated for over 100k HOI videos. (2) A joint-generation paradigm with a share-and-specialization denoiser generating representation and video simultaneously, mitigating multi-stage error accumulation. (3) State-of-the-art physics realism and temporal coherence on two real-world datasets plus strong open-world generalization.

## Experimental Setup

Training and evaluation use Taste-Rob (100,856 fixed-view videos with manual task descriptions, official split) and Taco (2,317 double-hand videos; only egocentric views are used; split 90%/10% by action-tool-object triplet). Metrics come from VBench: subject consistency (SC), imaging quality (IQ), i2v subject/background consistency (ISC/IBC), text-to-video alignment (VCS), and the weighted total score (TS). Sequences are 17 frames on Taste-Rob and 25 frames on Taco. Baselines: the underlying backbones CogVideoX and Wan2.1 and the two-stage FLOVD (CogVideoX-based instantiation), all fine-tuned on identical splits from identical checkpoints. Open-world generalization uses a newly collected 200-sample benchmark with unseen target objects, testing SCAR_W trained on Taste-Rob.

## Results

On Taste-Rob, SCAR_W achieves the best overall TS of 9.084 (SC 0.961, IQ 0.709, ISC 0.961, IBC 0.958, VCS 0.194) versus CogVideoX 8.959, Wan2.1 8.897, and FLOVD 8.888; SCAR_C reaches 9.043. On Taco, SCAR_W scores 8.899 and SCAR_C 8.793, again ahead of all baselines. Qualitatively, CogVideoX produces distorted hands, Wan2.1 ignores the task description, and FLOVD suffers optical-flow error propagation (hallucinated objects); SCAR executes the described action with faithful representation fidelity. In the 200-sample open-world benchmark, baselines show severe distortion, temporal inconsistency, and instruction failures, while SCAR remains physics-realistic. Ablations on Taco (SCAR_C): substituting optical flow, hand-object masks, or depth alone degrades performance; removing contours (w/o HOC), contact region (w/o CG), or depth (w/o DM) each hurts metrics; adding 2D keypoints (+KP) also degrades, as the overly complex auxiliary target hinders optimization. Full SCAR is best on all five metrics (SC 0.916, ISC 0.951).

## Limitations

The paper has no dedicated limitations section. Constraints evident from its own statements: curation relies on human annotators reviewing and correcting masks; the contact region is a geometric proxy from intersecting dilated contours rather than measured contact; evaluation covers short clips of 17 and 25 frames.
