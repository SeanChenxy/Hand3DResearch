# iDiT-HOI: Inpainting-based Hand Object Interaction Reenactment via Video Diffusion Transformer

**Authors:** Zhelun Shen, Chenming Wu, Junsheng Zhou, Chen Zhao, Kaisiyuan Wang, Hang Zhou, Yingying Li, Haocheng Feng, Wei He, Jingdong Wang  
**Date:** 2025-06-15  
**Identifier:** [arXiv:2506.12847](https://arxiv.org/abs/2506.12847)  
**Zotero item:** `U87FNG9M` ([Zotero](zotero://select/library/items/U87FNG9M))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

iDiT-HOI is a two-stage video diffusion transformer (DiT) framework for hand-object interaction (HOI) reenactment: given a masked source video and a reference object image, it re-inserts the object into the hands and regenerates the interaction. Its Inpainting-based Token Process Unit (Inp-TPU) reuses the pretrained DiT's own attention parameters on masked tokens to inject object information with zero new parameters, yielding strong generalization to unseen objects and in-the-wild e-commerce livestreaming scenes and naturally supporting long-video generation.

## Background and Problem

Digital human video generation is advancing, but realistic HOI remains a bottleneck: hands and objects occlude each other, object shapes and orientations vary widely, physically precise contact is required, and methods must generalize to unseen humans and objects. Prior reenactment methods have drawbacks: HOI-Swap handles only object-centric single-hand grasping and generalizes poorly to novel objects; Re-HOLD needs human-curated layout videos and rendered hand meshes at inference and duplicates the main diffusion network to preserve object identity, inflating parameters; general video inpainting/editing models (e.g., VACE, AVID, COCOCO) rely on dual-stream or duplicated-parameter designs and struggle with fine-grained HOI. The problem is high-fidelity HOI reenactment from a source video plus a reference object image, without extra parameters or inference-time human input, extending beyond source domains to in-the-wild scenarios.

## Method

The framework trains a DiT (built on the pretrained Wan-14B I2V model plus FLUX.1-dev) in a self-supervised reconstruction manner: masked video and reference object image, both derived from the source video, condition reconstruction of the original video from noise in a VAE latent space. Inp-TPU temporally extends the reference image to video length, spatially aligns it to masked regions via mask centroids and sizes, VAE-encodes and patchifies both streams, downsamples the mask to latent resolution, and forms conditional tokens as Xcond = (1 - XM) * X_masked + XM * X_ref — reusing existing masked-region attention parameters instead of adding modules. Generation follows an image-then-video paradigm: stage 1 (Mimg) inserts the object into the hand region of a key frame via image-to-image denoising; stage 2 (Mvid) generates the remaining frames conditioned on the key frame. An adaptive masking strategy builds a soft ellipsoid mask from the oriented bounding box (axes enlarged by sqrt(2), aspect ratio matched to the target object) for shape-aware inpainting. Long videos chain clips: each clip's last frame seeds the next, requiring no dedicated long-video model.

## Contributions

(1) A unified inpainting-based token processing method (Inp-TPU) inside a DiT architecture that handles hand movements and object details while ensuring natural contact, without new trainable parameters. (2) Efficient reuse of the pretrained model's context perception capabilities, improving real-world generalization for HOI tasks such as object swapping, unlike duplicated-parameter approaches (Re-HOLD, VACE). (3) Extensive self-reenactment and cross-reenactment experiments, including challenging in-the-wild scenarios, demonstrating potential for industrial-scale HOI video generation.

## Experimental Setup

Training used 19,000 video clips on 8x NVIDIA H100 80G GPUs with full fine-tuning. Evaluation uses Re-HOLD (139 self-reenactment and 140 cross-reenactment videos, in-domain) and HOI-ITW (30 self-reenactment and 10 cross-reenactment videos collected in e-commerce livestreaming, unseen by training). Metrics: PSNR and FID (self-reenactment, computed on a fixed 81 frames), subject consistency, and motion smoothness, plus an IRB-approved user study (10 participants; 12 Re-HOLD and 10 HOI-ITW videos; 1-5 ratings on video quality, reference fidelity, temporal consistency). Baselines: Re-HOLD, HOI-Swap, VACE, AnimateAnyone, AnyV2V, RealisDance, VideoSwap.

## Results

On Re-HOLD (81-frame setting), iDiT-HOI achieves the best FID of 12.07 and PSNR 33.74 in self-reenactment (Re-HOLD: 13.79/32.96; VACE: 27.34/35.86), ranking top in 4 of 6 metrics. On the unseen HOI-ITW set, FID is 35.01 versus VACE's 59.94 and Re-HOLD's 127.8; cross-reenactment subject consistency is 0.953 (VACE 0.954, Re-HOLD 0.923). In the user study, iDiT-HOI scores 3.80/3.87/3.90 (video quality/reference fidelity/temporal consistency) on Re-HOLD and 3.525/3.875/3.55 on HOI-ITW, where Re-HOLD collapses (1.125/1.7/1.1) and VACE's reference fidelity drops from 2.78 to 1.55. Overall the method is top in 7 of 9 metrics. Ablations on HOI-ITW: removing key frame generation degrades all metrics (FID 81.81 in cross-reenactment), and removing object fusion in Inp-TPU causes a 27.08% FID drop (48.25 vs. 35.01).

## Limitations

The paper itself states: (1) the two-stage pipeline requires denoising twice rather than once; (2) generation relies solely on the object image and masked video, limiting fine-grained manipulations such as object rotation and flipping, which the authors propose to address in future work by incorporating 6D object pose information.
