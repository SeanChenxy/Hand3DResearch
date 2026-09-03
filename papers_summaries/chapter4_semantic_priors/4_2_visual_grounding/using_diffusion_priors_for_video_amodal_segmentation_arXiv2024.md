# Using Diffusion Priors for Video Amodal Segmentation

**Authors:** Kaihua Chen, Deva Ramanan, Tarasha Khurana (Carnegie Mellon University)  
**Date:** 2024-12 (arXiv v1); published at CVPR 2025  
**Identifier:** [arXiv:2412.04623](https://arxiv.org/abs/2412.04623)  
**Zotero item:** `7R4GU387` ([Zotero](zotero://select/library/items/7R4GU387))  
**Evidence status:** Zotero metadata and the full paper PDF (CVPR open-access version, all 10 pages including tables and ablations) were verified.

## Summary

This work tackles video amodal segmentation — segmenting objects to their full extent including occluded regions — together with video-level amodal content completion, which inplants plausible RGB content in the occluded areas. The method repurposes Stable Video Diffusion (SVD) as a conditional generative model: a first stage is conditioned on a sequence of modal (visible) masks from an off-the-shelf segmenter plus contextual pseudo-depth maps to predict amodal masks, and a second stage, using the same architecture with different conditioning (modal RGB content and the predicted amodal mask), fills in the occluded regions. The key insight is that foundation diffusion models pre-trained at scale carry strong shape and temporal-consistency priors: the multi-frame video formulation propagates object shape and content across time, so the shape of a fully occluded object can be inferred from frames where it is visible. Trained primarily on synthetic data, the approach achieves state of the art across four datasets, with up to 13% improvement for amodal segmentation in occluded regions, strong zero-shot generalization to real-world data, and multi-modal outputs that provide multiple plausible completions usable for downstream 4D reconstruction, scene manipulation, and pseudo-groundtruth generation.

## Background and Problem

Humans exhibit object permanence — inferring the complete outline of an occluded object — but present-day segmentation methods only segment visible or modal regions. Amodal segmentation and completion are ill-posed in a monocular setup: many plausible boundary extensions and content fills exist, so learning the multi-modal distribution requires generative priors. Image-based amodal methods struggle with high occlusion levels, since amodal masks cannot be inferred from a single frame when the object is heavily or fully occluded. Existing video amodal segmentation algorithms are typically limited to rigid objects, depend on additional inputs like camera poses or optical flow, and are trained and evaluated on synthetic datasets of rigid objects with limited scale and diversity. There was also no real-world dataset with ground-truth amodal masks and RGB contents. The goal is a scalable, video-level amodal segmentation and completion method that handles deformable objects and severe occlusion without extra inputs, and generalizes to real videos.

## Method

The pipeline has two diffusion-based stages sharing the same backbone: an open-source video latent diffusion model (SVD) trained with the EDM framework, whose 3D U-Net processes VAE-encoded latents with interleaved spatial and temporal blocks — temporal convolutions capture local features across frames while temporal attention propagates information between distant frames.

Stage 1 (amodal segmentation): given the input RGB video, modal masks produced by a conventional segmenter (e.g., Segment Anything 2), and pseudo-depth maps computed by Depth Anything V2, the model generates the full-extent amodal mask sequence. The off-the-shelf image-to-video SVD structure and conditioning are adapted to suit the mask-generation task. Pseudo-depth is preferred over RGB conditioning because occlusions are typically caused by objects closer to the camera, so depth maps provide implicit cues about potential occluders — an ablation shows depth-only conditioning beats RGB-only and even the mask+RGB combination on occluded-region metrics. Following ControlNet, the parameters of the first two channels in the input layer are retained while newly added conditioning channels are zero-initialized ("zero convolution"), preserving initial prediction capability; two-stage finetuning first trains the mask-conditioned model, then uses it to initialize the mask-and-depth-conditioned model. CLIP embeddings of the modal masks are injected into the transformer layers via cross-attention, providing temporal information about object visibility in surrounding frames.

Stage 2 (content completion): a second SVD model with the same architecture is conditioned on the modal RGB content of the object and the predicted amodal mask from stage 1, generating RGB content across the entire amodal region.

Synthetic data curation: since real amodal ground truth is scarce, training pairs for completion are constructed from SAIL-VOS sequences by selecting objects with near-complete visibility (above 95%) and sequentially overlaying random amodal mask sequences onto the fully visible object until its visibility falls below a set threshold, thereby simulating occlusion with ground-truth RGB for the occluded regions.

## Contributions

1. A two-stage video amodal segmentation and content-completion framework built by repurposing SVD with mask, pseudo-depth, and RGB conditioning, requiring no camera poses or optical flow.
2. A multi-frame formulation that propagates shape and appearance across time, enabling prediction of amodal masks for heavily and fully occluded — including deformable — objects from frames where they are visible.
3. A pseudo-depth conditioning design (shown superior to RGB conditioning) with zero-convolution injection and staged finetuning, plus a synthetic data curation recipe that generates occlusion ground truth from near-complete-visibility sequences.
4. State-of-the-art results on four amodal benchmarks with up to 13% improvement in occluded-region segmentation, zero-shot generalization to real-world data, and demonstrations in 4D reconstruction, scene manipulation, and pseudo-groundtruth generation.

## Experimental Setup

Training uses the SVD-xt 1.1 checkpoint with AdamW (β1 0.9, β2 0.999), learning rates 3e-5 and 3e-6 for models without and with pseudo-depth conditioning, batch size 8, 128×256 frames, about 30 hours on 8 NVIDIA RTX 3090 GPUs; inference uses 25 EDM denoising steps, guidance scale 1.5, and 256×512 frames. Evaluation covers SAIL-VOS (210 GTA-V sequences, 162 objects, yielding 21,237 25-frame clips), MOVI-B/D (13,997 and 12,010 Kubrics sequences of simulated environments with strong camera motions), and TAO-Amodal (993 validation sequences, segmented to 1,392 object sequences since it provides only bounding boxes). Metrics are mIoU and mIoU_occ (IoU over occluded pixels only) for mask datasets and AP25/50/75 over bounding boxes for TAO-Amodal, with Top-K metrics (best of K predictions) accounting for multi-modal generation. Baselines span image amodal methods (Conv-M, AISFormer, PCNet-M, pix2gestalt), video amodal methods (SaVos, Bi-LSTM, EoRaS, C2F-seg), and regression baselines (VideoMAE, the SVD 3D U-Net backbone). A human A/B study on 20 randomly selected sequences compares completion outputs against pix2gestalt.

## Results

- SAIL-VOS: Top-1 mIoU 72.07 / mIoU_occ 55.12 versus PCNet-M 74.2 / 42.52 and pix2gestalt Top-1 60.79 / 33.76 — nearly a 13% improvement in Top-1 mIoU_occ over the second-best method; Top-3 reaches 79.23 / 59.69 with AP25 98.31, AP50 92.46, AP75 77.48.
- MOVI-B/D: despite strong camera motion, the method beats all prior state of the art by over 4% Top-1 mIoU_occ on both datasets (83.51/53.75 on MOVI-B; 77.03/44.23 on MOVI-D), adapting well without camera extrinsics or optical flow, unlike baselines that require them.
- TAO-Amodal (zero-shot, trained only on synthetic data): AP25 97.28 / AP50 89.25 / AP75 71.99 (Top-1), highlighting strong generalization to real-world data.
- Visibility analysis: performance stays consistent across the entire visibility range on SAIL-VOS, realistically hallucinating masks even for nearly fully occluded objects (e.g., an occluded chair).
- Human evaluation: 83.6% preference for this method over pix2gestalt; qualitative comparisons also show markedly higher temporal consistency across occlusions than the single-frame baseline.
- Top-K: mIoU and mIoU_occ improve monotonically with K (best-of-K), with gains gradually plateauing; different seeds yield multiple plausible interpretations of the occluded region (e.g., a person's occluded legs standing or sitting).
- Ablations: pseudo-depth conditioning gives the best occluded-region results (mIoU_occ 55.12 versus 53.3 with RGB and 51.28 mask-only); zero-initialized convolution and two-stage finetuning each improve performance (77.07/55.12 versus 73.73/41.35 without both).

## Limitations

The amodal segmentation and mask models are trained almost exclusively on synthetic data, so performance on real-world domains relies on transfer; the reported real-world evaluation is zero-shot with no fine-tuning, and the lack of real amodal ground truth precludes standardized real-data metrics (human preference is used instead). Completion quality degrades for unseen categories with unusual styles, and because the model is generative, single predictions may not match the ground truth — multiple samples are needed, which the Top-K protocol reflects. The pseudo-depth prior assumes occluders are closer to the camera, so occlusion by farther or coplanar objects is not modeled, and the two-stage pipeline inherits SVD's compute and memory demands (8 GPUs for training, fixed frame resolutions at inference). Temporal consistency, while better than single-frame methods, depends on the visibility of the object in neighboring frames; a fully occluded object visible in no frame cannot be recovered.
