# Adding Conditional Control to Text-to-Image Diffusion Models

**Authors:** Lvmin Zhang, Anyi Rao, Maneesh Agrawala  
**Date:** 2023-10-01 (ICCV)  
**Identifier:** [arXiv:2302.05543](https://arxiv.org/abs/2302.05543); DOI `10.1109/ICCV51070.2023.00355`  
**Zotero item:** `7V2UNHGP` ([Zotero](zotero://select/library/items/7V2UNHGP))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; summary content is derived from the paper with in-text caveats where detail is unavailable.  
## Summary
ControlNet addresses the weak spatial controllability of large text-to-image diffusion models. It freezes a pretrained diffusion backbone, adds a trainable copy of its encoding path, and connects the two with zero-initialized convolutions so that conditioning can be learned without disturbing the original model at initialization. The architecture accepts spatial signals such as edges, pose, depth, segmentation, or user sketches in addition to text. Experiments across multiple condition types and dataset sizes show robust controllable generation, with the paper also reporting comparisons and user studies against alternative adaptation methods.

## Background and Problem
Text prompts do not precisely specify composition, pose, boundaries, or depth. Existing adaptation methods can require substantial training or can damage the quality learned by the large pretrained model, especially when the new condition dataset is much smaller than the pretraining corpus. ControlNet takes a text prompt and an optional spatial condition map as input and outputs an image that follows both the semantic and spatial conditions.

## Method
The original diffusion model remains locked. ControlNet creates a trainable copy of the encoding layers and passes its features to the locked branch through zero convolutions whose weights and biases start at zero. Consequently, the added path initially contributes no perturbation and can gradually learn the condition during fine-tuning. The same design supports individual or multiple spatial conditions, with or without text.

## Contributions
- A zero-convolution architecture for adding spatial control to a frozen text-to-image diffusion model.
- Support for diverse conditions, including edges, lines, keypoints, segmentation, normals, depth, and sketches.
- A practical adaptation strategy that works across task-specific datasets much smaller than the original pretraining data.

## Experimental Setup
The paper evaluates Canny edges, Hough lines, user scribbles, human keypoints, segmentation maps, shape normals, and depth. It studies training sets ranging from fewer than 50,000 to more than 1 million examples and compares with T2I-Adapter, HyperNetworks, and LoRA-style alternatives, including user studies. The available evidence does not provide a complete numerical table for every condition.

## Results
ControlNet is reported to preserve the quality of the pretrained model while following the added spatial conditions across all tested modalities. The depth-conditioned system achieves competitive results with training on a single NVIDIA RTX 3090Ti, and the ablations support the zero-initialized connection design. Exact scores for each benchmark are not reported in the available evidence.

## Limitations
A separate task-specific training set is required for each condition type, and the model inherits biases and failure modes of the pretrained Stable Diffusion backbone. Combining multiple conditions increases computation. The paper does not establish that every spatial condition or out-of-distribution scene can be controlled reliably.
