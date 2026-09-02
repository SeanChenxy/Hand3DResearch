# AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning

**Authors:** Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, Bo Dai  
**Date:** 2023-07-10  
**Identifier:** [arXiv:2307.04725](https://arxiv.org/abs/2307.04725)  
**Zotero item:** `UE44YB7B` ([Zotero](zotero://select/library/items/UE44YB7B))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
AnimateDiff is a plug-and-play motion module trained once on WebVid-10M that can be inserted into any personalized text-to-image (T2I) model derived from the same base T2I (e.g., Stable Diffusion V1.5 community models from Civitai) to turn it into an animation generator without model-specific tuning. A three-stage training recipe (domain adapter, motion module learning, MotionLoRA) yields temporally smooth clips that preserve each personalized model's visual domain, and MotionLoRA adapts the motion module to new motion patterns such as camera zooming with as few as 20-50 reference videos.

## Background and Problem
Text-to-image diffusion models plus personalization techniques (DreamBooth, LoRA, textual inversion) let users create high-quality images cheaply, but adding motion to these personalized T2Is remains open: finetuning an image model directly on videos degrades its learned domain knowledge and appearance quality. Public video training data such as WebVid-10M is also much lower quality than image datasets (motion blur, compression artifacts, watermarks), creating a domain gap between the image and video distributions that naively propagates into the motion module.

## Method
AnimateDiff inflates a frozen base T2I to 5D video tensors and inserts a temporal transformer (motion module) with self-attention across frames so information exchanges along time. Training has three stages: (1) a LoRA-based domain adapter is fitted on static video frames to absorb the visual domain gap, and is dropped at inference (its scaler alpha can be set to 0); (2) the newly initialized motion module is trained on videos while base T2I and domain adapter stay frozen, so it learns transferable motion priors; (3) MotionLoRA fine-tunes the motion module with LoRA on a small set of reference videos, using rule-based augmentation (e.g., temporal zoom of the crop window) to synthesize camera-motion patterns. At inference the motion module is injected into any personalized T2I sharing the base model, and frames come from reverse diffusion.

## Contributions
(1) A once-for-all pipeline that animates arbitrary personalized T2Is without specific tuning while preserving their visual quality and domain knowledge. (2) A training strategy separating domain adaptation from motion learning, alleviating quality degradation from imperfect video data. (3) MotionLoRA, a lightweight adaptation to new motion patterns at roughly 30M of extra storage per pattern. Code and pre-trained weights are released (github.com/guoyww/AnimateDiff); the paper was published at ICLR 2024.

## Experimental Setup
Implemented on Stable Diffusion V1.5 with the motion module trained on WebVid-10M; detailed hyperparameters are in the supplementary material. Evaluation uses a diverse set of representative personalized T2Is from Civitai and Hugging Face spanning 2D cartoons to realistic photography, with prompts constructed from each model's trigger words. Baselines are Text2Video-Zero and Tune-a-Video; commercial comparisons use Gen-2 (text-to-video) and Pika Labs (image animation).

## Results
In the user study (Average User Ranking, higher is better) AnimateDiff scores 2.210 on text alignment, 2.280 on domain similarity, and 2.825 on motion smoothness, versus Text2Video-Zero (1.620/2.620/1.560) and Tune-a-Video (2.180/1.100/1.615). CLIP metrics: domain similarity 87.29 and smoothness 98.00 for AnimateDiff versus 84.84/96.57 (Text2Video-Zero) and 80.68/97.42 (Tune-a-Video), with text alignment 31.39. Qualitatively, the motion module transfers across domains and MotionLoRA achieves shot-type control with 20-50 reference videos and 2,000 iterations (about 1-2 hours). An ablation shows a convolutional motion module aligns all frames identical, supporting the temporal-transformer design.

## Limitations
The extracted main text contains no dedicated limitations section; supplementary-only details are not reported in the paper text available here. Stated constraints include that the motion module is compatible only with personalized T2Is originating from the same base T2I, and that raw video datasets such as WebVid-10M contain motion blur, compression artifacts, and watermarks, which the domain adapter mitigates rather than removes. MotionLoRA camera-motion adaptation depends on rule-based augmentation of reference videos.

