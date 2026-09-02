# Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations

**Authors:** Yucheng Hu, Yanjiang Guo, Pengchao Wang, Xiaoyu Chen, Yen-Jen Wang, Jianke Zhang, Koushil Sreenath, Chaochao Lu, Jianyu Chen  
**Date:** 2025-05-04 (ICML 2025 Spotlight)  
**Identifier:** [arXiv:2412.14803](https://arxiv.org/abs/2412.14803)  
**Zotero item:** `KZI3TTUD` ([Zotero](zotero://select/library/items/KZI3TTUD))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
VPP uses video diffusion models (VDMs) as the visual backbone for robot policies, hypothesizing that their representations naturally contain both current scene information and predicted future dynamics. It learns an implicit inverse dynamics model conditioned on predicted future representations inside the VDM, and fine-tunes the pre-trained video foundation model on robot datasets together with Internet human manipulation data for more precise prediction. VPP reports an 18.6% relative improvement over the previous state of the art on the CALVIN ABC-D generalization benchmark and a 31.6% success-rate increase on complex real-world dexterous manipulation.

## Background and Problem
Conventional vision encoders pre-trained with single-image reconstruction or two-image contrastive learning capture static appearance but not dynamics, which are vital for embodied tasks. The paper targets generalist manipulation policies whose visual representations anticipate how the scene evolves, evaluated on long-horizon generalization and high-dimensional dexterous manipulation.

## Method
A pre-trained video foundation model is fine-tuned on robot datasets plus Internet human manipulation video so its representations predict precise futures. On top of these predictive representations, VPP learns an implicit inverse dynamics model that conditions action generation on the predicted future features inside the VDM, coupling visual foresight with control.

## Contributions
- Using video diffusion model representations — which bundle static semantics with predicted dynamics — as the visual basis of a generalist policy.
- An implicit inverse dynamics model conditioned on VDM-predicted future representations.
- Fine-tuning the video backbone with Internet human manipulation data in addition to robot data, yielding state-of-the-art CALVIN and real-world dexterous results.

## Experimental Setup
Simulation evaluation uses the CALVIN benchmark in the challenging ABC→D setting (trained in ABC environments, evaluated in unseen D, using only language-annotated ABC data, following GR-1's protocol) and MetaWorld for precision multi-task manipulation. Real-world experiments cover complex dexterous hand manipulation. Baselines include prior state-of-the-art policies and ablations over video pre-training and Internet data. Trial counts and complete task lists are not reproduced from the available evidence.

## Results
- CALVIN ABC-D: 18.6% relative improvement over the previous state of the art.
- Real-world dexterous manipulation: 31.6% higher success rate over the strongest baseline.
- VPP consistently outperforms compared baseline algorithms across the two simulated benchmarks in the reported comparisons.

## Limitations
The approach depends on the availability and quality of a pre-trained video foundation model and on Internet human manipulation video for fine-tuning; performance when such data are unavailable is not characterized in the available evidence. Quantitative per-task tables and failure analyses beyond the headline numbers are not reproduced from the available evidence.
