# UniHM: Unified Dexterous Hand Manipulation with Vision Language Model

**Authors:** Zhenhao Zhang, Jiaxin Liu, Ye Shi, Jingya Wang  
**Date:** 2026-02-28 (ICLR 2026)  
**Identifier:** [arXiv:2603.00732](https://arxiv.org/abs/2603.00732)  
**Zotero item:** `TGSXM6CB` ([Zotero](zotero://select/library/items/TGSXM6CB))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

UniHM is presented as the first unified framework for dynamic dexterous hand manipulation guided by free-form language commands. Prior language-guided approaches mostly produce static grasp poses and ignore temporal structure, while conventional pipelines depend on object-centric cues or precisely recorded interaction sequences. UniHM couples a Unified Hand-Dexterous Tokenizer mapping heterogeneous hand morphologies into one shared VQ codebook, a compact vision-language model generating manipulation token sequences from instructions, and a physics-guided refinement module optimizing sequences for contact, smoothness, and feasibility. Trained solely on annotated human-object interaction data without teleoperation, it achieves state-of-the-art results on DexYCB and OakInk and higher real-world success rates than baselines.

## Background and Problem

Dexterous hand manipulation requires contact-rich, sequential control coupling semantic intent with precise geometry and physics. Existing methods either optimize static grasp poses from object-centric inputs or translate HOI video into fixed sequences, and language-guided variants such as SemGrasp and AffordDexGrasp remain pose-centric without temporal structure. The task is: given an RGB-D observation and an open-vocabulary instruction, produce a physically feasible multi-step dexterous-hand manipulation sequence following a target object trajectory, across seen and unseen objects and multiple hand morphologies, while removing dependence on expensive teleoperation data.

## Method

The pipeline has three stages. Automated annotation: GPT-4o labels each HOI sequence with five open-vocabulary instructions from keyframes, and Dex-Retargeting transfers MANO poses to five dexterous hands (Shadow, Allegro, SVH, Leap, Panda). The Unified Hand-Dexterous Tokenizer is a shared VQ-VAE codebook where each hand type has its own encoder and decoder mapping into one discrete index space; new morphologies are integrated by distilling a new encoder against a reference encoder, bypassing the non-differentiable quantization step, enabling direct pose translation between hands. Manipulation generation: a CLIPort-style perception module infers the target SE(3) trajectory and Point-SAM segments the object point cloud from RGB-D input; a Qwen3-0.6B vision-language model, conditioned on initial hand pose tokens, target trajectory, object point cloud, and text, autoregressively generates codebook tokens decoded by the target hand. Training uses a progressive masking curriculum replacing ground-truth hand-pose tokens with a learnable mask token until generation is language-only. Physics-guided dynamic refinement solves a frame-by-frame Gauss-Newton problem combining contact energy (signed point-to-plane fingertip distances to the object), a generative prior preserving model intent, and a temporal prior enforcing smooth velocity and acceleration profiles.

## Contributions

- The first unified, language-conditioned framework for dynamic dexterous hand manipulation beyond static grasps, trained only on HOI data.
- A morphology-agnostic token codebook with cross-hand distillation enabling direct token reuse and transfer across robotic and anthropomorphic hands.
- Physics-guided dynamic trajectory optimization fusing contact, generative, and temporal priors for feasible execution.
- A decoupled design where only the CLIPort perception head is adapted under distribution shift, keeping the HOI generator unchanged.

## Experimental Setup

Evaluation uses DexYCB (582K frames, 1,000 sequences, 20 objects) and OakInk (230K multi-view frames, 100 objects), each split 80/20 into seen and unseen sets. Metrics are MPJPE, Final Position and Orientation Location errors (FPL, FOL), FID, Diversity, and real-world success rate. Baselines are TM2T, MDM, FlowMDM, and MotionGPT3, all post-processed with UniHM's physics refinement for fairness. Real-world experiments cover Grab, Pick and Place, Pull and Push, and Open and Close on seen and unseen objects.

## Results

On DexYCB (seen), UniHM attains MPJPE 61.40 versus 74.80 for MotionGPT3, the strongest baseline, with FID 31.24 versus 43.35; on unseen objects MPJPE is 63.56 versus 77.93. On OakInk (seen) it achieves MPJPE 52.73 versus 56.29 and FID 204.91 versus 221.10, with Diversity (165.47) closest to the ground truth (147.40). In real-world trials UniHM reaches 65%, 50%, 60%, and 55% success on the four task categories for seen objects versus at most 30% for the MDM and MotionGPT3 retargeting baselines, and 60%, 35%, 55%, and 45% on unseen objects. Ablations on DexYCB show removing depth input raises MPJPE to 85.47, removing the masking curriculum to 73.41, and removing physical refinement to 65.78, versus 61.40 for the full model.

## Limitations

The authors state that UniHM relies on RGB-D perception without tactile or force sensing, uses simplified energy terms for contact and friction, and covers a limited range of bimanual or tool-use scenarios. Future work targets richer contact priors, scaling the codebook to more morphologies, and online adaptation for sequential contact-rich tasks.
