# Prompt-Propose-Verify: A Reliable Hand-Object-Interaction Data Generation Framework using Foundational Models

**Authors:** Gurusha Juneja, Sukrit Kumar  
**Date:** 2023-12-23  
**Identifier:** [arXiv:2312.15247](https://arxiv.org/abs/2312.15247)  
**Zotero item:** `KTXSGKAN` ([Zotero](zotero://select/library/items/KTXSGKAN))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; the summary was written without full-text extraction, and unavailable details are marked as not reported.  
## Summary
Prompt-Propose-Verify addresses the lack of high-quality, well-described hand-object images for adapting diffusion models to interaction generation. It proposes interaction prompts, generates candidate images with a foundation image model, and verifies the candidates before using them as training data. The verified set is used to fine-tune a diffusion model for hand-object image synthesis. The paper reports better CLIPScore, ImageReward, fidelity, and alignment than the compared baselines, while the available evidence does not provide representative numerical values.

## Background and Problem
General-purpose diffusion models often produce incorrect hand anatomy, implausible contact, or weak correspondence between a hand and the manipulated object. The task is to construct a high-quality training set of hand-object interaction images rather than to infer a 3D pose from an existing image. The framework takes prompt-generation and image-generation resources as input and outputs verified, annotated or quality-controlled HOI images for diffusion-model fine-tuning.

## Method
The framework has three stages. Prompt creates textual descriptions of desired interactions. Propose uses those descriptions with a diffusion generator to produce candidate images. Verify evaluates the candidates for visual quality and hand-object alignment and keeps the acceptable subset. The verified images are subsequently used to adapt a diffusion model to HOI generation.

## Contributions
- A prompt–generation–verification pipeline for constructing HOI training data.
- Quality control based on image fidelity and hand-object semantic alignment.
- Evidence that verified synthetic data can improve diffusion-based HOI image generation.

## Experimental Setup
The paper evaluates a framework-generated HOI dataset and compares fine-tuned generation against baseline systems using CLIPScore, ImageReward, fidelity, and alignment measures. Exact dataset size, split construction, verifier implementation, baseline checkpoints, and numerical results are not reported in the available evidence.

## Results
The paper reports considerable gains over the baselines on the listed image-quality and alignment measures. It also identifies the verification stage as important for retaining useful training samples. The available evidence does not support reporting a specific score or percentage improvement.

## Limitations
Verification can miss subtle anatomical or contact errors, and the resulting model can retain a gap between synthetic and real imagery. The framework depends on the quality of its foundation models and on the coverage of the proposed prompts. Generalization to substantially broader HOI scenarios is not reported in the paper.
