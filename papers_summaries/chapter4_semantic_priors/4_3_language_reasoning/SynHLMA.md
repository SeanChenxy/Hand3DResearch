# SynHLMA: Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation

**Authors:** Zhi Wang, Yuyan Liu, Liu Liu, Li Zhang, Ruixuan Lu, Dan Guo  
**Date:** 2025-10-29  
**Identifier:** [arXiv:2510.25268](https://arxiv.org/abs/2510.25268)  
**Zotero item:** `MHAMZKS3` ([Zotero](zotero://select/library/items/MHAMZKS3))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

SynHLMA is a language-guided generation framework for Hand Articulated Object Interaction (HAOI): given an articulated object point cloud and a textual instruction, it autoregressively generates manipulation sequences (grasp plus articulation, e.g., opening a laptop or closing scissors). It tokenizes object joint states and hand poses with two modular VQ-VAEs into a hierarchical discrete representation, aligns these tokens with language via a LoRA fine-tuned Vicuna-7B model, and trains with an articulation-aware objective. On the new HAOI-Lang dataset (256 objects, 51,200 captioned sequences) it outperforms HOIGPT, Text2HOI, SemGrasp, and other baselines on HAOI generation, prediction, and interpolation, and transfers to a ShadowHand robot hand in simulation.

## Background and Problem

Language-guided grasp synthesis for rigid objects is well studied, but articulated objects require modeling not only a stable grasp but also temporally coherent manipulation along articulation — using scissors demands coordinated opening/closing control. Existing approaches fall short: robotic-hand methods lack human-hand realism, skeleton-driven methods neglect physically grounded contact, contact-centric methods struggle to couple language semantics with articulated dynamics, and diffusion models degrade on long horizons due to weak structural priors. Articulated interaction adds configuration-dependent contacts, joint-coupled motion dependencies, and evolving affordances, often causing hand-motion/object-deformation inconsistency. The paper defines HAOI sequence generation under language-guided intent with three tasks: generation (text plus object state to full sequence), prediction (first 20% observed, remaining 80% predicted), and interpolation (a 40-50% gap completed coherently). The key insight is that articulated manipulation exhibits discrete structural regularities analogous to linguistic tokens and grasp taxonomies.

## Method

(1) Discrete Articulated Manipulation Representation: a single-layer VQ-VAE quantizes object joint parameters (revolute/prismatic) into token <j>, while a multi-level VQ-VAE decomposes MANO grasp parameters (global rotation, 90-dim pose, translation) into hierarchical tokens <g, l, r> for global configuration, local finger articulation, and refinement residuals; decoding is coarse-to-fine, conditioned on <j>, with final parameters obtained by residual composition. (2) HAOI Manipulation Language Model: with the VQ-VAEs frozen, manipulation tokens wrapped in special markers are aligned with text embeddings through a linear projection and a fine-tuned Vicuna-7B-v1.5 with LoRA, trained in two stages (multimodal alignment, then instruction tuning) as next-token prediction. (3) Articulation-aware objective: a penetration loss on hand-object mesh interpenetration, a joint reconstruction loss for object state, hierarchical reconstruction losses at the three levels, VQ-VAE commitment losses, next-token NLL, and a temporal pose consistency loss matching inter-frame hand and object joint motion (rotational via Rodrigues-form matrices, translational via displacements).

## Contributions

1) A hierarchical discrete tokenization scheme for articulated manipulation enabling structured, controllable sequence generation. 2) A manipulation language model aligning tokenized HAOI sequences with language in a shared semantic space, supporting generation, prediction, and interpolation in one autoregressive formulation. 3) An articulation-aware training objective enforcing geometric validity, joint-state alignment, and temporal coherence. 4) HAOI-Lang, a new language-annotated HAOI dataset built with RaiSim physics simulation, reinforcement-learning grasping (following GraspXL), and GPT-4 annotations with human refinement.

## Experimental Setup

HAOI-Lang covers seven articulated categories (stapler, box, laptop, scissors, cabinet, dishwasher, eyeglasses) with 256 instances and 51,200 episodes, each instance providing 200 episodes with unique GPT-4 captions; only single-joint-type objects with predefined motion ranges are considered. Four codebooks (each 1024x512) discretize hand and object representations. Metrics: FID, Diversity, MMDist, Interaction Volume (IV), ADE, FDE, and Codebook Update Coverage (CUC). Baselines: HOIGPT, Text2HOI, SemGrasp, NL2Contact, plus motion-generation models T2MGPT, MotionGPT, and TM2T.

## Results

On HAOI generation, SynHLMA reaches FID 14.121, Diversity 40.484, MMDist 12.793, IV 5.919, ADE 0.976, FDE 1.147 — versus HOIGPT's FID 19.040 and Diversity 26.498 — a 4.919% FID improvement and 12.530% diversity gain. For prediction it attains FID 21.739 vs. 36.379 (14.64% improvement) and Diversity 48.691 vs. 29.119 (19.572% gain); for interpolation, FID 25.225 vs. 34.956 (9.731% reduction) and Diversity 44.012 vs. 24.052 (19.969% gain). Ablations: removing the geometric loss raises FID to 15.872 and IV to 6.452; removing the temporal loss raises ADE to 1.112; the full token design <g,l,r,j> achieves FID 0.699, beating shared-codebook and semantic-free variants. Backbone ablations show Gemma at FID 22.576, Llama at 51.911, and LoRA rank r=8 degrading to FID 126.954. Sequences also transfer to a ShadowHand in RaiSim via keypoint-fitting optimization for dexterous robotic execution.

## Limitations

The paper assumes each object contains only a single joint type (revolute or prismatic, possibly multiple joints of that type) with predefined operational ranges, so compound/multi-joint-type articulation is outside the current setting. The conclusion notes future work toward "more fine-grained and coordinated bimanual manipulation," indicating the framework currently addresses single-hand manipulation. The authors otherwise do not report a dedicated limitations section beyond these remarks.
