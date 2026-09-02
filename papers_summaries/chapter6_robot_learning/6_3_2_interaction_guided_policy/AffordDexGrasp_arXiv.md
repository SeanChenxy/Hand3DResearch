# AffordDexGrasp: Open-set Language-guided Dexterous Grasp with Generalizable-Instructive Affordance

**Authors:** Yi-Lin Wei, Mu Lin, Yuhao Lin, Jian-Jian Jiang, Xiao-Ming Wu, Ling-An Zeng, Wei-Shi Zheng  
**Date:** 2025 (ICCV; the PDF is the CVF open-access version, no explicit date recorded in Zotero)  
**Identifier:** no identifier recorded in Zotero metadata or PDF text  
**Zotero item:** `AZ3EXQQK` ([Zotero](zotero://select/library/items/AZ3EXQQK))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Language-guided dexterous grasp generation fails on unseen categories because of a large gap between high-level language semantics and low-level hand actions. AffordDexGrasp bridges this gap with a "generalizable-instructive affordance": a per-point map of all graspable regions sharing the same semantic intention, generalizing via category-agnostic cues while still guiding high-DoF hands. Two cascaded flow-matching models with MLLM pre-understanding and test-time optimization surpass prior language-guided and contact-based methods in simulation and the real world.

## Background and Problem
The paper introduces open-set language-guided dexterous grasp: given a scene point cloud, RGB images, and a language command, generate a dexterous grasp pose (rotation, translation, joint angles) for seen and unseen categories. Prior language-conditioned generative models work only within known categories, while open-set generalization matters because dexterous-hand data collection is expensive. The core obstacle is the semantic-to-action gap, with a representational trade-off: fine-grained contact maps generalize poorly, whereas coarse object parts cannot steer high-DoF hands (the same mug body supports different intentions).

## Method
The affordance unions the contact maps of grasps sharing intention, contact part, and grasp direction on an object, smoothed by a Gaussian filter. The pipeline has four stages: (1) an MLLM (GPT-4o) pre-understanding stage extracts category, intention, contact part, and a discretized six-direction grasp direction into a compact sentence; (2) Affordance Flow Matching (AFM) generates the affordance map from PointNet++, CLIP, and direction features via a Perceiver IO transformer; (3) Grasp Flow Matching (GFM) generates hand poses conditioned on affordance, language, and direction, trained with pose regression, hand Chamfer, and fingertip losses, excluding penetration loss from training; (4) non-parametric test-time optimization applies affordance contact, penetration, and joint-limit objectives.

## Contributions
- The new task of open-set language-guided dexterous grasp and the language-action gap behind it.
- The generalizable-instructive affordance representation, validated against contact maps and object parts.
- A two-stage flow-matching framework (AFM and GFM) with MLLM pre-understanding and affordance-guided test-time optimization.
- An open-set tabletop language-guided dexterous grasp dataset with scene-level data.

## Experimental Setup
The dataset contains 33 categories, 1,536 objects, 1,909 scenes, and 43,504 grasps for Shadow Hand and Leap Hand, with two splits (Open Sets A and B; 80% of seen-category objects for training, 20% plus all unseen categories for testing). Metrics cover intention consistency (FID, Chamfer Distance, R-Precision Top-1), grasp quality (Isaac Gym success rate, Q1, penetration depth), and diversity. Baselines: ContactGen, Contact2Grasp, GraspCVAE, SceneDiffuser, DexGYS; real tests use a Leap Hand on a Kinova Gen3 arm.

## Results
On Open Set A, AffordDexGrasp reaches R-Precision Top-1 0.480 and success rate 45.1% versus DexGYS at 44.2% and 0.317; on Open Set B it attains 38.9% success and 0.532 Top-1 versus 35.2% and 0.294. In the close set, R-Precision Top-1 improves from 0.590 (DexGYS) to 0.779. One-shot learning lifts novel-category Top-1 to 0.586 versus 0.342 without affordance. Across eight real-world object-command settings of 10 attempts each, it succeeds 66 times versus 39 for DexGYS. Ablations confirm the affordance is necessary for unseen-category generalization.

## Limitations
The paper has no dedicated limitations section, but it acknowledges the trade-off between generalization and instruction, that penetration loss during training harms intention alignment and diversity, and that excessive grasp diversity can produce unnatural postures.
