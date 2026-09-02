# Text2Grasp: Grasp synthesis by text prompts of object grasping parts

**Authors:** Xiaoyun Chang, Yi Sun  
**Date:** 2024-04-09  
**Identifier:** [arXiv:2404.15189](https://arxiv.org/abs/2404.15189)  
**Zotero item:** `899SN9RJ` ([Zotero](zotero://select/library/items/899SN9RJ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Text2Grasp is a dexterous grasp synthesis method controlled by text prompts that name the object grasping part (e.g., "Grasp the handle of the mug"), which is less ambiguous than intent- or task-level language. It uses a two-stage design: a text-guided conditional diffusion model, TextGraspDiff, generates a coarse MANO grasp from the object point cloud and text, followed by a text-guided contact optimization built on finger perception and text-guided object-part perception. Trained with semi-automatically generated template and LLM-expanded personalized captions, it attains 87.76% grasp part accuracy on OakInk while achieving lower penetration and simulation displacement than GrabNet, and supports task-level and personalized text control without extra manual annotations via LLM decomposition.

## Background and Problem

Controllable grasp synthesis is key to downstream task execution: different tasks need different grasps (a knife is grasped by the handle for cutting, by the blade for safe hand-over). Existing control signals are ambiguous: fixed intent sets (use, pass, twist, lift) and task-level text both fail to specify which object part to grasp — "lifting a mug" may mean handle or body, while "lifting" and "twisting" may share the same grasp of a bottle neck — complicating dataset annotation and model convergence. The paper proposes part-level guidance via the template "Grasp the [Object Part] of the [Object Category]" as a lower-uncertainty control signal, and addresses the lack of part-level text annotations with a semi-automatic labeling scheme, while supporting task-level and personalized text through an LLM that decomposes task descriptions into grasping steps naming object parts.

## Method

(1) Semi-automatic text generation: for each grasp in existing datasets, hand-object contact points are computed and the object part with the most contact points supplies the "[Object Part]" in the template; the LLM expands each template into paraphrased personalized descriptions, from which one is randomly sampled as the training label. (2) TextGraspDiff: a DDPM over a 66-dimensional grasp vector g comprising MANO pose (48), shape (10), hand-object centroid distance (3), and a 5-d finger vector indicating which fingers grasp. The denoising network uses a Transformer backbone with timestep-residual blocks; object and text features come from PointNet++ and pretrained CLIP, fused by a Multi-Modal Attention module (point-cloud feature as query, text feature as key/value) rather than simple addition; training minimizes the L2 error between predicted and ground-truth grasp vectors. (3) Text-guided contact optimization: the predicted pose, shape, and distance are optimized with losses that (a) minimize distance to the object only for the fingers marked 1 in the finger vector (finger perception, avoiding all-finger closed grasps), (b) weight hand contact points toward the targeted object part segmented by a pretrained text-guided segmentation network TextSegNet, and (c) penalize penetration, joint angle violations, and hand self-collision.

## Contributions

1) Text2Grasp, a grasp synthesis method guided by text prompts of object grasping parts, offering more natural interaction and precise grasp control than intent- or task-level signals. 2) A two-stage pipeline combining the TextGraspDiff text-guided diffusion model with a contact optimization process ensuring plausibility and diversity. 3) LLM-based support for task-level and personalized text guidance without additional manual annotations, plus a semi-automatic captioning scheme annotating existing grasp datasets at part level.

## Experimental Setup

Training uses the OakInk shape-based subset (OakShape): 1,308 objects for training and 183 unseen objects for testing, from 1,800 models in 32 categories. Generalization is tested out-of-domain on 180 AffordPose instances (30 each for the six categories shared with OakInk: bottle, disperser, earphone, knife, mug, scissors). Objects are represented by 2,048 surface points. TextGraspDiff trains with Adam (learning rate 1e-4) for 1,000 epochs, batch size 64, T = 100 diffusion steps on a single RTX 4090; optimization uses Adamax for 200 epochs. Metrics: penetration depth (PD), solid intersection volume (SIV), physics-simulation displacement under gravity, diversity via 20-cluster K-means entropy and average cluster size, and grasp part accuracy against the input text. For each test object, 20 prompts based on its parts yield 20 grasps; GrabNet, retrained on OakInk, is the main baseline.

## Results

On OakInk, Text2Grasp with template text reaches PD 0.40, SIV 1.89, displacement 2.49 +/- 2.51, entropy 2.92, cluster size 4.70, and 87.76% part accuracy; with personalized text, PD 0.41, SIV 1.73, and 82.32% accuracy. GrabNet yields PD 0.48, SIV 2.97, displacement 2.84 +/- 2.81, and no text controllability; ground truth has PD 0.11, SIV 0.65. On out-of-domain AffordPose, Text2Grasp (template) gets PD 0.66, SIV 5.05, displacement 2.93 +/- 2.67, 78.53% accuracy, versus GrabNet's PD 0.54, SIV 3.77 — comparable generalization with higher diversity (cluster size 4.88 vs. 2.52). Ablations: a VAE variant scores 77.38% accuracy with SIV 8.44 versus diffusion's 85.25% and SIV 2.82; attention fusion beats feature addition (85.25% vs. 83.44%); global all-finger optimization cuts SIV by 35.46% and displacement by 20.33% but limits diversity, while the proposed finger-perception plus part-perception optimization reaches 87.76% accuracy and outperforms GrabNet's RefineNet in balancing penetration and displacement.

## Limitations

In the discussion, the authors note that part control transfers only among seen categories: for unseen categories such as a faucet, the method can generate a reasonable grasp but cannot identify the correct contact part because the training data lack such categories, and a grasp dataset with more categories is not currently available. They also note the method performs task-level static grasp synthesis only; correct part grasping is just the first step of a task, and dynamic object manipulation remains future work.
