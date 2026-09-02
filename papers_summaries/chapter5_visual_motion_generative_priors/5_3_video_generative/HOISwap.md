# HOI-Swap: Swapping Objects in Videos with Hand-Object Interaction Awareness

**Authors:** Zihui Xue, Mi Luo, Changan Chen, Kristen Grauman  
**Date:** 2024-06-11  
**Identifier:** [arXiv:2406.07754](https://arxiv.org/abs/2406.07754)  
**Zotero item:** `3WZ3AN3K` ([Zotero](zotero://select/library/items/3WZ3AN3K))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HOI-Swap is a two-stage, self-supervised diffusion framework that swaps the object a hand is touching in a video with a new object given a single reference image, adjusting the hand's grasp to the new object and allowing users to control how closely the edited video follows the source motion.

## Background and Problem

The paper addresses precise swapping of hand-interacting objects in videos: given a source video, a mask locating the source object, and one reference object image, generate an edited video where the reference object replaces the original while remaining realistically manipulated. Existing video diffusion editors fail here for three reasons: (1) HOI awareness — inpainting methods such as Paint by Example and AnyDoor copy the original grasp or occlude hands unnaturally when object shape or functionality changes; (2) spatial alignment — the reference object may appear in an arbitrary pose and must be re-oriented to where the hand is ready to grasp; (3) temporal alignment — motion in HOI videos is tied to object properties, yet prior editors enforce a rigid (typically 100%) degree of motion alignment using structural signals like optical flow or depth that encode the original object's shape. The authors pose the task as video inpainting and note it also enables synthetic data generation for robot learning.

## Method

HOI-Swap trains two stages separately, fully self-supervised (real edited-video pairs are unavailable). Stage I is an image latent diffusion model (UNet) for object swapping in a single frame: the square bounding-box-masked frame is VAE-encoded and channel-concatenated with the noised latent, while a DINO feature (768-d) of the reference object is injected via cross-attention. Training pairs are built from the same video: the reference image comes from a different random frame and is strongly augmented (resize, flip, rotation, perspective), with a text-guided inpainting model filling hand-occluded object regions. Stage II propagates the single-frame edit to the whole sequence: r% points sampled within the object region of a randomly chosen conditioning frame are tracked with RAFT optical flow to warp a sparse, incomplete sequence; a video LDM (2D UNet inflated with temporal layers) conditioned on the warped-sequence latent plus a CLIP feature of the anchor frame learns to fill gaps. Varying r from 0 to 100 during training yields controllable motion alignment at inference: fewer points for large shape/function changes, more for faithful motion transfer.

## Contributions

1. First framework for swapping in-contact objects in videos with HOI awareness, addressing HOI awareness, spatial alignment, and temporal alignment jointly.
2. A fully self-supervised two-stage training recipe requiring no paired edited videos; stage I also works as a standalone image editor.
3. Controllable motion guidance via sparsity of flow-tracked sampled points, letting users tune motion alignment to object changes — unlike prior editors with fixed alignment.

## Experimental Setup

Training uses HOI4D and EgoExo4D: 106.7K frames for stage I (512x512, 25K steps) and 26.8K 2-second clips for stage II (14 frames at 7 fps, 256x256, 50K steps). Evaluation covers 1,250 source images x4 reference objects (5,000 edits; 80% hand-present) and 25 source videos x4 objects (100 edits), including zero-shot sources from EPIC-Kitchens and TCN Pouring. Image baselines: Paint by Example, AnyDoor, Affordance Diffusion; video baselines: per-frame image editing, image editing + AnyV2V, and VideoSwap. Metrics: contact agreement, hand agreement, hand fidelity, VBench subject consistency and motion smoothness, and a 15-participant user study (260 image edits, 100 video edits).

## Results

Image editing: HOI-Swap reaches 87.9 contact agreement, 79.8 hand agreement, 97.4 hand fidelity, and 72.1% user preference, versus 15.6% for AnyDoor, 7.6% for Affordance Diffusion, and 4.5% for PBE. Video editing: 92.4 subject consistency, 98.2 motion smoothness, 78.6 contact agreement, 97.6 hand agreement, 93.1 hand fidelity, and 86.4% user preference, versus 1.2% for VideoSwap and AnyV2V and 0.2% for per-frame editing. Baselines persistently failed despite official code and tuned hyperparameters. Ablations show motion-point sparsity controls divergence from source motion, and a one-stage variant fails to preserve reference-object identity, validating the two-stage design.

## Limitations

The paper identifies three limitations: (1) generalization to very different unseen objects — e.g., depicting a hand holding scissors never seen in training requires world knowledge beyond the training data; (2) long videos with complex HOI — the two-stage pipeline assumes interactions remain stable over a short clip, while longer sequences contain multiple distinct, dynamically changing hand interactions; (3) controllability — motion guidance is global, with no spatial support to specify which source-video regions should receive motion transfer.
