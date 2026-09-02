# FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation

**Authors:** Huajian Zeng, Lingyun Chen, Jiaqi Yang, Yuantai Zhang, Fan Shi, Peidong Liu, Xingxing Zuo  
**Date:** 2026-02-13 (arXiv preprint)  
**Identifier:** [arXiv:2602.13444](https://arxiv.org/abs/2602.13444)  
**Zotero item:** `II4NURRU` ([Zotero](zotero://select/library/items/II4NURRU))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
VLA models produce plausible end-effector motions but fail at long-horizon, contact-rich tasks because the underlying hand-object interaction (HOI) structure is not explicitly represented. FlowHOI is a two-stage flow-matching framework generating HOI sequences — hand poses, object poses, and contact states — conditioned on an egocentric observation, a language instruction, and a 3D Gaussian splatting (3DGS) scene reconstruction. It decouples geometry-centric grasping from semantics-centric manipulation and pre-trains a grasping prior on HOI data reconstructed from egocentric videos. On GRAB and HOT3D it attains the highest action-recognition accuracy, a 1.7x higher physics-simulation success rate than the strongest diffusion baseline, and a 40x inference speedup.

## Background and Problem
Household manipulation is interaction-centric: success depends on where contact occurs given scene geometry, how stable contact is preserved, and whether object state changes match the instruction. The paper targets HOI motion generation: given a first egocentric observation, an action description, and a 3DGS scene, generate an N-frame sequence of bi-manual MANO hand states and object poses as an embodiment-agnostic representation retargetable to robots. Challenges include joint geometric consistency and semantic grounding, the 3-7 s inference of diffusion-based HOI generators, and scarce high-fidelity HOI supervision.

## Method
A reconstruction pipeline converts EgoDex egocentric videos into aligned HOI training data via transition detection from wrist motion, object mesh reconstruction, and MANO hand fitting with fingerpad-contact and non-penetration constraints. The grasping stage uses an x-prediction conditional flow-matching model conditioned on object geometry (Basis Point Set), an MLLM-extracted grasp sub-instruction, and the initial state, pre-trained on the reconstructed data. The manipulation stage generates the full HOI sequence with soft inpainting of the grasp prefix and a hard-clamped transition state, conditioned on object geometry, the full instruction, and scene tokens: FPS-sampled Gaussian centroids carrying fused geometric and semantic embeddings, plus a global token from a voxelized occupancy grid. A TMR-style InfoNCE motion-text alignment loss grounds motion in language.

## Contributions
- A two-stage HOI generation framework separating geometry-centric grasping from semantics-centric manipulation with flow matching, with up to 40x speedup over diffusion baselines.
- Semantic grounding via a motion-text alignment loss and hybrid 3D scene tokens.
- A reconstruction pipeline extracting large-scale, high-fidelity HOI data from egocentric videos for a robust grasping prior.

## Experimental Setup
Evaluation uses GRAB (mocap) and HOT3D (real egocentric recordings with reconstructed scenes). Baselines DiffH2O and LatentHOI are retrained with the same splits and T5 text encoders. Metrics cover interpenetration volume/depth (IV/ID), contact ratio (CR), action-recognition accuracy (AR), and physics-based feasibility in Isaac Gym after retargeting to an Allegro Hand (success rate SR, holding time HT), plus inference time. Real-robot tests use two Franka Panda arms with Allegro Hand v5 on pouring, drinking, tilting, and squeezing-dressing tasks.

## Results
On GRAB, FlowHOI reaches IV 10.93 cm3, CR 6.85%, AR 0.95, SR 55.96%, HT 1.50 s, and 0.16 s inference versus DiffH2O (33.03% SR, 6.34 s) and LatentHOI (28.44% SR, 3.57 s). On HOT3D it achieves the highest CR (2.14%) and AR (0.78), with SR 5.33% versus 4.00% and 0.00%. Ablations show egocentric pre-training halves grasp error (0.10 m to 0.06 m), T5 with alignment loss raises AR from 0.89 (CLIP) to 0.94. All four real-robot tasks were retargeted and executed with stable contact.

## Limitations
The authors state that the framework assumes accurate initial hand and object state estimation and degrades under heavy occlusion or unreliable reconstruction, and that generated trajectories are kinematic and contact-consistent, relying on downstream controllers for dynamics and compliance.
