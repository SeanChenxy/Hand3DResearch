# ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions

## Summary
ArtHOI is an optimization-based framework that integrates and refines priors from multiple foundation models (3D generative models, MLLMs) to reconstruct 4D human-articulated-object interactions from a single monocular RGB video, addressing the highly ill-posed problem of articulated HOI without requiring pre-scanning or multi-view input, and contributes two new datasets (ArtHOI-RGBD, ArtHOI-Wild) for evaluation.

## 1. Problem and Setting
- 4D reconstruction of human-articulated-object interactions from a single monocular RGB video.
- Input: monocular RGB video of a hand interacting with an articulated object (e.g., laptop, scissors, pliers, drawer).
- Output: 4D articulated object reconstruction (per-part pose trajectory) and hand pose over time.
- Task: hand-articulated-object interaction reconstruction; uses foundation model priors (shape, contact reasoning).

## 2. Core Method
- Optimization-based framework that integrates and refines priors from multiple foundation models.
- Novel methodologies to resolve the inherent inaccuracies and physical unreality of these priors:
  1. Adaptive Sampling Refinement (ASR): optimizes the object's metric scale and pose for grounding its normalized mesh in world space.
  2. MLLM-guided hand-object alignment: uses contact reasoning from a Multimodal Large Language Model as constraints of hand-object mesh composition optimization.
- How FM priors are injected: 3D generative FMs provide initial articulated object shape; MLLMs provide contact reasoning for hand-object alignment.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models: 3D generative model for articulated object shape; MLLM for contact reasoning.
- Domain knowledge: articulated object URDF structure; hand-object physical constraints.
- Training data: pretraining on general 3D object datasets; the optimization is test-time.
- New datasets: ArtHOI-RGBD (lab-captured) and ArtHOI-Wild (in-the-wild).
- Assumption: articulated object category can be recognized by MLLM; the initial 3D shape prior is approximately correct up to metric scale.

## 4. Experiments and Findings
- Datasets: ArtHOI-RGBD and ArtHOI-Wild (introduced), plus HOI4D and similar.
- Metrics: per-part pose error, hand-object contact accuracy, articulation angle error, rendering quality.
- Validates robustness and effectiveness across diverse objects and interactions.
- The ASR and MLLM-guided alignment components both contribute to final accuracy.

## 5. Strengths and Limitations
### Strengths
- First framework specifically for monocular 4D articulated HOI reconstruction.
- Multiple foundation model priors combined with principled optimization.
- Adaptive Sampling Refinement handles the metric-scale ambiguity of generated 3D shapes.
- MLLM-based contact reasoning provides physically grounded alignment.
- New datasets enable systematic evaluation of articulated HOI.

### Limitations
- Optimization-based pipeline is slow.
- Depends on the quality of the initial 3D shape prior (which may be inaccurate for unusual articulated objects).
- MLLM contact reasoning is approximate.
- The optimization may get stuck in local minima.

## 6. Takeaway
ArtHOI tackles the unexplored but significant problem of 4D articulated HOI reconstruction from a single monocular video by combining multiple foundation model priors (3D generation, MLLM reasoning) with novel optimization techniques. The work represents a sophisticated "FM-orchestrated" approach where the strengths of different foundation models are combined to solve a problem no single FM can handle alone, contributing both a method and new datasets that advance the field.
