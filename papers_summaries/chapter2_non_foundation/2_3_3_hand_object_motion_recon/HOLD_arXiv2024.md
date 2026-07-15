# HOLD: Category-Agnostic 3D Reconstruction of Interacting Hands and Objects from Video

## Summary
HOLD reconstructs 3D hands and unknown objects from monocular video using a compositional implicit representation with explicit hand-object contact modeling, producing state-of-the-art results without any object templates or category priors.

## 1. Problem and Setting
- Joint 3D reconstruction of interacting hands and manipulated objects from monocular RGB video.
- Input: monocular RGB video; output: MANO hand meshes per frame + object 3D shape (implicit SDF) + object 6D pose per frame.
- Template-free, category-agnostic, single-hand or bimanual interaction. Camera may be static or in mild motion.

## 2. Core Method
- A per-video optimization framework with compositional neural implicit representations:
  1. Hand representation: MANO parametric model, with per-frame pose optimized.
  2. Object representation: a canonical SDF modeled by an MLP, shared across all frames, with per-frame 6D rigid transformation.
  3. Contact modeling: key innovation — an explicit hand-object contact field that encourages the hand to touch (but not penetrate) the object at contact regions. The contact prior is learned from a large-scale hand-object interaction dataset.
- Renders both hand (mesh rasterization) and object (volumetric SDF rendering) for photometric loss.
- Additional losses: 2D hand keypoints, hand mask, temporal smoothness, contact consistency.
- Can optionally take a coarse object mask as initialization.

## 3. Knowledge, Supervision, and Assumptions
- Training data: the contact prior is pretrained on hand-object interaction data (GRAB, ARCTIC); the per-video reconstruction itself is test-time optimization.
- Supervision: RGB pixels, 2D hand keypoints, optional object masks, contact priors.
- Uses MANO for hand.
- Assumes object is rigid; hand-object interaction involves meaningful contact; video captures sufficient viewpoints; lighting is consistent.

## 4. Experiments and Findings
- Datasets: HO3D, HOI4D, ARCTIC, in-the-wild videos.
- Metrics: Chamfer Distance, F-score (object); MPJPE (hand); PSNR (rendering).
- Outperforms HOMAN and other video-based methods on both hand and object metrics. The contact prior significantly improves reconstruction, especially for heavily occluded object regions. Handles bimanual interaction well.

## 5. Strengths and Limitations
### Strengths
- State-of-the-art template-free hand-object reconstruction at the time.
- Contact prior provides physically meaningful regularization.
- Handles bimanual interaction.
- Works on diverse real-world videos.

### Limitations
- Per-video optimization is slow (~1 hour per sequence).
- Contact prior is trained on specific datasets and may not generalize to novel interaction types.
- Struggles with very small or thin objects.
- Object SDF representation may produce artifacts in completely unobserved regions.

## 6. Takeaway
HOLD represented a significant advance by incorporating learned contact priors into the per-video optimization framework, bridging the gap between purely geometric methods and data-driven approaches. The contact-aware compositional representation proved that modeling physical interaction is as important as visual reconstruction quality.
