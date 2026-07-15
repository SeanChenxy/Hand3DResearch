# In-Hand 3D Object Reconstruction from a Monocular RGB Video

## Summary
Reconstructs a hand-held object from monocular video by decomposing the problem into explicit hand-object pose tracking and object surface reconstruction with a hybrid SDF-volume rendering approach that handles severe occlusions.

## 1. Problem and Setting
- Reconstruct the 3D geometry of an unknown hand-held object from a monocular RGB video where the hand rotates the object.
- Input: monocular RGB video; output: 3D object mesh + time-varying hand-object relative poses.
- Template-free; camera is static; the hand rotates a rigid object to reveal different viewpoints.

## 2. Core Method
- Two-stage approach:
  - Hand-object pose tracking: estimates MANO hand parameters and object 6D pose per frame using a combination of 2D keypoint detection, hand-object contact constraints, and temporal smoothness.
  - Hybrid SDF-volume reconstruction: the object surface is represented as an SDF in a canonical frame. Given the tracked poses, rays from all frames are used to train the SDF via volumetric rendering, but with a surface-aware sampling strategy that concentrates samples near the SDF zero-level-set.
- Key innovation: unlike pure NeRF approaches, the hybrid SDF-volume representation enables sharper surface reconstruction by concentrating capacity near the surface rather than modeling full volumetric density.
- Explicit pose tracking before reconstruction avoids the challenging joint optimization problem of simultaneous tracking + reconstruction.

## 3. Knowledge, Supervision, and Assumptions
- Training data: per-video test-time optimization (no offline training for reconstruction).
- Supervision: RGB pixel values; 2D hand keypoints from off-the-shelf detectors; optional object mask.
- Uses MANO for hand.
- Assumes object is rigid; hand sufficiently rotates the object; camera is static; reasonably good 2D hand keypoint detection is available.

## 4. Experiments and Findings
- Datasets: HO3D, DexYCB, self-captured in-hand rotation videos.
- Metrics: Chamfer Distance, F-score, PSNR.
- The two-stage decomposition (pose tracking then reconstruction) is more robust than joint optimization. SDF-based surface representation captures finer geometric details than pure NeRF approaches.

## 5. Strengths and Limitations
### Strengths
- Two-stage pipeline is more robust than end-to-end joint optimization.
- SDF-based representation yields sharper surfaces.
- Explicit pose tracking enables handling of larger motions and faster movement.

### Limitations
- Errors in pose tracking propagate to reconstruction (no feedback loop).
- Requires the object to be substantially rotated for complete reconstruction.
- Textureless or specular objects cause tracking failures.
- Static camera assumption.

## 6. Takeaway
Jiang et al. showed that decomposing the hand-object reconstruction problem into sequential tracking-then-reconstruction stages, rather than joint optimization, can improve robustness. This "track first, reconstruct later" strategy is a practical design pattern adopted by many follow-up works.
