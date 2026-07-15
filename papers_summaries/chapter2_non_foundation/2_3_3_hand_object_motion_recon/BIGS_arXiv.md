# BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting

## Summary
Reconstructs bimanual hand-object interactions from monocular video using 3D Gaussian Splatting (3DGS), achieving faster optimization and higher-quality rendering compared to NeRF-based methods while handling two hands and an unknown object.

## 1. Problem and Setting
- Joint reconstruction of two hands and an unknown object from monocular video.
- Input: monocular RGB video; output: MANO meshes for both hands per frame + object 3D Gaussian representation + per-frame poses.
- Bimanual interaction (two hands manipulating one or more objects), template-free, category-agnostic. Video-based per-scene optimization.

## 2. Core Method
- Replaces NeRF-based object representations (used in HOMAN, HOLD) with 3D Gaussian Splatting for the object:
  - The object is represented as a set of 3D Gaussians (position, covariance, color, opacity) in a canonical frame.
  - Per frame, the Gaussians are rigidly transformed by the predicted object pose and rendered via splatting.
- Hands are still represented as MANO meshes, rasterized per frame.
- Joint optimization: MANO parameters + object Gaussian parameters + per-frame object poses are optimized to minimize photometric loss.
- 3DGS enables much faster rendering and optimization than NeRF-based volumetric rendering (minutes vs. hours).

## 3. Knowledge, Supervision, and Assumptions
- Training data: per-video optimization (test-time only).
- Supervision: RGB pixels, 2D hand keypoints, optional object masks.
- Uses MANO for both hands.
- Assumes object is rigid; two-hand interaction; sufficient viewpoints captured in video.

## 4. Experiments and Findings
- Datasets: HOI4D, ARCTIC, custom bimanual captures.
- Metrics: PSNR, SSIM (rendering); Chamfer Distance (object); MPJPE (hands).
- 10-100x faster optimization than NeRF-based methods (HOMAN, HOLD) with comparable or better visual quality. Handles bimanual interactions robustly.

## 5. Strengths and Limitations
### Strengths
- 3DGS representation dramatically accelerates both optimization and rendering.
- Explicit point-based representation enables easier geometry extraction.
- Handles bimanual interactions naturally.

### Limitations
- 3DGS may produce floaters or noisy geometry in unseen regions.
- Gaussian optimization can be sensitive to initialization.
- Still a per-video optimization (not feed-forward).
- Requires good initial hand pose estimates.

## 6. Takeaway
BIGS demonstrated that 3D Gaussian Splatting is a superior representation for hand-object reconstruction compared to NeRF, offering faster optimization while maintaining quality. This shift from NeRF to 3DGS mirrors the broader trend in dynamic scene reconstruction and has been rapidly adopted by follow-up hand-object works.
