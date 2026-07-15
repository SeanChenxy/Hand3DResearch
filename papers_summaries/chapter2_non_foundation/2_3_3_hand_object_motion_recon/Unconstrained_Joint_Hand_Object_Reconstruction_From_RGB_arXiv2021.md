# Towards Unconstrained Joint Hand-Object Reconstruction From RGB Videos (HOMAN)

## Summary
HOMAN jointly reconstructs 3D hand and object meshes from monocular RGB video without requiring known object templates, using a combination of MANO for hands and a learnable neural implicit shape for objects, optimized per-video.

## 1. Problem and Setting
- Joint 3D reconstruction of hand pose and object shape from a monocular RGB video.
- Input: RGB video; output: MANO hand mesh per frame + object 3D shape (implicit/NeRF) + object 6D pose per frame.
- Template-free; video input; static camera. Both hand and object reconstructed jointly. The object is unknown and category-agnostic.

## 2. Core Method
- Per-video optimization framework with three components optimized jointly:
  1. MANO hand: per-frame pose and shape parameters.
  2. Object NeRF: a canonical volumetric representation (density + color) of the object.
  3. Object poses: per-frame 6D rigid transformations mapping canonical object to world coordinates.
- Photometric loss across all video frames drives the optimization, comparing rendered pixels (combining hand mesh rasterization and object NeRF volume rendering) against the input RGB.
- Hand and object rendering are composited via depth ordering to handle occlusions correctly.
- Additional losses: 2D hand keypoint reprojection, hand-object non-penetration (via collision detection), temporal smoothness on hand poses.

## 3. Knowledge, Supervision, and Assumptions
- Training data: per-video test-time optimization (no offline training of reconstruction model).
- Supervision: RGB pixel values, 2D hand keypoints (from MediaPipe or similar), optional object masks.
- Uses MANO for hand.
- Assumes object is rigid; camera is static; hand moves the object sufficiently to expose multiple views; lighting is reasonably consistent.

## 4. Experiments and Findings
- Datasets: HO3D, EPIC-KITCHENS (in-the-wild egocentric), custom captures.
- Metrics: Chamfer Distance (object), MPJPE (hand), PSNR (novel view synthesis).
- First method to demonstrate plausible joint hand-object reconstruction from in-the-wild videos without object templates. Works on egocentric and third-person videos.

## 5. Strengths and Limitations
### Strengths
- Fully template-free: no object CAD models or category priors needed.
- Joint optimization ensures hand and object are spatially consistent.
- Works on diverse real-world videos (HO3D, EPIC-KITCHENS).
- Unified NeRF + mesh rendering framework.

### Limitations
- Per-video optimization is very slow (hours per sequence).
- Object reconstruction quality is limited to viewpoints seen in the video — invisible regions are poorly reconstructed.
- Struggles with small objects, fast motion, and heavy occlusion.
- Static lighting assumption; color constancy may fail under changing illumination.

## 6. Takeaway
HOMAN was a landmark paper that showed joint hand-object reconstruction is feasible from monocular video without any object priors, by combining MANO-based hand tracking with NeRF-based object modeling in a unified optimization framework. It established the per-video optimization paradigm that many later works built upon.
