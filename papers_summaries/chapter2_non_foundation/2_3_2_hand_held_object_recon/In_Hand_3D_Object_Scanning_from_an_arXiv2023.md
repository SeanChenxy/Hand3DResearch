# In-Hand 3D Object Scanning from an RGB Sequence

## Summary
Reconstructs an unknown object's 3D shape, appearance, and the hand-object motion from a monocular RGB video using a NeRF-based neural implicit representation that simultaneously models the object, hand, and their dynamic interaction.

## 1. Problem and Setting
- Jointly reconstruct the 3D shape and appearance of an unknown object held by a hand, along with the time-varying hand-object poses, from a monocular RGB video.
- Input: monocular RGB video; output: object 3D shape (implicit density field), object appearance (color field), hand pose per frame (MANO), object pose per frame.
- Template-free, joint hand-object tracking and reconstruction. The camera is stationary; the hand moves the object.

## 2. Core Method
- A NeRF-based framework that models the scene as two components: (1) the hand, represented by a canonical MANO mesh deformed per frame; (2) the object, represented by a canonical NeRF that is rigidly transformed per frame.
- Joint optimization of: hand MANO parameters per frame, object 6D pose per frame, and the shared object NeRF (density + color).
- Hand-aware ray sampling: rays passing through the hand are handled via a volumetric hand model that contributes density, preventing the hand from being reconstructed as part of the object.
- Pose refinement: both hand and object poses are refined jointly during NeRF training through differentiable rendering.
- Does not assume known camera extrinsics — the object pose relative to the hand is the unknown variable.

## 3. Knowledge, Supervision, and Assumptions
- Training data: per-video test-time optimization (no offline training).
- Supervision: RGB pixel values (photometric loss); optional 2D hand keypoints for initialization.
- Uses MANO for hand geometry.
- Assumes object is rigid; camera is stationary; hand-object motion mostly consists of object rotation in-hand; initial hand pose estimates are reasonably good.

## 4. Experiments and Findings
- Datasets: HO3D, custom captured in-hand scanning sequences.
- Metrics: Chamfer Distance, PSNR for novel view synthesis.
- Produces high-quality textured 3D object reconstructions from monocular video. Hand-aware modeling significantly reduces artifacts where the hand occludes the object.

## 5. Strengths and Limitations
### Strengths
- Unified NeRF framework jointly optimizes hand tracking and object reconstruction.
- Hand-aware ray sampling effectively disentangles hand and object.
- Produces both geometry and appearance (textured mesh via NeRF density extraction).

### Limitations
- Per-video optimization is slow (hours per sequence).
- Requires significant object rotation in the video for complete coverage.
- Struggles with fast motion or motion blur.
- Assumes static camera and lighting.

## 6. Takeaway
This work demonstrated that NeRF-based joint optimization can simultaneously track hands and reconstruct unknown objects from monocular video, producing appearance-aware 3D models. The hand-aware volumetric modeling approach established a template followed by many subsequent video-based hand-object reconstruction methods.
