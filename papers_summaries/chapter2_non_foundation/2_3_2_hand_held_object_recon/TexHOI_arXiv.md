# TexHOI: Reconstructing Textures of 3D Unknown Objects in Monocular Hand-Object Interaction Scenes

## Summary
Extends hand-held object reconstruction to include high-fidelity texture recovery by separately modeling albedo, lighting, and hand-object shadows within a physically-based rendering framework.

## 1. Problem and Setting
- Reconstruct both 3D geometry AND texture of an unknown hand-held object from a monocular RGB video.
- Input: monocular video; output: 3D object mesh with albedo texture + hand pose per frame + environment lighting.
- Template-free. Unlike prior works that only reconstruct geometry, this method recovers object appearance by explicitly modeling the image formation process (albedo, lighting, shadows).

## 2. Core Method
- Builds on existing hand-object tracking (MANO + object pose estimation) to obtain per-frame geometry.
- For texture reconstruction, models the scene with a physically-based decomposition:
  - Object albedo: a canonical texture map learned via an MLP.
  - Environment lighting: estimated as spherical harmonics.
  - Hand shadow modeling: explicitly computes shadows cast by the hand onto the object using ray tracing against the MANO mesh.
- The rendering equation combines these components to produce the final color for each pixel, which is compared against the observed RGB.
- Texture is optimized across all video frames where the object surface point is visible (occlusion-aware).

## 3. Knowledge, Supervision, and Assumptions
- Training data: per-video test-time optimization.
- Supervision: RGB pixel values across video frames; 2D hand keypoints for pose initialization.
- Uses MANO for both hand geometry and shadow computation.
- Assumes object is rigid with Lambertian reflectance; lighting can be approximated by low-order spherical harmonics; hand pose tracking is reasonably accurate.

## 4. Experiments and Findings
- Datasets: HO3D (real), custom captures.
- Metrics: PSNR, SSIM, LPIPS for texture quality; qualitative evaluation of recovered albedo maps.
- Produces convincingly textured 3D object models from monocular video. Shadow modeling significantly improves texture quality by preventing hand shadows from being "baked into" the albedo.

## 5. Strengths and Limitations
### Strengths
- First method to recover high-quality textures (not just geometry) for unknown hand-held objects.
- Physically-based shading decomposition disentangles albedo from lighting and shadows.
- Produces relightable 3D object models.

### Limitations
- Lambertian reflectance assumption limits applicability to specular objects.
- Shadow ray tracing against MANO mesh adds computational cost.
- Texture quality bounded by hand tracking accuracy and viewpoint coverage.
- Only single-hand, rigid-object scenarios.

## 6. Takeaway
TexHOI addressed the under-explored problem of appearance recovery in hand-object reconstruction, showing that explicit physical modeling of shadows and lighting is crucial for clean texture extraction. This work points toward the goal of creating complete, appearance-ready 3D assets from casual hand-held object videos.
