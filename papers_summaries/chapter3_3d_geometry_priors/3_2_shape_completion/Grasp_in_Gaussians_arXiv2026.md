# Grasp in Gaussians: Fast Monocular Reconstruction of Dynamic Hand-Object Interactions

## Summary
GraG is a fast and robust method for reconstructing dynamic 3D hand-object interactions from a single monocular video, replacing heavy neural representations with a compact Sum-of-Gaussians (SoG) representation initialized from a SAM3D pipeline, achieving 6.4x faster reconstruction with 13.4% better object reconstruction and 65%+ reduction in hand per-joint error compared to prior work.

## 1. Problem and Setting
- Fast 3D reconstruction of dynamic hand-object interaction from a monocular video.
- Input: a monocular RGB video of hand-object interaction.
- Output: 3D hand mesh (MANO parameters per frame), 3D object shape represented as a collection of 3D Gaussians, and per-frame hand-object 6D poses.
- Task: dynamic hand-object reconstruction. Classified under shape completion priors because SAM3D (a foundation model for open-vocabulary 3D segmentation) provides the prior for separating and tracking the object from the scene.

## 2. Core Method
- A lightweight Sum-of-Gaussians (SoG) object representation that can be updated efficiently frame-by-frame, replacing slow neural implicit optimization.
- Initialization via SAM3D: SAM3D segments the hand-held object from the first frame in 3D, providing an initial Gaussian set; this is converted into a compact SoG via subsampling.
- For subsequent frames, the Gaussians are transformed according to estimated object motion, and new Gaussians are added for newly visible object surfaces.
- Hand pose is tracked from off-the-shelf monocular hand pose initialization, refined using 2D joint and depth alignment losses.
- How FM prior is injected: SAM3D provides a semantic/geometric prior for what constitutes "the object" vs. the hand and background, enabling robust initialization. Unlike prior methods that use diffusion models for shape hallucination, GraG uses the FM prior for segmentation and initialization, relying on actual multi-view observations for shape completion.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: SAM3D (a 3D-aware segmentation foundation model) for object extraction; monocular depth foundation model for geometric initialization.
- Domain knowledge: hand model (MANO); rigid object motion (with potential extensions); physical contact constraints.
- Training data: SAM3D and depth models are pre-trained on large-scale datasets; GraG does not require HOI-specific training.
- Assumption: object is rigid; video provides sufficient motion for multi-view observation.

## 4. Experiments and Findings
- Datasets: HO3D, DexYCB, and in-the-wild videos.
- Metrics: reconstruction speed (FPS), object shape accuracy (Chamfer distance), hand pose error (MPJPE), and rendering quality.
- Reconstructs temporally coherent hand-object interactions on long sequences 6.4x faster than prior work while improving object reconstruction by 13.4% and reducing hand's per-joint position error by over 65%.
- SAM3D-based initialization is significantly more robust than hand-crafted segmentation methods, especially for diverse object categories and challenging backgrounds.

## 5. Strengths and Limitations
### Strengths
- Dramatically faster than optimization-based methods; suitable for real-time applications.
- SoG representation is lightweight, editable, and supports real-time rendering.
- SAM3D prior enables robust, open-vocabulary object segmentation.
- Progressive reconstruction naturally handles newly visible surfaces.
- Strong quantitative improvements (6.4x speedup, 13.4% better object, 65% lower hand error).

### Limitations
- Does not hallucinate unseen object back-faces (only reconstructs what is observed), unlike diffusion-based methods.
- Object shape completeness depends on the diversity of observed views.
- Gaussian representation may not capture fine geometric details as well as neural implicit methods.
- Relies on accurate camera pose/motion estimation.
- Less suitable for objects that remain in a fixed pose throughout the video.

## 6. Takeaway
GraG demonstrates an important alternative in the FM-prior design space: rather than using diffusion models to "hallucinate" unseen geometry, it uses FMs (SAM3D) for robust initialization and tracking, then relies on actual multi-view observations for reconstruction. This "observe-don't-hallucinate" philosophy trades completeness for geometric fidelity on visible surfaces and enables dramatically faster inference. The use of SAM3D as a 3D segmentation prior rather than a shape prior represents a distinct and pragmatic way to inject FM knowledge into HOI reconstruction.
