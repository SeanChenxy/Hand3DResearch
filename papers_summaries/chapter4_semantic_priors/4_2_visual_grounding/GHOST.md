# GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting (Cross-reference)

## Summary
This entry is a cross-reference to the detailed summary in Chapter 3 (3D Geometry Priors, section 3.3 Shape Retrieval). GHOST is a fast, category-agnostic framework for reconstructing dynamic hand-object interactions from monocular RGB videos using 2D Gaussian Splatting, with three key innovations: geometric-prior retrieval and consistency loss, grasp-aware alignment, and hand-aware background loss.

## 1. Problem and Setting
- Fast, category-agnostic 3D reconstruction of dynamic hand-object interactions from monocular RGB videos.
- Input: monocular RGB video of hand-object interaction.
- Output: 3D hand mesh, 3D object as Gaussians, 6D object pose trajectory.
- Visual grounding prior: the geometric-prior retrieval grounds the 3D object shape in a database of object embeddings, providing a visual-grounded shape prior.

## 2. Core Method
- 2D Gaussian Splatting for both hands and objects.
- Three innovations: geometric-prior retrieval, grasp-aware alignment, hand-aware background loss.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: foundation model embeddings for retrieval; 3D Gaussian Splatting for efficient representation.
- Domain knowledge: hand model (MANO); physical plausibility constraints.
- Training data: no HOI-specific training; FM embeddings and 3D object database.

## 4. Experiments and Findings
- Datasets: ARCTIC, HO3D, in-the-wild datasets.
- State-of-the-art accuracy with order of magnitude faster speed.

## 5. Strengths and Limitations
### Strengths
- Fast reconstruction suitable for interactive applications.
- Category-agnostic.
- Combines retrieval and optimization strengths.

### Limitations
- Depends on database coverage.
- May not capture fine details as well as SDF methods.

## 6. Takeaway
GHOST demonstrates a practical synthesis of retrieval and optimization for HOI. In the context of visual grounding (chapter 4), the FM-based retrieval provides a visual-grounded shape prior. See chapter 3 section 3.3 for the full technical details.
