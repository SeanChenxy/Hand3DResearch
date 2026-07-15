# EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild (Cross-reference)

## Summary
This entry is a cross-reference to the detailed summary in Chapter 3 (3D Geometry Priors, section 3.2 Shape Completion). EasyHOI aims to reconstruct hand-object interactions from a single-view image by composing multiple off-the-shelf large foundation models (segmentation, depth estimation, inpainting, multi-view diffusion) into a training-free pipeline.

## 1. Problem and Setting
- 3D reconstruction of hand and object from a single RGB image in unconstrained conditions.
- Input: a single RGB image showing a hand interacting with an object.
- Output: 3D hand mesh, 3D object shape, relative hand-object 6D pose.
- Visual grounding prior: the VLM (SAM) provides visual-grounded segmentation of the hand and object regions.

## 2. Core Method
- A modular, zero-shot pipeline chaining frozen foundation models:
  1. Hand-object segmentation via a VLM (SAM + text prompts).
  2. Depth estimation via a monocular depth foundation model.
  3. Object inpainting to remove the hand and fill in the occluded object region.
  4. Multi-view 3D generation via a view-synthesis diffusion model, followed by 3D lifting.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models: SAM/GroundingDINO, Depth Anything, Stable Diffusion Inpainting, Zero-1-to-3.
- Domain knowledge: MANO; implicit assumptions about object convexity.
- Training data: zero training — entirely test-time inference.

## 4. Experiments and Findings
- Datasets: HO3D, DexYCB, in-the-wild images.
- Competitive or state-of-the-art without task-specific training.
- Each FM component contributes additively to final quality.

## 5. Strengths and Limitations
### Strengths
- Completely training-free.
- Modular design allows FM upgrades.
- Strong zero-shot generalization.
- Demonstrates that chaining FMs solves a complex 3D vision task.

### Limitations
- Multi-stage pipeline is prone to error accumulation.
- Computationally expensive.
- Bounded by view-synthesis FM's capabilities.
- Struggles with thin structures and transparent objects.

## 6. Takeaway
EasyHOI demonstrates that sophisticated HOI reconstruction can be achieved by composing off-the-shelf foundation models without any task-specific training. In the context of visual grounding (chapter 4), the segmentation FM provides the visual-grounded prior for distinguishing hand from object. See chapter 3 section 3.2 for the full technical details.
