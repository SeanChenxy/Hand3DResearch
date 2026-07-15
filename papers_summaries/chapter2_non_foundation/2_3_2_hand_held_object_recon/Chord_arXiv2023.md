# Chord: Category-level Hand-held Object Reconstruction via Shape Deformation

## Summary
Reconstructs hand-held objects at the category level by deforming a category-level shape prior to match the observed image, jointly optimizing hand and object meshes via differentiable rendering.

## 1. Problem and Setting
- Category-level reconstruction of hand-held objects from RGB images or video, where the object category is known but the specific instance shape is unknown.
- Input: RGB image(s); output: MANO hand mesh + deformed category-level 3D object mesh.
- Category-level (not template-free) — a category-specific shape prior is available (e.g., "mug", "bottle"). Both hand and object reconstructed.

## 2. Core Method
- Learns a category-level shape space via a deformation network trained on ShapeNet object categories. Given an object category, a base mesh is deformed to fit the observed instance.
- Joint optimization framework: uses differentiable rendering (Neural Mesh Renderer) to optimize hand MANO parameters and object deformation parameters simultaneously, driven by photometric consistency and silhouette losses.
- For video input, enforces temporal smoothness on both hand pose and object shape/deformation.
- Can work with single images (weaker) or multi-view/video (stronger).

## 3. Knowledge, Supervision, and Assumptions
- Training data: ShapeNet for category-level shape priors; HO3D, ObMan for hand-object interaction data.
- Supervision: 2D supervision (silhouette, photometric) during test-time optimization; 3D shape supervision for the deformation network pretraining.
- Uses MANO for hand.
- Assumes object category is known a priori; object instances within a category share a common topology; object is rigid.

## 4. Experiments and Findings
- Datasets: HO3D (real), ObMan (synthetic), custom in-the-wild captures.
- Metrics: Chamfer Distance, IoU for object; MPJPE for hand.
- Category priors significantly improve reconstruction quality over template-free methods, especially under occlusion. The deformation-based approach produces watertight, plausible meshes even for heavily occluded regions.

## 5. Strengths and Limitations
### Strengths
- Category prior enables reconstruction of fully occluded object parts by "hallucinating" plausible geometry.
- Produces complete, watertight meshes (unlike occupancy/SDF approaches that may have holes).

### Limitations
- Requires known object category — cannot handle completely unknown objects.
- Deformation space is limited to the training categories (ShapeNet coverage).
- Test-time optimization is slow compared to feed-forward methods.
- Assumes rigid objects with fixed category topology.

## 6. Takeaway
CHORD showed that category-level priors offer a powerful middle ground between template-based (requires exact CAD model) and template-free (struggles under occlusion) approaches. The deformation-based paradigm enables physically plausible completions of occluded regions, making it practical for real-world applications where object categories are often known.
