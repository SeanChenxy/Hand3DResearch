# AlignSDF: Pose-Aligned Signed Distance Fields for Hand-Object Reconstruction

## Summary
Jointly reconstructs hand and object as pose-aligned SDFs from monocular RGB images, bridging the gap between parametric mesh and implicit surface representations through a unified coordinate-aligned framework.

## 1. Problem and Setting
- Joint 3D reconstruction of both the hand and the manipulated object from a single RGB image.
- Input: single RGB image; output: 3D hand mesh (MANO) + 3D object surface (SDF).
- The hand is parametric (MANO), the object is represented as a signed distance field (SDF) in a pose-aligned coordinate frame.

## 2. Core Method
- A shared image encoder feeds two task-specific heads: a hand head predicting MANO parameters, and an object head predicting SDF values.
- Pose-aligned SDF: the object SDF is predicted in a canonical coordinate frame aligned to the hand pose (specifically to each finger bone). This decomposes the object shape into per-finger-aligned local patches that are easier to learn.
- Each query point is transformed to multiple local coordinate systems (one per finger bone) before querying the SDF decoder, enabling the model to reason about object shape relative to each finger's configuration.
- Final SDF is aggregated from per-bone predictions via a learned fusion module.

## 3. Knowledge, Supervision, and Assumptions
- Training data: synthetic data from ObMan + real images from HO3D.
- Supervision: 3D object SDF values (from synthetic data), 2D/3D hand keypoints, MANO pose parameters.
- Uses MANO for hand.
- Assumes the object is rigid; single-hand interaction; the hand pose is reasonably predictable from the image.

## 4. Experiments and Findings
- Datasets: HO3D, ObMan.
- Metrics: Chamfer Distance, F-score (object); MPJPE, PA-MPJPE (hand).
- Pose-aligned SDF significantly outperforms global SDF representations for object reconstruction, particularly for articulated grasping poses where global coordinate reasoning is ambiguous.

## 5. Strengths and Limitations
### Strengths
- Pose-aligned representation elegantly decomposes the complex 3D reasoning task into simpler per-bone sub-problems.
- Unified framework handles both hand and object in a single forward pass.
- Better generalization to novel grasping poses compared to global SDF methods.

### Limitations
- Still requires synthetic data with 3D supervision for training.
- Object SDF quality is bounded by hand pose accuracy.
- Only single-hand, single-rigid-object settings.
- Fusion of per-bone predictions can produce artifacts at bone boundaries.

## 6. Takeaway
AlignSDF introduced the idea that aligning the object representation to the articulated hand structure dramatically simplifies learning. This "pose-aligned" or "articulation-aware" coordinate frame concept became influential in subsequent hand-object reconstruction works, particularly gSDF and geometry-driven methods.
