# GEARS: Local Geometry-Aware Hand-Object Interaction Synthesis

## Summary
Introduces a local geometry-aware framework that explicitly reasons about fine-grained surface geometry (normals, curvatures) at contact regions to synthesize hand grasps that adapt to local object shape variations, achieving state-of-the-art contact precision.

## 1. Problem and Setting
- Task: synthesize a 3D hand grasp on a given object mesh, with emphasis on precise finger-to-surface alignment even for objects with complex local geometry (ridges, concavities, thin parts).
- Input: 3D object mesh/point cloud; Output: MANO hand mesh placed in contact with the object.
- Key challenge: prior methods often treat object geometry globally and fail to capture local geometric cues (sharp edges, varying curvature) that dictate where and how fingers should be placed.

## 2. Core Method
- Multi-scale local geometry encoder: computes local geometric features (point coordinates, normals, curvature, SDF values) at multiple scales around each object surface point using a graph neural network or local PointNet-style encoder.
- Contact heatmap prediction: the local geometry features feed into a transformer-based contact predictor that outputs per-point contact probability and per-contact-point hand-part association.
- Grasp optimization: MANO parameters are optimized via an energy function that aligns predicted hand-part vertices to the corresponding object contact points, with additional terms for local normal alignment and curvature-aware penetration avoidance.
- Key innovation: explicit local geometry conditioning — the model "sees" edges, corners, and curvature variations and learns that certain local features (e.g., concave regions for fingertips, flat regions for palm) are strongly predictive of contact.

## 3. Knowledge, Supervision, and Assumptions
- Training data: GRAB and ObMan datasets; possibly augmented with self-generated synthetic grasps on diverse object shapes.
- Supervision: contact labels, hand-part labels, and local geometry descriptors computed from ground-truth hand-object pairs.
- Domain knowledge: differential geometry of surfaces (normals, principal curvatures); MANO model.
- Assumption: rigid object, static single-hand grasp, full 3D object geometry available.

## 4. Experiments and Findings
- Datasets: GRAB, ObMan, and an in-the-wild test set with real object scans (e.g., from OakInk or DexYCB).
- Metrics: contact IoU, chamfer distance between predicted and GT hand vertices, penetration depth, simulation success rate.
- Main findings: GEARS consistently outperforms global-geometry methods, especially on objects with complex local structure (tools, non-convex shapes); ablation shows that both multi-scale encoding and curvature awareness independently improve results; generalization to unseen object categories is strong.

## 5. Strengths and Limitations
### Strengths
- Local geometry awareness is a significant and intuitive improvement — fingers go where the geometry "invites" them.
- Multi-scale encoding captures both fine details and broader surface context.

### Limitations
- Computationally heavier than single-scale methods due to per-point local geometry computation.
- Still requires iterative grasp optimization, limiting real-time capability.
- Does not handle dynamic grasps or task-specific functional intent.

## 6. Takeaway
GEARS demonstrates that local geometric detail matters critically for grasp synthesis: conditioning on per-point normals, curvatures, and multi-scale geometry produces noticeably better finger placement than global object encodings. This insight — that grasp models should be locally geometry-aware — has influenced subsequent work in contact prediction and dexterous manipulation.
