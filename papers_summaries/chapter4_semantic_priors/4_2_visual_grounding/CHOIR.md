# CHOIR: Contact-aware 4D Hand-Object Interaction Reconstruction (Cross-reference)

## Summary
This entry is a cross-reference to the detailed summary in Chapter 3 (3D Geometry Priors, section 3.4 Spatial Geometry). CHOIR presents a versatile and differentiable hand-object interaction representation that supports multiple downstream tasks (reconstruction, refinement, synthesis) by encoding the spatial and contact relationships between hand and object in an end-to-end learnable manner.

## 1. Problem and Setting
- Creating a general-purpose, differentiable representation of hand-object interaction that is useful across reconstruction, pose refinement, and grasp synthesis tasks.
- Input: hand mesh (MANO) and object mesh (or point cloud) with known relative pose.
- Output: a differentiable encoding of their spatial interaction (distance fields, contact maps, penetration maps).
- Visual grounding prior: the differentiable HOI representation serves as a visual-grounded signal (through contact, penetration, and distance) for downstream tasks.

## 2. Core Method
- A differentiable hand-object interaction (HOI) representation that encodes:
  - Signed distance field (SDF) between hand vertices and object surface (and vice versa).
  - Contact probability map (per-vertex likelihood of contact).
  - Penetration map indicating where hand vertices penetrate the object.
- All components are computed differentiably from hand and object meshes.
- Usable as: (a) a differentiable loss for refining hand-object poses; (b) a conditioning signal for grasp synthesis; (c) an evaluation metric.

## 3. Knowledge, Supervision, and Assumptions
- The representation itself is hand-crafted (not learned from data).
- For training downstream models that use this representation, standard HOI datasets apply.
- Assumes access to hand mesh (typically MANO) and object mesh at test time.
- No training is needed for the representation itself.

## 4. Experiments and Findings
- Multi-task evaluation: pose refinement, grasp synthesis, reconstruction evaluation.
- The HOI representation as a refinement loss consistently improves hand-object pose accuracy.
- The contact probability component is effective for grasp synthesis.
- Provides physically meaningful supervision without requiring contact annotations.

## 5. Strengths and Limitations
### Strengths
- Versatile representation applicable to multiple tasks.
- Fully differentiable for end-to-end integration.
- Provides physically meaningful signals (contact, penetration).
- Task- and architecture-agnostic.

### Limitations
- Not a learned representation; may not capture functional interaction semantics.
- Requires known hand and object meshes at inference.
- Computational cost scales with mesh resolution.
- Captures geometric but not dynamic interaction.

## 6. Takeaway
CHOIR highlights the value of a well-designed, differentiable hand-object interaction representation that serves as a bridge between geometric reconstruction and physical reasoning. In the context of visual grounding (chapter 4), it provides a visual-grounded, differentiable contact signal that downstream tasks can leverage. See chapter 3 section 3.4 for the full technical details.
