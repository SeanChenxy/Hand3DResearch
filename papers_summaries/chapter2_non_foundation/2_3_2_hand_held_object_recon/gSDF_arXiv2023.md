# gSDF: Geometry-Driven Signed Distance Functions for 3D Hand-Object Reconstruction

## Summary
Improves hand-object SDF reconstruction by explicitly using geometric priors (surface normals and hand-part segmentation) to guide the SDF learning, achieving state-of-the-art object reconstruction quality.

## 1. Problem and Setting
- Joint reconstruction of hand and manipulated object 3D shape from a single RGB image.
- Input: single RGB image; output: MANO hand mesh + object SDF.
- Template-free object reconstruction. Builds on AlignSDF but adds richer geometric supervision signals.

## 2. Core Method
- Extends the AlignSDF framework with two key geometric priors:
  - Surface normal prediction: an additional decoder head predicts surface normals for each 3D query point. The normals provide local geometric orientation cues that help the SDF decoder learn sharper surfaces.
  - Hand-part segmentation guidance: uses predicted 2D hand-part segmentation to inform which finger regions are in contact with the object, improving spatial reasoning about where the object should be relative to each finger.
- The normal prediction is jointly trained and used as an auxiliary loss, ensuring the SDF field respects local surface orientation.
- Maintains the pose-aligned coordinate frame from AlignSDF.

## 3. Knowledge, Supervision, and Assumptions
- Training data: ObMan (synthetic) + HO3D (real) + ContactPose (real).
- Supervision: 3D SDF values, surface normals (computed from ground-truth meshes), 2D hand-part segmentation, MANO parameters.
- Uses MANO for hand representation.
- Assumes object is rigid and single-hand grasping; surface normal supervision is available from training meshes.

## 4. Experiments and Findings
- Datasets: HO3D, ObMan, ContactPose.
- Metrics: Chamfer Distance, F-score, Normal Consistency for object; MPJPE for hand.
- Outperforms AlSDF and DDF-HO on object reconstruction metrics. Normal supervision leads to visibly sharper object surfaces. Hand-part segmentation improves contact-region accuracy.

## 5. Strengths and Limitations
### Strengths
- Geometry-driven losses produce sharper, more accurate object surfaces than purely occupancy/SDF-based approaches.
- Normal prediction provides dense 3D supervision without requiring explicit 3D mesh labels.

### Limitations
- The normal prediction head adds computational overhead.
- Performance still degrades under heavy occlusion from the hand.
- Relies on having 3D mesh training data to compute surface normal ground truth.
- Single-hand, rigid-object constraint.

## 6. Takeaway
gSDF demonstrated that enriching implicit representations with mid-level geometric cues (normals, part segmentation) substantially improves reconstruction fidelity. This "geometry-driven" philosophy — using predictable geometric properties as auxiliary supervision — has been broadly adopted in later implicit 3D reconstruction works.
