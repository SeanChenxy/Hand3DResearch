# HOISDF: Constraining 3D Hand-Object Pose Estimation with Global Signed Distance Fields

## Summary
Jointly estimates hand and object 3D poses from a single image by constraining predictions via a global SDF that enforces physical non-penetration between hand and object meshes.

## 1. Problem and Setting
- Joint 3D pose estimation of hand and manipulated object from a single RGB image.
- Input: single RGB image; output: MANO hand parameters + 6D object pose (for known object templates) + contact constraints.
- Object template is known (not template-free). Focus is on pose estimation quality rather than shape reconstruction.

## 2. Core Method
- A feed-forward network predicts initial hand (MANO) and object (6D pose, assuming known CAD model) parameters from the RGB image.
- Global SDF constraint: during inference-time optimization, the predicted hand and object meshes are refined by minimizing an SDF-based interpenetration loss. For each vertex on the hand mesh, the SDF value w.r.t. the object mesh (and vice versa) is computed; positive values inside the object are penalized.
- The global SDF approach is more physically meaningful than sparse contact-point constraints used in prior work, as it prevents penetration across the entire mesh surface.
- Can be applied as a refinement step on top of any base pose estimator.

## 3. Knowledge, Supervision, and Assumptions
- Training data: HO3D, ObMan, DexYCB for the base pose estimator.
- Supervision: 3D hand keypoints, 6D object pose, MANO parameters for training the base network; SDF constraint is self-supervised at test time.
- Uses MANO for hand.
- CAD models required at test time (known object templates).
- Assumes object 3D model is known in advance; object is rigid.

## 4. Experiments and Findings
- Datasets: HO3D, DexYCB.
- Metrics: MPJPE, PA-MPJPE (hand); ADD, ADD-S (object pose); penetration depth.
- The SDF constraint significantly reduces hand-object interpenetration while maintaining or improving pose accuracy. Outperforms contact-point-based methods on both hand and object metrics.

## 5. Strengths and Limitations
### Strengths
- Dense, physically-grounded constraint that prevents penetration across the entire mesh.
- Plug-and-play — can be applied on top of any hand-object pose estimator.
- Self-supervised refinement (no extra training data needed).

### Limitations
- Requires known object CAD models (not template-free).
- Inference-time optimization adds latency (~seconds per frame).
- Cannot correct grossly incorrect initial pose estimates.
- Only prevents penetration; does not enforce proper contact.

## 6. Takeaway
HOISDF showed that global SDF-based constraints provide a principled and effective way to enforce physical plausibility in hand-object pose estimation. The idea of using implicit shape representations as a collision constraint has broader applicability beyond just hand-object interaction.
