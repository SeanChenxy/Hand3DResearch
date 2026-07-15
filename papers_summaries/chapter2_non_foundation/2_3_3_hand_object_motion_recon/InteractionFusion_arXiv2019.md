# InteractionFusion: Real-Time Reconstruction of Hand Poses and Deformable Objects in Hand-Object Interactions

## Summary
First real-time system for jointly tracking hand poses and reconstructing deforming objects from RGB-D input, using a unified probabilistic optimization framework with hand-object collision constraints.

## 1. Problem and Setting
- Real-time joint reconstruction of 3D hand poses and deformable/non-rigid object shapes during manipulation.
- Input: RGB-D video stream; output: MANO hand pose per frame + deforming object mesh per frame.
- RGB-D input (not RGB-only); real-time performance; handles deformable objects (e.g., clay, cloth).

## 2. Core Method
- A unified energy minimization framework that jointly optimizes:
  1. Hand tracking: MANO parameters fitted to RGB-D observations (depth, 2D keypoints).
  2. Object reconstruction: non-rigid deformation field applied to an initial object scan, refined over time as new depth observations become available.
  3. Hand-object collision: explicit penalty preventing hand mesh vertices from penetrating the object surface.
- The object model starts from an initial depth scan (or a simple geometric primitive) and is progressively refined via a volumetric TSDF fusion approach adapted for non-rigid deformation.
- Hand-object interactions are modeled via a penalty-based collision term that pushes intersecting hand vertices out of the object.

## 3. Knowledge, Supervision, and Assumptions
- Training data: none (online optimization, no learning).
- Supervision: depth maps, RGB images (for keypoint detection), collision constraints.
- Uses MANO for hand.
- Assumes RGB-D sensor available; initial object scan or primitive available; object deformation is smooth and relatively slow.

## 4. Experiments and Findings
- Datasets: custom captures with deformable objects (clay, foam).
- Metrics: hand tracking error, object reconstruction error (against ground-truth scans), runtime.
- Real-time performance (~30 fps) with reasonable hand tracking and object deformation reconstruction quality. Collision constraints improve both hand and object accuracy.

## 5. Strengths and Limitations
### Strengths
- Real-time performance — one of the first real-time joint hand-object systems.
- Handles deformable objects (unique at the time).
- Probabilistic framework with principled collision modeling.

### Limitations
- Requires RGB-D sensor (depth camera), not RGB-only.
- Needs an initial object scan/model.
- Object deformation model is simple (cannot handle topological changes).
- Limited to scenes where the depth sensor has clear view of both hand and object.

## 6. Takeaway
InteractionFusion showed that real-time joint hand-object tracking is achievable with RGB-D input, and that physical constraints (collision) provide crucial regularization. While superseded by learning-based RGB-only methods for pose estimation, its probabilistic collision modeling and real-time fusion approach influenced downstream robotics and AR applications.
