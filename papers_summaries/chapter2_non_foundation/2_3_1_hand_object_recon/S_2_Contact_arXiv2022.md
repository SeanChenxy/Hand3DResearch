# S²Contact: Graph-Based Network for 3D Hand-Object Contact Estimation with Semi-supervised Learning

## Summary
A graph-based semi-supervised framework for estimating dense 3D hand-object contact (which hand vertices touch which object regions) that leverages both fully annotated and unlabeled data through consistency regularization.

## 1. Problem and Setting
- Dense 3D contact estimation between a hand mesh and an object mesh, i.e., predicting per-vertex contact labels on both the hand and object surfaces.
- Input: 3D hand mesh (MANO) and object mesh (CAD/template) with known relative pose. Output: per-vertex contact probability for both hand and object vertices.
- Static 3D setting; given the meshes and their relative pose, predict where contact occurs.
- Both hand and object; focus is on the contact interface between them.

## 2. Core Method
- Graph-based network (GCN) operating on the hand mesh and object mesh as separate graphs with inter-graph message passing.
- Hand mesh graph: nodes are MANO vertices with kinematic edges. Object mesh graph: nodes are object vertices with surface edges.
- Cross-graph attention/communication: hand vertices attend to nearby object vertices (and vice versa) based on spatial proximity, enabling the network to reason about which hand-object vertex pairs are in contact.
- Semi-supervised training: a teacher-student consistency framework. A teacher model (EMA of the student) generates pseudo-labels on unlabeled hand-object pairs. The student is trained on labeled data (supervised loss) plus a consistency loss between its predictions and the teacher's on unlabeled data under different augmentations.
- Graph augmentation includes random rotations, scaling, and vertex dropout to encourage robust contact reasoning.

## 3. Knowledge, Supervision, and Assumptions
- Labeled training data: hand-object pairs with dense ground-truth contact annotations (e.g., ContactPose, HO-3D with contact labels).
- Unlabeled data: hand-object mesh pairs without contact labels (only mesh geometry and relative pose).
- Supervision signals: binary cross-entropy for vertex-level contact classification on labeled data; consistency loss (e.g., MSE) between student and teacher predictions on unlabeled data.
- Requires both MANO for the hand and a known object mesh/template.
- Assumption: contact patterns are largely determined by 3D geometry and relative pose; the graph network can learn this mapping.

## 4. Experiments and Findings
- Evaluated on ContactPose, HO-3D, and ObMan datasets.
- Metrics: contact classification accuracy, F1 score, AUC, precision/recall of predicted contact vertices.
- S²Contact achieves state-of-the-art contact estimation accuracy, with significant gains over fully supervised baselines when only limited labeled data is available.
- Semi-supervised learning provides the largest gains when labeled data is scarce (< 20% labeled), demonstrating effective use of unlabeled hand-object pose data.
- Ablation: cross-graph attention is critical for accuracy; removing it causes a large drop in contact prediction quality.

## 5. Strengths and Limitations
### Strengths
- Addresses the practical problem of scarce contact annotations through semi-supervised learning.
- Graph-based architecture naturally captures the geometric structure of hand-object contact.
- Cross-graph attention mechanism effectively models the interplay between hand and object surfaces.

### Limitations
- Requires known hand and object meshes (MANO + CAD template); not applicable to unknown objects.
- Assumes the relative hand-object pose is given; errors in pose estimation would cascade to contact prediction.
- The graph construction depends on spatial proximity heuristics which may miss long-range contact dependencies.
- Semi-supervised gains depend on the quality and diversity of unlabeled data.

## 6. Takeaway
S²Contact showed that dense hand-object contact estimation benefits significantly from semi-supervised learning, reducing the dependency on expensive contact annotations. Its graph-based cross-attention design provides an effective architecture for reasoning about the geometry of contact, and the semi-supervised paradigm has influenced subsequent works that aim to learn physical interaction priors from limited labeled data.
