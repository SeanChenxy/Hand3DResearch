# THOR-Net: End-to-end Graformer-based Realistic Two Hands and Object Reconstruction with Self-supervision

## Summary
An end-to-end framework combining Graph and Transformer (Graformer) architectures to reconstruct two interacting hands and a manipulated object with physically plausible alignment, using self-supervised physical constraints to reduce 3D annotation requirements.

## 1. Problem and Setting
- Joint 3D reconstruction of two hands and an object during bimanual interaction from a single RGB image.
- Input: single RGB image of two hands manipulating an object. Output: MANO meshes for both hands and an object mesh/point cloud.
- Static image setting; both hands and the object reconstructed jointly.
- Explicitly targets realistic, physically plausible spatial alignment between the three entities.

## 2. Core Method
- Graformer architecture: combines Graph Convolutional Networks (GCNs, which preserve mesh topology) with Transformer layers (which model long-range dependencies between hands and object).
- The GCN component processes each entity (left hand, right hand, object) independently while preserving their surface topology, producing per-entity features.
- The Transformer component enables cross-entity attention, allowing hand vertices to attend to object vertices and the other hand, modeling the spatial and functional relationships between them.
- Self-supervised physical constraints: penetration loss (penalizing hand-object and hand-hand interpenetration) and contact loss (encouraging proximity at contact regions) act as training signals without requiring ground-truth contact annotations.
- The model is trained end-to-end with a mix of supervised losses (when 3D annotations are available) and the self-supervised physical losses.

## 3. Knowledge, Supervision, and Assumptions
- Training data: datasets with 3D hand and object annotations where available (e.g., H2O, InterHand2.6M), supplemented by self-supervised signals.
- Supervision signals: 3D hand and object mesh vertices (supervised where available), penetration loss, contact loss (self-supervised).
- Uses MANO for both hands.
- Object representation is flexible (mesh or point cloud), but training benefits from known object meshes.
- The self-supervised losses assume that hands and objects should not interpenetrate and should be in proximity during interaction — physically reasonable priors.

## 4. Experiments and Findings
- Evaluated on H2O and InterHand2.6M datasets.
- Metrics: MPJPE (hand joint error), mesh vertex error, penetration depth, contact distance.
- THOR-Net achieves competitive accuracy with fully supervised methods while using less 3D annotation.
- The self-supervised physical losses significantly reduce penetration artifacts and improve hand-object alignment compared to purely supervised baselines.
- Ablation: both the GCN topology preservation and the Transformer cross-entity attention contribute substantially to accuracy.

## 5. Strengths and Limitations
### Strengths
- Graformer design elegantly combines topology-aware GCN processing with cross-entity Transformer attention.
- Self-supervised physical constraints reduce reliance on expensive 3D annotations.
- Explicitly addresses the bimanual + object setting, which is more challenging and realistic than single-hand scenarios.

### Limitations
- Architecture complexity is higher than single-hand methods; may be harder to train.
- Self-supervised losses provide useful regularization but cannot fully replace 3D supervision for fine-grained accuracy.
- Object reconstruction quality depends on the object representation capability and training data diversity.
- Does not handle category-agnostic objects (assumes some known object priors).

## 6. Takeaway
THOR-Net advanced bimanual hand-object reconstruction by integrating GCNs and Transformers into a unified Graformer architecture, showing that topology-aware processing plus cross-entity attention is a powerful combination for multi-entity interaction modeling. The self-supervised physical constraints demonstrated that simple geometric priors (no penetration, proximity at contact) can serve as effective training signals, a theme that influenced subsequent self-supervised and physics-aware approaches.
