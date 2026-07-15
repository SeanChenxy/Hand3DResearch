# Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba

## Summary
Introduces the Mamba state-space model architecture to 3D hand reconstruction, using a graph-guided bi-directional scanning mechanism that efficiently processes hand mesh topology while achieving linear computational complexity.

## 1. Problem and Setting
- 3D hand mesh reconstruction from a single RGB image.
- Input: single RGB image (hand crop). Output: MANO hand mesh (3D vertices, joints, pose, shape).
- Static image setting; hand-only reconstruction.
- The primary contribution is architectural: adapting the Mamba SSM (State Space Model) to the hand mesh domain.

## 2. Core Method
- Hamba replaces the conventional Transformer or GCN decoder with a Mamba-based architecture.
- Graph-guided Bi-Scanning Mamba: the MANO hand mesh vertices are arranged as a 1D sequence (via graph traversal), and a bi-directional Mamba block processes this sequence. The scanning order is guided by the mesh topology (e.g., kinematic tree traversal, breadth-first search from the wrist), ensuring that neighboring vertices in the mesh are close in the sequence.
- Bi-directional scanning means the Mamba processes the vertex sequence both forward and backward, capturing dependencies in both directions.
- The Mamba architecture provides linear computational complexity in sequence length (vs. quadratic for Transformers), making it more efficient for processing dense mesh vertices.
- The image features from a CNN or ViT backbone are projected to initial MANO parameters, which are then refined by the Mamba decoder through iterative vertex sequence processing.

## 3. Knowledge, Supervision, and Assumptions
- Trained on standard hand mesh datasets (FreiHAND, HO-3D, DexYCB) with MANO annotations.
- Supervision: 3D joint positions, MANO pose (θ) and shape (β), 3D vertex coordinates.
- Uses MANO as the hand model.
- The graph-guided scanning order is based on the known MANO mesh topology.
- Fully supervised training; the novelty is efficiency and representation quality.

## 4. Experiments and Findings
- Evaluated on FreiHAND, HO-3D, and DexYCB datasets.
- Metrics: PA-MPJPE, PA-MPVPE, F-scores, inference time, parameter count.
- Hamba achieves competitive or better accuracy than Transformer-based methods, with lower computational cost and fewer parameters.
- The graph-guided scanning order is critical: random scanning degrades performance significantly, confirming the importance of topology-aware sequence construction.
- Mamba-based processing is 2-3x faster than comparable Transformer decoders for the same mesh resolution.

## 5. Strengths and Limitations
### Strengths
- First application of Mamba/SSM architecture to hand mesh reconstruction, demonstrating the viability of linear-complexity alternatives to Transformers.
- Graph-guided scanning effectively transfers mesh topology priors to the 1D sequence domain.
- Better speed-accuracy trade-off than prior Transformer-based methods.

### Limitations
- Hand-only; no object reconstruction or hand-object interactions.
- Relies on MANO topology for scan order; may not generalize to other mesh structures without redesign.
- Mamba is a relatively new architecture; training stability and best practices are less established than for Transformers.
- Performance ceiling may still be below the best heavy Transformer models on very large datasets.

## 6. Takeaway
Hamba brought the Mamba state-space model to 3D hand reconstruction, demonstrating that the linear-complexity SSM architecture can match or exceed Transformer performance when paired with topology-aware sequence construction. This paper is part of a broader trend exploring post-Transformer architectures for 3D vision, and showed that graph-guided scanning is a key design choice for adapting sequence models to mesh-structured data.
