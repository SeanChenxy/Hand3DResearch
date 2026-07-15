# CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow

# Paper Summary

## Summary
CroCo v2 scales up cross-view completion pre-training for binocular 3D vision tasks by collecting millions of real-world image pairs, replacing absolute positional embeddings with Rotary embeddings (RoPE), and enlarging both the ViT encoder and the decoder — achieving state-of-the-art stereo matching and optical flow with a generic, task-agnostic architecture and no classical task-specific modules.

## 1. Problem and Setting
- **Task**: Self-supervised pre-training of a transformer for dense geometric vision tasks (stereo matching, optical flow) — then fine-tuning on each downstream task.
- **Input/Output**: Pre-training — pairs of real images of the same scene with one image partially masked; downstream — two images → disparity map (stereo) or 2D optical flow field.
- **Difficulties**:
  - MIM and instance discrimination have advanced high-level tasks (classification, detection) but not dense geometric tasks (stereo, flow).
  - Prior CroCo (Weinzaepfel et al. 2022) trained only on synthetic Habitat pairs and used standard absolute positional embeddings — both limit real-world transfer and resolution generalization.
  - Classical stereo/flow pipelines rely on hand-engineered modules (cost volumes, image warping, iterative refinement, multi-scale feature pyramids) that are difficult to scale and specialize to a single task.
  - Vanilla ViTs with cosine positional embeddings do not generalize to new image resolutions and are sensitive to cropping.

## 2. Core Method
**Pipeline**: Pairs of real images (ARKitScenes, MegaDepth, 3DStreetView, IndoorVL) with controlled overlap → CroCo v2 pre-training (cross-view completion with ViT encoder + cross-attention decoder, RoPE) → fine-tune on stereo / optical flow with a DPT head → output disparity or flow field.

**Key components**:
1. **Large-scale real-world pre-training data**: A new automated pipeline that gathers millions of image pairs from ARKitScenes, MegaDepth, 3DStreetView, and IndoorVL, and filters them by visual overlap so that pairs are neither trivially redundant nor non-overlapping. Replaces the prior synthetic-only Habitat training set.
2. **Rotary Positional Embeddings (RoPE)**: Replaces the absolute cosine positional embedding of standard ViTs. RoPE encodes relative position between token pairs, so the model generalizes to new image resolutions and is robust to cropping — critical for dense prediction downstream.
3. **Larger ViT encoder and decoder**: Encoder and decoder are both scaled up; the larger decoder (responsible for combining the two views) is enabled by the larger real-image pre-training corpus.
4. **Generic, task-agnostic fine-tuning head**: A single DPT (Dense Prediction Transformer) head consumes the frozen-then-fine-tuned backbone and produces disparity (stereo) or 2D optical flow directly — no cost volume, no warping, no iterative refinement, no multi-scale pyramid.
5. **Models released**: CroCo-Stereo and CroCo-Flow for downstream use.

**Essential difference from existing methods**:
- Pre-trains with cross-view completion on real-world pairs at millions of examples, not synthetic-only.
- Uses RoPE for resolution-agnostic dense prediction.
- Demonstrates that a generic architecture can match hand-engineered SOTA on stereo and flow — a step toward universal vision models.

## 3. Knowledge, Supervision, and Assumptions
- **Pre-training data**: Millions of real image pairs from ARKitScenes, MegaDepth, 3DStreetView, IndoorVL; filtered by 3D meshes, LiDAR, or SfM-based overlap control.
- **Supervision**: Self-supervised cross-view completion — visible patches of one image + a reference image reconstruct the masked patches; pixel reconstruction loss.
- **Foundation-model usage**: Inherits CroCo (Weinzaepfel et al. 2022) as the pre-training framework; uses CroCo-Stereo and CroCo-Flow as the downstream baselines.
- **Assumptions**:
  - Real-world image pairs with controlled overlap are sufficient to teach generic 3D scene layout.
  - RoPE is sufficient to make vanilla ViTs perform well on dense geometric tasks at variable resolution.
  - A single ViT encoder + plain transformer decoder + DPT head can replace cost volumes and other task-specific structures.
- **Learned vs. provided**: All representations are learned from real pairs; downstream labels (disparity, flow) come from standard benchmark training sets (KITTI 2015, ETH3D, Spring, MPI-Sintel).

## 4. Experiments and Findings
- **Datasets (downstream)**: KITTI 2015, ETH3D, Spring, MPI-Sintel for stereo matching and optical flow.
- **Datasets (pre-training)**: ARKitScenes, MegaDepth, 3DStreetView, IndoorVL — millions of curated real-world pairs.
- **Metrics**: Standard stereo benchmarks (D1 / outlier rate for KITTI, bad-pixel rates for ETH3D, etc.) and optical-flow endpoint error / % outliers for Sintel/KITTI (referenced; specific numbers not in extracted excerpt).
- **Key results stated**:
  - CroCo v2 reaches state-of-the-art on stereo matching and optical flow benchmarks without using classical task-specific techniques (no cost volume, no image warping, no iterative estimation, no multi-scale reasoning).
  - This is the first demonstration that a generic architecture can match hand-engineered SOTA on these tasks.
  - RoPE substantially improves dense performance over cosine positional embeddings, especially at non-square resolutions and with cropping.
  - Scaling the encoder and decoder together improves results once real-world pairs are available.
- **Ablations (referenced in paper)**: real-vs-synthetic pre-training; RoPE vs cosine; encoder/decoder scale.

## 5. Strengths and Limitations
### Strengths
- **State-of-the-art with a generic architecture**: Reaches the top of stereo and optical flow benchmarks using only a plain ViT + cross-attention decoder + DPT head.
- **Real-world pre-training data**: The new pair-collection pipeline removes the synthetic-to-real domain gap that limited prior CroCo.
- **RoPE for resolution-agnostic dense prediction**: Enables training and inference at non-square resolutions and with cropping.
- **Step toward universal vision models**: A single pre-training recipe transfers to both stereo and flow without task-specific design.
- **Released code and checkpoints**: CroCo-Stereo and CroCo-Flow on GitHub.

### Limitations
- **Pre-training data curation is non-trivial**: Pair overlap must be carefully controlled (high overlap → trivial; low overlap → MIM).
- **No cost volume, no warping**: Although simpler, this removes a strong inductive bias — efficiency and data needs may be larger than hand-engineered pipelines.
- **Downstream fine-tuning still required**: The pre-trained model is not a frozen feature extractor; downstream fine-tuning on labeled stereo/flow data is needed.
- **Resolution generalization trade-offs**: RoPE improves but does not eliminate the resolution/cropping sensitivity of vanilla ViTs.
- **Evaluation focused on benchmarks**: Generalization to in-the-wild driving / robotics sequences is not characterized in the extracted excerpt.
- **Quadratic attention**: Dense ViT-based decoders are still memory-heavy on large stereo pairs.

## 6. Takeaway
CroCo v2 demonstrates that **a generic ViT, pre-trained with cross-view completion on millions of real image pairs and positionalized with RoPE, can replace hand-engineered cost-volume and warping pipelines for stereo matching and optical flow** — reaching state-of-the-art without any task-specific inductive bias. By attacking the two long-standing bottlenecks of CroCo (synthetic-only pre-training, absolute positional embeddings) with a real-world pair collection pipeline and RoPE, the work delivers a "universal" pre-training recipe for dense 3D vision. For HOI research, CroCo v2's cross-view completion pretext is a strong candidate for learning geometry-aware hand-object representations from large unlabeled video corpora, providing depth, correspondence, and pose priors that transfer to downstream HOI tasks.