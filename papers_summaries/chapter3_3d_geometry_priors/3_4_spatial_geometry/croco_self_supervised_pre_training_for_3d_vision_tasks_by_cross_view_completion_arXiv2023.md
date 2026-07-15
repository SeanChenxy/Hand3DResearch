# CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion

# Paper Summary

## Summary
CroCo introduces cross-view completion — a self-supervised pre-training task in which a masked image is reconstructed from its visible patches plus a second, unmasked view of the same scene — to learn representations that transfer well to monocular and binocular 3D vision tasks (depth, optical flow, pose) without any task-specific architecture or training data.

## 1. Problem and Setting
- **Task**: Self-supervised pre-training of a transformer for 3D vision / low-level geometric downstream tasks (depth, optical flow, relative pose).
- **Input/Output**: Pre-training — pairs of images of the same scene, one partially masked; the model reconstructs the masked patches. Downstream — image (or pair) → depth map, flow field, or relative pose.
- **Difficulties**:
  - MIM (e.g., MAE) learns high-level semantic features useful for classification / detection but transfers poorly to dense 3D / low-level geometric tasks.
  - Instance discrimination learns global image features, losing local pixel information needed for dense prediction.
  - Standard dense self-supervised methods are tailored to dense semantic tasks (segmentation, detection) rather than 3D / geometric tasks.
  - Existing self-supervised pre-training requires either category-specific object priors or RGB-D data, neither of which scale to general 3D scenes.
  - Classical binocular pipelines (e.g., optical flow) rely on hand-engineered cost volumes and iterative refinement.

## 2. Core Method
**Pipeline**: Two images of the same scene (Habitat synthetic) → randomly mask one image (typically 90% of patches) → ViT encoder on the visible patches of the masked image and on the full reference image → transformer decoder with cross-attention → reconstruct masked patches.

**Key components**:
1. **Cross-view completion pretext**: Given image x¹ and reference image x² of the same scene, randomly mask a high fraction (e.g., 90%) of patches in x¹. Encode visible patches of x¹ and all patches of x² in a Siamese ViT, then use a cross-attention decoder to reconstruct the masked patches of x¹ conditioned on the reference encoding.
2. **Why cross-view completion works**: Single-view MIM is ambiguous for masked regions and the model can only fall back on semantic priors. Cross-view completion resolves much of the ambiguity with the reference image — but only if the model understands the scene geometry and the spatial relationship between the two views. This forces 3D-aware pre-training.
3. **Architecture**: Siamese ViT encoder E_θ (shared weights) and a transformer decoder D_φ with alternating self-attention and cross-attention (CrossBlock), plus a "CatBlock" variant that concatenates tokens before self-attention. A simple pixel reconstruction loss over masked patches, similar to MAE.
4. **Training data**: Pairs of images of the same scene from synthetic indoor renderings in the Habitat simulator.
5. **Downstream transfer**:
   - **Monocular tasks** (e.g., NYUv2 depth): drop the decoder, use the ViT encoder as a backbone.
   - **Binocular tasks** (e.g., optical flow, relative pose): keep the full CroCo architecture, attach a small regression head (e.g., 2D flow per pixel, 6-DoF pose).
6. **Masking ratio**: Empirically, very high ratios (90%) give the best pre-training.

**Essential difference from existing methods**:
- MIM variants (MAE, BEiT, iBOT) focus on semantic transfer; CroCo's cross-view design forces geometric understanding.
- Dense pixel-level contrastive methods target dense semantic tasks, not 3D geometry.
- Hand-engineered optical flow / stereo pipelines are task-specific; CroCo pre-trains a generic architecture.

## 3. Knowledge, Supervision, and Assumptions
- **Pre-training data**: Synthetic indoor scene renderings from the Habitat simulator (pairs of same-scene views).
- **Supervision**: Self-supervised pixel reconstruction loss over masked patches — no labels, no depth, no pose.
- **Foundation-model usage**: Builds on ViT, MAE-style reconstruction objectives, and Siamese encoders; introduces cross-view completion as a new pretext task.
- **Assumptions**:
  - High masking ratios (90%) force the model to exploit the reference view.
  - Synthetic Habitat renderings provide sufficient variety of indoor 3D scenes for transferable pre-training.
  - Pixel reconstruction is a sufficient objective for learning 3D-aware features.
- **Learned vs. provided**: Encoder and decoder are learned from scratch on Habitat pairs. Downstream labels (NYUv2 depth, Taskonomy, etc.) come from standard benchmark training sets.

## 4. Experiments and Findings
- **Pre-training data**: Habitat-simulated indoor scene pairs.
- **Downstream datasets**: NYUv2 (monocular depth), Taskonomy (a diverse set of dense 2D and 3D regression tasks), ImageNet (classification, for comparison).
- **Metrics**: Standard depth metrics (e.g., RMSE on NYUv2), Taskonomy task-specific scores, ImageNet top-1 accuracy.
- **Key results stated**:
  - CroCo significantly improves over MIM pre-training (MAE, etc.) on monocular 3D vision tasks (depth, etc.).
  - For binocular tasks (optical flow, relative pose), CroCo achieves competitive results "without bells and whistles" — i.e., a generic architecture, no cost volume, no iterative refinement, no task-specific design.
  - High masking ratios (90%) yield the best downstream performance.
  - CroCo is *less* competitive than MIM/contrastive methods on high-level ImageNet classification — expected, because Habitat pre-training is geometric, not semantic.
  - CroCo is more compute-efficient on geometric tasks than MIM trained on the same data.
- **Ablations** (referenced): effect of masking ratio; effect of pre-training dataset (Habitat vs. ImageNet); effect of architecture (CrossBlock vs. CatBlock).

## 5. Strengths and Limitations
### Strengths
- **Strong transfer to geometric 3D tasks**: Outperforms MIM and contrastive methods on depth, optical flow, and pose.
- **Generic architecture**: No cost volume, no warping, no iterative refinement — same pre-trained model works for both monocular and binocular tasks.
- **Self-supervised**: No 3D labels needed for pre-training.
- **Simple objective**: Pixel reconstruction loss, similar to MAE.
- **Open source**: Code and pre-trained checkpoints released by Naver Labs.

### Limitations
- **Synthetic pre-training data only**: Habitat indoor renderings — limited diversity, indoor scenes only, and synthetic-to-real domain gap on outdoor / object-centric settings.
- **Reduced semantic transfer**: CroCo underperforms MIM/contrastive methods on ImageNet classification because the pretext is geometric, not semantic.
- **Requires image pairs at pre-training time**: Cannot use single images; pair collection for new domains is non-trivial.
- **High masking ratio (90%) is unusual**: Less efficient at low mask ratios; the design is calibrated to cross-view completion.
- **Downstream fine-tuning still required**: CroCo is a pre-trained model, not a frozen feature extractor; downstream training on labeled data is needed.
- **Quadratic attention**: Pre-training on high-resolution pairs is memory-heavy.

## 6. Takeaway
CroCo shows that **the way a model is asked to fill in missing pixels defines the kind of geometry it learns**: by reconstructing a masked image conditioned on a second view of the same scene, a vanilla ViT pre-trained with cross-view completion acquires 3D-aware features that transfer well to monocular depth, optical flow, and relative pose — without any hand-engineered geometric module or 3D label. This insight — that the pretext task itself can encode 3D-awareness — was the foundation for downstream methods like DUSt3R, MASt3R, CroCo v2, and VGGT. For HOI research, CroCo-style pre-training is a powerful recipe for hand-object geometry encoders: by forcing the network to predict masked hand- or object-region pixels from a second view, the resulting features learn depth, correspondence, and pose priors useful for downstream HOI reconstruction.