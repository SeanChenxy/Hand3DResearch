# MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details

# Paper Summary

## Summary
MoGe-2 extends the monocular geometry estimation approach MoGe — which predicts affine-invariant 3D point maps from a single image — with a metric scale prediction head and a unified real-data refinement pipeline that injects sharp synthetic-style detail into the geometry predictions, simultaneously achieving state-of-the-art relative geometry accuracy, metric scale precision, and fine-grained detail recovery.

## 1. Problem and Setting
- **Task**: Monocular geometry estimation (MGE) — predict a metric-scale 3D point map of a scene from a single image, recovering accurate geometry, real-world metric scale, and fine-grained detail simultaneously.
- **Input/Output**: Single RGB image → metric-scale 3D point map (with predicted camera intrinsics) + metric depth (the z-channel of the point map).
- **Difficulties**:
  - Three desirable MGE properties — geometry accuracy, metric scale, and detail granularity — are typically achieved by mutually exclusive design choices; no prior method has all three.
  - Predicting absolute metric point maps directly suffers from focal-distance ambiguity.
  - Real training data has noisy and incomplete depth at object boundaries (LiDAR / SfM misalignment), which washes out fine-grained detail.
  - Synthetic data labels are sharp but have a domain gap to real images; pure synthetic-data methods (Depth Anything V2, Depth Pro second stage) sacrifice accuracy for sharpness.

## 2. Core Method
**Pipeline**: Single RGB image → DINOv2 ViT backbone → Conv neck + Conv head for affine-invariant point map → MLP head for metric scale factor (decoupled) → metric-scale point map × mask → final metric geometry; trained with mixed synthetic + real data, where real data is refined by synthetic-style sharpness priors.

**Key components**:
1. **Decoupled affine-invariant point map + global scale factor**: Instead of predicting absolute metric point maps directly (which suffers from focal-distance ambiguity), the model predicts (a) an affine-invariant point map P̂ via the same robust L1 alignment solver as MoGe, and (b) a separate global scale factor ŝ via an MLP head on the CLS token. The metric point map is ŝ · P̂. This decoupled representation mitigates focal-distance ambiguity and yields more accurate results than a fully-metric output.
2. **DINOv2 ViT backbone**: Inherits DINOv2's strong visual features — same as MoGe.
3. **Unified real-data refinement pipeline**: To inject sharp detail without sacrificing real-world accuracy, the paper develops a pragmatic data-refinement approach that:
   - Filters mismatched / false depth values in real training data (mostly at object boundaries).
   - Performs edge-preserving depth inpainting using a model trained on sharp synthetic data to fill missing regions.
   - This produces sharp "synthetic-style" labels for real images, retaining their real-world geometry while gaining sharpness.
4. **Loss**: Robust L1 alignment loss on point maps (same as MoGe) plus metric-scale supervision.
5. **Mixed training**: Trained on a large corpus of synthetic + real datasets, with real data refined through the unified pipeline.

**Essential difference from existing methods**:
- Decoupled affine + global scale avoids focal-distance ambiguity while preserving relative geometry.
- Real-data refinement injects sharp detail without sacrificing real-world accuracy.
- First method to simultaneously achieve top scores on relative geometry accuracy, metric scale precision, and detail granularity.

## 3. Knowledge, Supervision, and Assumptions
- **Backbone**: DINOv2 ViT — frozen then adapted for geometry prediction.
- **Training data**: Mixed corpus of synthetic (sharp labels) and real (LiDAR / SfM-captured, refined by the unified pipeline) datasets.
- **Supervision**: Robust L1 alignment loss on point maps (vs. ground-truth 3D) + metric-scale supervision.
- **Foundation-model usage**: Heavy reliance on DINOv2 features; inherits MoGe's alignment solver and multi-scale supervision.
- **Assumptions**:
  - The focal-distance ambiguity can be mitigated by decoupling affine-invariant point map prediction from global scale prediction.
  - Real-data labels can be made "synthetic-style sharp" via filtering + edge-preserving inpainting without losing real-world accuracy.
  - DINOv2 features are sufficient as the visual backbone for monocular geometry estimation.
- **Learned vs. provided**: Camera intrinsics, affine-invariant point map, and metric scale are all predicted by the network; ground-truth geometry comes from the training datasets (synthetic + refined real).

## 4. Experiments and Findings
- **Benchmarks**: Multiple monocular geometry / depth benchmarks — comparison against Depth Anything V2, UniDepth V2, Metric3D V2, Depth Pro, MoGe†.
- **Metrics**:
  - Relative Geometry (RG) accuracy.
  - Metric Geometry (MG) precision.
  - Sharp Detail (SD) recovery.
  - Standard monocular depth metrics (AbsRel, RMSE, etc.).
- **Key results stated**:
  - MoGe-2 ranks 1st across all three dimensions (RG, MG, SD) in comprehensive evaluations (Fig. 1 of paper) — no prior method achieves all three simultaneously.
  - Open-domain metric depth predictions outperform UniDepth V2, Metric3D V2, Depth Anything V2, Depth Pro, and MoGe.
  - Real-data refinement strategy significantly improves sharpness over MoGe without compromising geometric accuracy.
  - Decoupled scale design yields more accurate results than direct metric point-map prediction.
- **Ablations** (referenced in paper): shift-invariant vs affine-invariant representation; data refinement contribution; backbone choice.

## 5. Strengths and Limitations
### Strengths
- **Simultaneous RG + MG + SD**: First MGE method to achieve top scores on all three.
- **Decoupled representation**: Affine-invariant point map + global scale avoids focal-distance ambiguity and yields accurate metric geometry.
- **Unified real-data refinement**: Practical pipeline that makes real labels sharp without losing real-world accuracy — works across LiDAR and SfM artifact types.
- **Strong baseline performance**: Beats prior SOTA on monocular depth benchmarks.
- **Open source** (NeurIPS 2025).

### Limitations
- **Inherits MoGe's failure modes**: Affine-invariant point map representation is sensitive to scale alignment failures on extreme scenes.
- **Real-data refinement is a pipeline, not end-to-end**: Requires running an inpainting model on real training data; the quality of refined labels is bounded by the inpainting model.
- **Two-stage scale prediction**: The metric scale factor is decoupled from per-pixel geometry, which can produce globally-scaled but locally-inaccurate regions in some scenes.
- **Outdoor / driving focus**: Most detailed comparisons are on indoor + mixed scenes; extreme outdoor / driving conditions are less characterized.
- **Single-image only**: No temporal or multi-view consistency; out-of-distribution videos may drift.

## 6. Takeaway
MoGe-2 demonstrates that **the three long-conflicting goals of monocular geometry estimation — relative geometry accuracy, metric scale precision, and fine-grained detail — can be jointly achieved** by (a) decoupling the affine-invariant point map from the global scale factor to avoid focal-distance ambiguity, and (b) unifying real training data through a synthetic-style sharpness-refinement pipeline. This combination produces the first MGE method that simultaneously ranks first on RG, MG, and SD benchmarks, beating Depth Anything V2, UniDepth V2, Metric3D V2, Depth Pro, and MoGe. For HOI research, MoGe-2's metric-scale point maps provide calibrated single-image geometry priors useful for hand-object reconstruction when multi-view cues are weak, and its sharpness makes it a strong base for downstream contact prediction and grasp synthesis.