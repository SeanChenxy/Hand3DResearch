# ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos

## Summary
ForeHOI presents a feed-forward framework that directly reconstructs complete 3D object shapes from monocular HOI videos without per-video optimization, leveraging the ubiquity of monocular HOI videos to train on large-scale synthetic data with foundation-model-based shape priors distilled into a fast network for real-time deployment.

## 1. Problem and Setting
- Feed-forward 3D reconstruction of hand-held objects from monocular HOI videos — given a short video clip, directly predict the object's 3D shape without iterative optimization.
- Input: a short monocular RGB video clip of hand-object interaction, plus tracked hand poses.
- Output: 3D object shape (as a neural field or explicit mesh) in a single feed-forward pass.
- Task: hand-held object reconstruction with shape completion. This paper represents the "feed-forward with distilled FM priors" approach.

## 2. Core Method
- A feed-forward encoder-decoder network trained on large-scale synthetic HOI data where ground-truth shapes are available.
- Video encoder (spatio-temporal transformer or 3D CNN) processes input frames and aggregates features into a canonical object-centric representation using tracked hand/object poses.
- A shape decoder (triplane or similar efficient 3D representation) maps aggregated features to object geometry.
- During training, a pre-trained multi-view diffusion model provides feature-level distillation — the network learns to predict object features consistent with the diffusion model's internal representations, encoding the shape completion prior into the feed-forward weights.
- How FM prior is injected: diffusion model features are used as a teacher during training (knowledge distillation). The pre-trained diffusion model's internal representations encode rich 3D shape knowledge, transferred to the feed-forward network. At inference time, only the feed-forward network runs.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: pre-trained multi-view diffusion model (e.g., Zero-1-to-3 or similar) used for feature distillation during training.
- Domain knowledge: hand model (MANO); synthetic HOI data provides ground-truth geometry supervision.
- Training data: large-scale synthetic HOI dataset (generated via physics simulation or grasp synthesis on 3D object assets like Objaverse), plus the pre-trained diffusion model.
- Assumption: training distribution covers real-world objects; tracked hand poses are reasonably accurate.

## 4. Experiments and Findings
- Datasets: training on a large-scale synthetic HOI dataset (generated from Objaverse objects with simulated grasps); evaluation on HO3D, DexYCB, and in-the-wild videos.
- Metrics: Chamfer distance, F-score, and inference time (milliseconds vs. minutes/hours).
- Achieves competitive shape accuracy with optimization-based methods while being 3-4 orders of magnitude faster.
- Ablation removing the diffusion distillation loss shows a clear drop in shape quality for occluded regions — the distilled FM prior is essential for hallucinating plausible unseen geometry.

## 5. Strengths and Limitations
### Strengths
- Feed-forward inference enables real-time or near-real-time HOI reconstruction.
- Distillation effectively encodes FM prior knowledge into a compact, fast network.
- Once trained, does not require the expensive FM at inference time.
- Addresses the key bottleneck (speed) of optimization-based methods.

### Limitations
- Requires large-scale synthetic training data, which may have a domain gap to real videos.
- Generalization to object categories unseen in training is limited.
- The distilled prior is fixed after training; cannot benefit from improved FMs without retraining.
- Feed-forward nature may produce deterministic, less diverse outputs.
- Training is computationally expensive (requires both synthetic data generation and diffusion model feature extraction).

## 6. Takeaway
ForeHOI represents the "distillation" approach to FM priors for shape completion: instead of using FMs at test time, it bakes their knowledge into a fast feed-forward network during training. This paradigm addresses the critical speed limitation of optimization-based FM-prior methods and points toward practical deployment of HOI reconstruction systems, while demonstrating that synthetic HOI data combined with FM distillation is a viable strategy for training HOI models without real 3D annotations.
