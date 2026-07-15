# HandX: Scaling Bimanual Motion and Interaction Generation

## Summary
HandX is a large-scale bimanual hand-motion benchmark that consolidates 54.2 hours of contact-rich two-hand motion with fine-grained multi-level textual descriptions, pairs it with two benchmark generative models (diffusion and autoregressive over FSQ tokens), and demonstrates that jointly scaling dataset size and model capacity yields consistent improvements in text alignment and bimanual contact accuracy.

## 1. Problem and Setting
- Task: Generate realistic, text-conditioned bimanual hand motion that captures fine-grained finger articulation, well-timed contact, and inter-hand coordination, plus a unified benchmark to evaluate these properties.
- Input: Natural-language text prompt decomposed into three sub-prompts — left-hand description, right-hand description, and inter-hand interaction description (T = {T_L, T_R, T_I}); optional conditioning signals such as keyframes, wrist trajectories, or partial hand poses.
- Output: A two-hand motion sequence p = {p_1, …, p_F} with p_i ∈ R^{2J×3} representing 3D joint coordinates of both hands per frame (J joints per hand), with optional MANO-mesh recovery via post-optimization; or a complete sequence at versatile conditioning modes (motion in-betweening, keyframe-guided, wrist-trajectory control, hand-reaction synthesis, long-horizon).
- Span: Hand motion, not full body. Per-frame hand skeleton; two hands jointly.
- Why difficult: Most motion corpora use coarse hand annotations or treat hands as rigid SMPL end-effectors; hand-object interaction datasets are object-centric and rarely cover inter-hand contact dynamics; mismatched skeletons, frame rates, and protocols block source unification; existing metrics rarely evaluate hand fidelity or bimanual coordination; manual annotation at this scale is prohibitive.

## 2. Core Method
The HandX framework has three parts.

**Dataset construction.**
1. Aggregate and canonicalize public sources — Motion-X, InterAct, BOTH2Hands, HandDiffuse, InterHand2.6M, GigaHands, HOT3D, ARCTIC, H2O, HoloAssist — into a unified skeletal representation and coordinate system, filter low-quality or implausible sequences, segment into clips, and apply an intensity-aware filter based on joint angular velocity to remove dominated static / near-static segments.
2. Capture new data with a 36-camera OptiTrack optical mocap system in a dedicated studio, with 25 reflective markers per hand covering wrist, palm, fingers, and fingertips. Joint centers and bone-length anatomical constraints are estimated and refined per frame for kinematic consistency, targeting high-quality bimanual interaction with rich inter-hand contact.

**Two-stage automatic captioning.**
1. Kinematic feature extraction. Compute kinematic descriptors (finger flexion, finger-palm distances, inter-hand spatial relations) per frame, segment each descriptor's temporal evolution into events (change intervals + stable intervals), and emit a structured JSON representation.
2. LLM reasoning over the JSON. Prompt a large language model to produce five levels of descriptions (concise / balanced / comprehensive) that explicitly cover left hand, right hand, and inter-hand relations, and require reporting critical events (contact, separation, hyperextension) with temporal context.

**Benchmark generative models.**
- *Diffusion model.* Each hand joint is represented by 3D coordinates plus a compact rotation scalar. Text embeddings for T_L, T_R, T_I are encoded separately (each with a learnable CLS token) and cross-attended with the noisy motion latent z_t, then fused via residual: z̃ = z_t + Σ_{k∈{L,R,I}} CrossAttention(z_t, T_k). An MLP decoder G maps z̃ to motion. A single model supports versatile generation via inference-time partial denoising: at each step the input constraint is blended with the current sample x_t, enabling in-betweening, keyframe-guided, trajectory-controlled, hand-reaction, and long-horizon generation from one trained network.
- *Autoregressive model.* A 1D CNN encoder-decoder tokenizer is trained with a Finite Scalar Quantization (FSQ) bottleneck. A T5 text-prefix transformer predicts the next FSQ token causally conditioned on y_<k and the text prefix, with bidirectional attention over text and causal attention over motion. A downsampling factor of 2 is used in time. Decoding is deterministic (argmax).

## 3. Knowledge, Supervision, and Assumptions
- Training data: aggregated and canonicalized public hand + egocentric + HOI motion corpora plus newly captured OptiTrack bimanual interactions (HandX, 54.2 h / 5.9M frames / 485.7K text annotations).
- Supervision signals: mocap marker trajectories → joint centers with bone-length constraints; per-frame joint coordinates plus rotation scalar; LLM-generated multi-granularity text descriptions aligned with kinematic events; intensity-aware quality filtering (joint angular velocity).
- Domain knowledge: skeletal canonicalization across heterogeneous sources; explicit modeling of inter-hand contact events; structural separation of left-hand / right-hand / interaction descriptions; anatomical bone-length constraints during mocap reconstruction; partial-denoising as a generic constraint-injection mechanism.
- Foundation models used: T5-based text encoder for the autoregressive model; large LLM (closed-form, not named in the paper) for automatic captioning; deterministic decoding with argmax at inference.
- Assumptions: (i) structured kinematic events + LLM reasoning can substitute for expensive manual annotation of bimanual motion; (ii) masking-conditioned diffusion can replace task-specific architectures for versatile generation; (iii) bimanual hand motion can be generated independently of full body, with later optional integration onto a humanoid.

## 4. Experiments and Findings
- Dataset comparison (Table 1). HandX is 54.2 h (HQ) / 5.9M frames with fine-grained text granularity and 485.7K text annotations, vs Motion-X (144.2 h, coarse, 8.1K text), InterAct (30.7 h, coarse, 48.6K text), BOTH2Hands (8.31 h, coarse, 23.5K text), HandDiffuse (2.0 h, no text), InterHand2.6M (24.0 h, no text), GigaHands (2.58 h HQ / 34.0 h raw, coarse, 84K text), HOT3D (0.44 / 3.90 h, no text), ARCTIC (1.06 / 2.02 h, action, no text), H2O (0.47 / 1.06 h, action, no text), HoloAssist (49.3 / 161.2 h, coarse, 1.8K text). HandX also reports the highest contact ratio, contact frequency, and contact duration among HOI / egocentric sources (Figure 1 right).
- Diffusion model ablations (Table 2). Sweeping decoder layers {4, 8, 12, 16} and data ratios {5%, 20%, 100%}, the 12-layer model trained on 100% data is best overall (Top-1 R-Prec 0.427, Top-3 R-Prec 0.631, FID 1.349, intra-hand CF1 0.641). 16 layers or further scaling to a 6.7× larger variant degrades all metrics, indicating a saturation point.
- Autoregressive model ablations (Table 3). Sweeping FSQ codebook {512, 1024, 2048, 4096} and AR model size {4.63M – 215.31M}, best FID (1.721) at codebook 4096 + 215.31M params; best Top-1 R-Prec (0.384) at codebook 1024 + 4.63M params. Codebook scaling alone does not reliably help; the gains come from jointly increasing model size and codebook size.
- Computational scaling (Figure 4). On a fixed 5% data budget, Top-3 R-Precision follows a near log-linear relationship with FLOPs: R_prec = 0.4391 × log10(FLOPS) − 3.8707, with correlation 0.96.
- Qualitative (Figures 3, 5, 6). Show that larger data and more layers improve text alignment and contact realism, supporting a humanoid + dexterous-hand retargeting demo.
- New metrics: contact precision (C_prec), recall (C_rec), and F1 (C_F1) at a 2 cm threshold, computed from inter-hand contact events extracted from interaction annotations.

## 5. Strengths and Limitations
### Strengths
- Largest reported bimanual hand motion corpus with fine-grained, multi-level text descriptions; the canonicalization pipeline unifies heterogeneous sources and the new mocap capture fills the contact-rich inter-hand gap that existing HOI corpora leave open.
- Decoupled kinematic-event-then-LLM captioning yields fine-grained, multi-level, and diverse annotations without manual labeling.
- Single diffusion model supports a wide range of versatile tasks (in-betweening, keyframe-guided, wrist-trajectory, hand-reaction, long-horizon) through inference-time partial denoising, avoiding task-specific architectures.
- Concrete, multi-axis scaling study across diffusion depth, data fraction, AR model size, and FSQ codebook size, with a fitted FLOPs vs R-Precision log-linear law and clear saturation behavior.

### Limitations
- Saturation: pushing the diffusion model to 16 layers or 6.7× more parameters degrades performance, indicating the data / capacity regime is narrow.
- AR model is sensitive to codebook–capacity matching; over-scaling one component alone reduces R-Precision and F1.
- Captured data uses a marker-based optical mocap system in a controlled studio, which limits visual diversity and may not capture in-the-wild hand appearance and occlusion patterns.
- Focus is hand-only; full-body coherence (legs, locomotion, gaze) is not modeled, and integration onto the humanoid is presented as a demo rather than a full-body policy.
- Bimanual evaluation is dominated by inter-hand contact metrics; grasp stability with objects, force, and physics plausibility are not explicitly scored.
- The new data scale (54.2 h HQ) is modest compared with HOI pretraining corpora; the scaling-law finding is anchored to 5% of this small set.

## 6. Takeaway
HandX reframes bimanual hand motion generation as a data-plus-benchmark problem: a unified, contact-rich corpus with fine-grained multi-level text, paired with a versatile masking-conditioned diffusion model and a tokenized autoregressive baseline, shows that jointly scaling data and model capacity — not either alone — is the lever that drives text alignment and bimanual contact fidelity, and that bimanual hand generation has identifiable saturation regimes that future work must respect.
