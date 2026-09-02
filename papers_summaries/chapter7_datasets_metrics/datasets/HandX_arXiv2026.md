# HandX: Scaling Bimanual Motion and Interaction Generation

**Authors:** Zimu Zhang, Yucheng Zhang, Xiyan Xu, Ziyin Wang, Sirui Xu, Kai Zhou, Bing Zhou, Chuan Guo, Jian Wang, Yu-Xiong Wang, Liang-Yan Gui  
**Date:** 2026-03-30  
**Identifier:** [arXiv:2603.28766](https://arxiv.org/abs/2603.28766)  
**Zotero item:** `57HD2PPL` ([Zotero](zotero://select/library/items/57HD2PPL))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HandX is a unified foundation for bimanual hand motion generation spanning data, annotation, and evaluation. It consolidates existing motion and hand-object datasets into a standardized 54.2-hour, 5.9M-frame corpus with 485.7K fine-grained text annotations, and adds newly captured 36-camera OptiTrack motion capture of dexterous two-hand interaction. A decoupled captioning pipeline extracts kinematic features and uses LLM reasoning to produce multi-granularity descriptions, and the authors benchmark diffusion and autoregressive (AR) generators under versatile conditioning, observing clear scaling trends: jointly increasing data and model capacity improves text alignment and bimanual contact accuracy.

## Background and Motivation

Whole-body motion datasets treat hands as rigid end-effectors or under-annotate finger articulation, while hand-centric datasets are object-centric, coarse, or small, so text-to-motion systems miss the fine cues that make hand motion believable: finger-level dynamics, contact timing, and inter-hand coordination. Mismatched skeletons, frame rates, and annotation protocols across sources also prevent unification, and standard generation metrics do not measure hand contact fidelity. HandX is positioned as a response to all three gaps: a hand-centric corpus with consistent representation, scalable fine-grained annotation, and contact-aware evaluation, motivated by applications in immersive media, telepresence, embodied AI, and human-computer interaction.

## Dataset Construction

Construction proceeds in two strands. First, high-quality open datasets with bimanual motion (including Motion-X, InterAct, BOTH2Hands, GigaHands, HOT3D, ARCTIC, H2O, and HoloAssist, among others cited in Table 1) are aggregated, mapped to a unified 21-joint skeletal topology and canonical right-handed coordinate system, segmented into 60-frame clips (2 seconds at 30 fps) with non-overlapping windows, and filtered by an intensity-aware rule based on weighted joint angular velocity that removes static or near-static segments (thresholds tau_hand = 25 and tau_avg = 30). Second, new data are captured in a dedicated studio with a 36-camera OptiTrack optical system; actors wear 25 miniature (3 mm) reflective markers per hand covering wrist, palm, MCP/PIP/DIP joints, and fingertips, and skeletons are reconstructed by offsetting markers along anatomical normals with per-frame wrist optimization against calibrated bone lengths. Compared in Table 1, HandX reaches 54.2 hours, 5.9M frames, and 485.7K text annotations with fine-grained granularity, and exhibits the highest contact ratio, contact duration, contact frequency, and motion intensity among the compared datasets.

## Evaluation Protocol

Annotation is produced by a decoupled two-stage pipeline: six kinematic descriptors (finger flexing, finger spacing, finger-finger distance, finger-palm distance, palm-palm relation, wrist trajectory) are computed and segmented into JSON-formatted events, after which an LLM prompted with these events generates five annotations of increasing detail, each structured as left-hand, right-hand, and two-hands-relation descriptions. A 20-participant user study scores annotation quality against a Gemini 3 Pro direct video captioning baseline and motion quality against GigaHands and HoloAssist. Generation is benchmarked with a diffusion model (T5 text encoder; separately encoded left/right/interaction prompts with learnable CLS tokens cross-attended to motion embeddings; decoder sizes of 4, 8, 12, and 16 layers) and an AR model (Finite Scalar Quantization tokenizer with local motion representation and text-prefix next-token prediction; codebooks of 512 to 4,096 and 8 to 16 layers). Masked partial denoising supports motion in-betweening, keyframe control, wrist trajectory control, hand-reaction synthesis, and long-horizon generation within one model. Metrics follow the text-to-motion protocol (FID, Diversity, R-Precision, Matching/MM Dist with an InfoNCE-trained evaluator) plus contact precision, recall, and F1 computed from intra-hand thumb-to-fingertip and inter-hand contacts with a 2 cm threshold.

## Findings and Analysis

Scaling data and capacity together consistently improves R-Precision and contact metrics: with the full training set, the 12-layer diffusion model achieves the best overall contact performance (intra-hand CF1 of 0.641), while a 6.7x larger variant degrades across all metrics, revealing a saturation point beyond which extra capacity hurts. For AR models, enlarging the FSQ codebook alone does not help; codebook and model size must scale jointly. Under a fixed 5% data budget, Top-3 R-Precision follows an approximately log-linear law in compute, Rprec = 0.4391 x log10(FLOPS) - 3.8707, with a correlation coefficient of 0.96. A user study on data scaling prefers the model trained on 100% of the data (48% of votes versus 33% for 5% and 19% for 20%). Qualitatively, larger models produce better text-aligned motion with improved bimanual contact, and the learned skills are demonstrated on a real humanoid platform with dexterous robot hands.

## Contributions

A large-scale bimanual motion corpus (54.2 hours, 5.9M frames, 485.7K fine-grained texts) built by consolidation with strict quality control plus new high-fidelity OptiTrack capture of dexterous two-hand interaction; a scalable decoupled captioning framework grounded in kinematic event descriptors and LLM reasoning; a benchmark of diffusion and autoregressive bimanual generation with versatile masked conditioning and new hand-focused contact metrics; and an empirical scaling analysis including a log-linear compute-to-quality relationship and evidence of saturation for model-only scaling.

## Limitations

The paper's discussion names two limitations explicitly: the dataset, despite its scale, remains finite in volume and diversity and cannot cover the full spectrum of human dexterity or interaction scenarios; and because part of the corpus is aggregated from public sources, residual quality issues such as minor jitter or kinematic implausibility cannot be fully eliminated despite filtering and interpolation. The paper also notes that scaling gains are not strictly monotonic for every metric, and that model-only over-scaling yields negative returns.
