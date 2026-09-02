# Text2HOI: Text-guided 3D Motion Generation for Hand-Object Interaction

**Authors:** Junuk Cha, Jihyeon Kim, Jae Shin Yoon, Seungryul Baek  
**Date:** 2024-03-31  
**Identifier:** [arXiv:2404.00562](https://arxiv.org/abs/2404.00562)  
**Zotero item:** `TDP36NQC` ([Zotero](zotero://select/library/items/TDP36NQC))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Text2HOI is the first text-guided method to generate full 3D hand-object interaction motion sequences (bimanual MANO hands plus object translation, rotation, and articulation) from a text prompt and a canonical object mesh. It decomposes the task into three stages: a VAE-based contact map predictor conditioned on text and object scale, a Transformer-encoder diffusion model that directly estimates the clean motion with distance-map and relative-orientation geometric losses, and a non-diffusion feed-forward hand refiner that suppresses penetration and stabilizes contacts in 0.013 s. Text labels were newly annotated over H2O, GRAB, and ARCTIC. It outperforms retrained T2M, MDM, and IMOS baselines on accuracy, FID, and physical realism (e.g., 0.9218 accuracy and 0.8839 realism on GRAB) and generalizes to unseen objects.

## Background and Problem

While text-to-human-motion generation is well developed, no prior work generated 3D hand-object interaction sequences from text, and the missing object-interaction context limits the semantic expressiveness of body-motion methods. The central challenge is data scarcity: existing ground-truth HOI datasets are far from generalizable in interaction type and object category, so a model trained on them cannot capture the real-world diversity of hand type (left/right/both), object category, structure, scale, and contact regions with correct physical implications from text. Prior hand-object motion generators either handle only grasping of stationary objects or require object trajectories and initial hand poses, which a user cannot supply. The paper's task: given a text prompt, a canonical object mesh, and object scale, generate a physically and semantically plausible 3D sequence of hand and object motion — including articulated objects whose joint angle is estimated — without object trajectory or initial hand pose.

## Method

Stage 1, contact map prediction: a VAE-based network takes an FPS-sampled, scale-normalized object point cloud, CLIP text features, object scale, and a 64-d Gaussian noise vector, and outputs an N x 1 per-point contact probability map plus a 1,024-d object feature; scale conditioning makes predictions scale-variant (smaller objects yield wider contact regions). Training uses binary cross-entropy, dice, and KL losses. Stage 2, motion generation: motion is represented per frame as 99-d vectors per hand (translation plus 16 x 6 6D MANO pose) and a 10-d object vector (translation, 6D rotation, 1-d articulation angle). A Transformer-encoder diffusion model with T = 1,000 steps and a cosine schedule directly predicts the clean sample at each step (following MDM), conditioned on text features, contact map, object features, and scale, with frame-wise and agent-wise positional encodings distinguishing frames and the three agents, and masking driven by a CLIP-selected hand type and a predicted motion length (up to 150 frames). Training adds a joint-to-object distance map loss (threshold 2 cm) and a hand-object relative orientation loss. Stage 3, hand refinement: a Transformer refiner with the same positional encodings but no diffusion and no conditions takes the generated hands, joints, contact map, deformed object points, and a distance-based attention map, and outputs refined hands in one feed-forward pass, trained with L2, penetration, and contact losses.

## Contributions

1) The first approach generating 3D hand-object interaction sequences of various styles and lengths from a text prompt. 2) A compositional framework (contact map generation + motion diffusion + refinement) that models high-quality interaction from limited data, with contact maps as strong spatial priors enabling generalization to unseen objects. 3) A fast, efficient hand refinement module improving physical realism (penetration-free interaction) without any test-time optimization. 4) Newly annotated text labels for existing hand-object motion datasets (H2O, GRAB, ARCTIC), to be released publicly.

## Experimental Setup

The model is evaluated on H2O (660 interactions, 8 objects, 11 verbs; 272 auto-generated sentences), GRAB (1,335 motions, 51 objects, 29 actions; 1,104 sentences), and ARCTIC (4,597 manually annotated motions, 11 action labels; 644 sentences), with prompts following "{action} {object category} with {hand type}" plus passive/gerund augmentations. Metrics follow IMOS: accuracy (top-3, RNN-based action classifier), FID, diversity, multi-modality, plus a ManipNet-style physical realism score (0/1 per frame); experiments run 20 times with 95% confidence intervals. Baselines T2M, MDM, and IMOS are retrained on the hand-object motion data for fair comparison.

## Results

On GRAB, Text2HOI achieves accuracy 0.9218, FID 0.3017, diversity 0.8351, multimodality 0.5216, and physical realism 0.8839, versus MDM's 0.5127/0.6023/0.8012/0.5194/0.7382 and IMOS's 0.4097/0.6147/0.6861/0.2845/0.6418. On H2O it reaches accuracy 0.8295, FID 0.1744, and realism 0.7574 (MDM: 0.5832, 0.3015, 0.5572). On ARCTIC it attains accuracy 0.9205, FID 0.1329, and realism 0.8760 (IMOS: 0.8190, 0.1826, 0.7569). Ablations on GRAB show both positional encodings matter (removing both drops accuracy to 0.8294), the geometric losses help realism (without the distance-map loss, realism falls to 0.6410), and the contact map and scale conditions each contribute accuracy. The refiner is critical: without it realism is 0.8312, and removing its penetration and contact losses collapses realism to 0.6249 (without the contact loss, 0.1467), while the feed-forward refiner takes only 0.013 s versus 28.5 s for MDM and 101 s for IMOS inference. Qualitative results confirm generation on unseen objects such as a teddy bear and a microwave.

## Limitations

The authors state in the supplementary limitation section that generated motions account for relative 3D location and contact between hands and object, but forces between hand and object are missing, which would provide better physical understanding; future work should consider this aspect.
