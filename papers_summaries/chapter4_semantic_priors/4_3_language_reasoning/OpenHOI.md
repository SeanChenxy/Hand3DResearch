# OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model

**Authors:** Zhenhao Zhang, Ye Shi, Lingxiao Yang, Suting Ni, Qi Ye, Jingya Wang  
**Date:** 2025-05-25  
**Identifier:** [arXiv:2505.18947](https://arxiv.org/abs/2505.18947)  
**Zotero item:** `4GPAL9Z9` ([Zotero](zotero://select/library/items/4GPAL9Z9))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
OpenHOI is presented as the first open-world hand-object interaction (HOI) synthesis framework: a fine-tuned 3D multimodal large language model (MLLM) grounds object affordances and decomposes free-form instructions into sub-tasks, and an affordance-driven diffusion model with training-free physical refinement generates long-horizon HOI sequences for unseen objects, outperforming Text2HOI, MotionGPT, MDM, and TM2T on GRAB, ARCTIC, and cross-dataset H2O evaluations.

## Background and Problem
Generating 3D HOI sequences from natural language is critical for AR/VR and dexterous robotics. Traditional methods rely on handcrafted motion priors, while diffusion approaches (e.g., Text2HOI) map text to action but handle only closed sets of objects and predefined tasks, and produce short sequences. LLM-based methods such as HOIGPT generate long sequences but lack 3D perception, so they cannot ground instructions in object geometry; 3D MLLMs handle affordance perception but not interaction synthesis. The paper targets open-world HOI synthesis: generating long-horizon manipulation sequences for novel objects guided by open-vocabulary, intent-centric instructions (e.g., "I'm feeling thirsty, could you find a water bottle and take a sip?").

## Method
Stage 1 fine-tunes ShapeLLM-7B (ReCon++ 3D point encoder, LLaMA language model, LoRA) for joint affordance grounding and task decomposition. A special segmentation token <AFF> (following LISA/PixelLM-style designs) is added to the vocabulary; the MLLM outputs sub-task text interleaved with <AFF> tokens whose hidden states are decoded into per-point affordance masks via cross-attention. Training is coarse-to-fine: first on large-scale static object-centric affordance datasets, then on dynamic HOI data with voxel-derived fine-grained affordance maps, supervised by autoregressive cross-entropy (sub-tasks) plus Dice and BCE losses (affordance), reaching an affordance AUC of 87.02 versus 68.77 without fine-tuning. Stage 2 trains a transformer-based HOI diffusion model (T=1000, cosine schedule, classifier-free guidance with 10% condition masking and guidance scale 2.5) conditioned on the affordance map, CLIP sub-task embedding, and object point cloud, with distance-map and relative-orientation losses. At inference, training-free refinement injects three objectives (affordance alignment, penetration avoidance, motion in-between for sub-sequence transitions) via a DSG-inspired Spherical Gaussian Constraint that mixes steepest-descent and random sampling directions without distribution shift.

## Contributions
- The first open-world HOI synthesis framework generating long-horizon sequences for unseen objects from open-vocabulary instructions.
- Fine-tuning of a 3D MLLM that jointly learns geometric affordance priors and semantic task decomposition with multi-token <AFF> affordance decoding.
- Affordance-driven HOI Diffusion combined with training-free physical refinement (affordance, penetration, and motion in-between guidance) that preserves the learned diffusion distribution.
- State-of-the-art generalization to unseen objects, multi-stage tasks, and complex instructions, validated with ablations and significance tests.

## Experimental Setup
Training and evaluation use GRAB (51 everyday objects, single-hand) and ARCTIC (bimanual interaction with articulated objects), each split 80% training / 20% unseen testing; low-level motion descriptions are rewritten into open-vocabulary instructions by an LLM. An extreme-case test evaluates models trained on GRAB or ARCTIC on the entirely unseen H2O dataset. Experiments run on an NVIDIA A100. Metrics: MPJPE (hand joints), FOL (final object location error), FID in a pre-trained motion feature space, Diversity (distance to ground truth), and MModality. Physical realism and intersection volume (IV) are compared against Text2HOI.

## Results
On GRAB, OpenHOI achieves MPJPE 47.64 (seen) and 51.34 (unseen) versus 56.29/60.67 for Text2HOI, with FOL 0.26/0.27 and FID 26.43/28.29. On ARCTIC, it reaches MPJPE 45.15 (seen) and 47.25 (unseen) versus 52.16/57.83 for Text2HOI. Cross-dataset on H2O it scores MPJPE 75.78 (GRAB-trained) and 81.36 (ARCTIC-trained), best among all baselines despite the full distribution shift. Physical realism improves to 0.93/0.89 (seen/unseen) versus 0.87/0.79 for Text2HOI, with lower intersection volume (9.25/10.35 versus 11.74/14.63). Motion in-between refinement reduces SmoothRate from 38.18 to 2.98 on GRAB seen. Ablations on both datasets confirm contributions of affordance grounding (e.g., GRAB unseen MPJPE rises to 60.37 without it), CFG, penetration loss, and affordance loss.

## Limitations
The paper's own failure analysis and discussion note that the model cannot target specific instances (e.g., "open the second cabinet") because it has not been trained on large-scale 3D QA data, limiting logical reasoning; performance degrades after more than three consecutive actions (over 450 frames) due to accumulated error; fine-grained dynamics such as fluid simulation for pouring remain challenging; and compositional long-horizon tasks like "cook a meal" exceed its hierarchical decomposition capability, with chain-of-thought reasoning and faster inference (DPM-Solver) suggested as future work.
