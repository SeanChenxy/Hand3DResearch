# SemGrasp: Semantic Grasp Generation via Language Aligned Discretization

**Authors:** Kailin Li, Jingbo Wang, Lixin Yang, Cewu Lu, Bo Dai  
**Date:** 2024-04-04  
**Identifier:** [arXiv:2404.03590](https://arxiv.org/abs/2404.03590)  
**Zotero item:** `9HJFJSBJ` ([Zotero](zotero://select/library/items/9HJFJSBJ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
SemGrasp generates static human grasp poses controlled by language instructions by discretizing grasps into three semantic tokens (orientation, manner, refinement) with a hierarchical VQ-VAE and fine-tuning a Vicuna-7B-based multimodal LLM over object point clouds, grasp tokens, and text, supported by the CapGrasp dataset of about 260k captions and 50k grasps with low-level, high-level, and conversational annotations.

## Background and Problem
Human grasping depends on more than object geometry: intent and semantics determine whether one grips a mug by the handle for drinking or avoids a hot surface. Prior grasp generation methods use robotic poses, MANO parameters, contact regions, or implicit fields, conditioning only on shape or coarse affordance vectors; some filter sampled grasps with vision-language models but cannot integrate detailed language into generation. The paper argues that human grasp planning proceeds in three steps — choose a general orientation from object category and instruction semantics, select a grasp manner/taxonomy from intent and shape, then refine the pose for physical plausibility — and that a grasp representation should mirror this structure while embedding semantics. The task is semantic grasp generation: given an object point cloud and a language description, produce a MANO grasp aligned with the instruction.

## Method
The method has two components. (1) Grasp discretization: a hierarchical VQ-VAE (codebook K=512 entries, latent dimension 256, PointBERT encoders conditioned on the object point cloud) quantizes the grasp G=(T, theta, beta) into three tokens: <o> captures the global hand-object transform T, <m> captures local pose and shape (theta, beta) conditioned on <o>, and <r> captures delta parameters refining T, theta, beta for physical plausibility. (2) Grasp-aware language model: following LLaVA, PointBERT object features (513 embeddings of dimension 384, projected to 4096, with an extra <OS> token encoding object size) plus the frozen VQ-VAE's grasp tokens are aligned with a Vicuna-7B backbone fine-tuned via LoRA (rank 64, about 6% of parameters tuned) in two stages — multimodal alignment (predict grasp tokens from object features and text) and instruction tuning — using next-token negative log-likelihood loss. Grasp tokens are decoded back to MANO poses through the VQ-VAE decoder.

## Contributions
- A discrete, interpretable grasp representation aligned with language via three semantic tokens (orientation, manner, refinement) that mirrors the human grasping process.
- A grasp-aware MLLM integrating object, grasp, and language modalities in a unified semantic space for language-controlled grasp generation.
- CapGrasp, to the authors' knowledge the first semantic grasp dataset with low-level contact annotations, high-level intent/force captions generated with GPT-4 and GPT-4v, and conversational annotations: about 1.8k OakInk object models, roughly 50,000 hand-object grasp pairs, and about 260k captions (on average 5 per grasp).

## Experimental Setup
CapGrasp adopts the OakInk split (80% train, 10% validation, 10% test). The MLLM is trained with batch size 128 over 20 epochs on 4 A100 GPUs (80 GB), with learning rates 5e-4 and 3e-5 for the two stages and cosine annealing. Physical plausibility metrics: MPVPE (mm), penetration depth PD (cm), solid intersection volume SIV (cm3), and simulation displacement SD (cm) in PyBullet. Semantic consistency metrics: GPT-4v-assisted scoring (0-100), P-FID between generated and ground-truth point clouds, and a 5-volunteer 5-point Likert Perceptual Score. Applications use D-grasp (RaiSim) for AR/VR dynamic grasping and UniDexGrasp with ShadowHand retargeting (IsaacGym) for robotics.

## Results
For grasp reconstruction, the discrete representation reaches MPVPE 14.97 mm, PD 0.46 cm, and SIV 2.72 cm3, improving to 26% lower MPVPE and 9% lower SIV with the refinement token versus without; with test-time adaptation it attains best PD (0.37) and SIV (1.27) against GrabNet and Jiang et al. For language-guided generation, SemGrasp scores P-FID 2.28, GPT-4 score 74.5, and Perceptual Score 4.6 versus a BERT classification baseline (3.32, 47.3, 3.7). Controllability ablation shows fixing <o, m> tokens yields consistent grasp orientation and manner across object shapes. Ablations favor the three-token hierarchy over single-token (MPVPE 29.95) or two-token (25.73) variants, Vicuna over Llama (GPT-4 score 74.5 vs 58.9), two-stage training, LoRA rank 64, and including the object size token. In the AR/VR application, generated grasps achieve a 62.9% dynamic grasp success rate with D-grasp.

## Limitations
The paper's Limitations section states that SemGrasp covers static single-hand grasps (with dynamic grasps only via RL integration), leaving two directions unexplored: two-hand manipulation, which requires modeling both hands' cooperation, and end-to-end semantic grasp motion synthesis, which requires continuity of motion; both depend on extensive high-quality motion capture or synthesis data. The appendix also notes that the MLLM's grasp captioning output can hallucinate or miss interaction details.
