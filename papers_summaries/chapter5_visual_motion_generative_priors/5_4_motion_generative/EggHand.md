# EggHand: A Multimodal Foundation Model for Egocentric Hand Pose Forecasting

**Authors:** Jaeyoung Choi, Hyeondong Kim, Yujin Kim, Daehee Park  
**Date:** 2026-05-08  
**Identifier:** [arXiv:2605.07642](https://arxiv.org/abs/2605.07642)  
**Zotero item:** `289LMFZJ` ([Zotero](zotero://select/library/items/289LMFZJ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

EggHand is a foundation-model framework for forecasting future 3D hand pose sequences from egocentric video. It couples the action decoder of a Vision-Language-Action model (GR00T-N1.5-3B), which captures action-structured motion dynamics, with a frozen egocentric video-text encoder (EgoVideo) that is robust to the viewpoint shifts caused by ego-motion. On EgoExo4D it reduces wrist-relative MPJPE from 0.116 to 0.076 m versus EgoH4 (-34.5%) and MPJPE-F by -45.4%, stays robust on the top-10% highest-ego-motion clips, and supports language-conditioned, controllable forecasting.

## Background and Problem

Egocentric hand pose forecasting predicts how articulated 3D hands will evolve over time from first-person video, a prerequisite for intention understanding and proactive embodied applications such as AR/VR assistance and human-robot collaboration. The task is hard because hand motion is driven by complex human intent, involves highly dexterous articulation, and is observed under drastic ego-motion-induced viewpoint changes. Prior methods (EMAG's ego-motion-aware 2D trajectory prediction, EgoH4's body-pose-based 3D forecasting, transformer-based gesture/contact predictors) are task-specific, rely on low-level motion cues without modeling high-level semantics, or assume unrealistic inputs such as full-body pose or 3D tracking; generic VLM decoders built for low-dimensional 2D trajectories are ill-suited to articulated hand kinematics. EggHand formulates forecasting as multimodal contextual reasoning over motion, scene context, and intent, predicting future 3D joint positions in a normalized egocentric canonical frame anchored at the first observation timestep, without body pose or external tracking.

## Method

EggHand fuses two pretrained models. (1) The EgoVideo egocentric video-text encoder (kept frozen) encodes the observation RGB frames and a text description of the situation — the text can be generated from sampled frames with GPT-4o-mini — producing viewpoint-aware context stable under ego-motion; both streams are projected to a shared latent space via lightweight adapters with temporal/positional encodings. (2) The GR00T-based VLA action decoder, initialized from GR00T-N1.5-3B with flow-matching pretraining, receives the multimodal context plus the observed past 3D hand poses through a cross-attention encoder, and is fine-tuned with lightweight adapters to output deterministic future hand poses. Training combines an absolute L1 coordinate loss (MPJPE), a wrist-centered relative-coordinate loss that counters global drift, and a pairwise intra-hand distance regularizer (L2) for anatomical consistency, weighted (0.6, 0.2, 0.2).

## Contributions

1. A unified foundation model pairing a VLA action decoder with an egocentric video-text encoder, robust to ego-motion while jointly reasoning over scene semantics and 3D hand dynamics. 2. A geometry-aware training objective jointly enforcing absolute pose accuracy, wrist-anchored stability, and intra-hand geometric consistency. 3. State-of-the-art results on EgoExo4D with controllable prediction driven by high-level task prompts, validated by modality-corruption and ego-motion-stratified ablations.

## Experimental Setup

Evaluation uses EgoExo4D, with synchronized egocentric RGB at 10 fps and calibrated 3D hand annotations (42 joints, 21 per hand) from Meta Aria glasses; no exocentric views, body tracking, or external 3D signals are used. Following EgoH4, 20 observation frames (2 s) predict 10 future frames (1 s); four RGB frames are sampled and resized to 224x224; clips are converted to LeRobot format and coordinates min-max normalized. Baselines: Static (mean pose), CVM (constant velocity), USST, and EgoH4 under a unified re-evaluation protocol. Metrics: ADE/FDE on wrist trajectories and wrist-relative MPJPE/MPJPE-F, in meters. Training uses AdamW with a cosine schedule and 5% warm-up.

## Results

Under the EgoH4 protocol, EggHand achieves ADE 0.271, FDE 0.271, MPJPE 0.076, and MPJPE-F 0.077, versus EgoH4's 0.267/0.333/0.116/0.141 — comparable ADE (+0.004) but FDE -18.6%, MPJPE -34.5%, MPJPE-F -45.4%; Static and CVM sit at MPJPE 0.166 and USST at ADE 0.562. Loss ablations show the wrist-relative loss gives the largest gain (MPJPE 0.071, ADE 0.258 versus 0.077/0.288 with the absolute loss only). Modality corruption: Gaussian-noise vision raises MPJPE to 0.083 and dummy language to 0.077, and corrupting both degrades to 0.093, showing complementary roles — vision anchors geometry, text reinforces task intent. On the top-10% highest-ego-motion subset, EggHand reaches ADE 0.276/MPJPE 0.106 versus EgoH4 (0.294/0.139) and an InternVL3.5-1B-encoder variant (0.303/0.112). Randomly initializing the action head instead of VLA pretraining degrades ADE to 0.321 and MPJPE to 0.089. Qualitative results cover piano playing, COVID-19 rapid-antigen-test handling, and bike repair with sparse visibility.

## Limitations

The paper states that EggHand consumes past 3D hand poses from an off-the-shelf estimator and therefore inherits a structural dependency on that upstream module; end-to-end coupling with the hand pose estimator is left as future work, along with extending the prediction horizon and transferring predicted hand poses to humanoid manipulation policies. The paper also notes ADE is marginally higher than EgoH4 (0.271 versus 0.267), meaning the method's advantage is concentrated in FDE and joint-level accuracy rather than wrist-trajectory ADE.
