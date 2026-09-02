# mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs

**Authors:** Jonas Pai, Liam Achenbach, Victoriano Montesinos, Benedek Forrai, Oier Mees, Elvis Nava  
**Date:** 2025-12-19  
**Identifier:** [arXiv:2512.15692](https://arxiv.org/abs/2512.15692); DOI `10.48550/arXiv.2512.15692`  
**Zotero item:** `U87UUQYY` ([Zotero](zotero://select/library/items/U87UUQYY))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
mimic-video introduces the Video-Action Model (VAM) paradigm: instead of VLAs that must infer physical dynamics from scratch using robot trajectories, it pairs a pre-trained Internet-scale video model with a flow-matching action decoder conditioned on the video model's latent representations. The decoder acts as an inverse dynamics model, generating low-level robot actions from latent video-space action plans. This isolates control as the only task to learn from robot data. mimic-video achieves state-of-the-art performance on simulated and real-world manipulation — including dexterous bimanual tasks — with 10× greater sample efficiency and 2× faster convergence than traditional VLA architectures.

## Background and Problem
VLA backbones are pre-trained on static, disconnected web data, so physical causality must be learned implicitly from expensive expert robot data, creating an unsustainable collection burden. The paper argues video pre-training should capture semantics and visual dynamics jointly, leaving only low-level control for the robot-data stage; the task is generalizable manipulation across single-arm, bimanual, and dexterous embodiments.

## Method
The system conditions a flow-matching-based inverse dynamics decoder on latent representations of a pre-trained video model. The video model generates action plans in video space (semantic + dynamics); the decoder translates them into robot actions. Post-training on robot data is efficient because dynamics knowledge is already embedded in the video backbone. Predicted videos can serve as plans without decoding during autonomous execution.

## Contributions
- The Video-Action Model (VAM) class: grounding robot policies in pre-trained video models so that control is the only learned component.
- A flow-matching inverse dynamics decoder conditioned on video-model latents, converting video-space plans into low-level actions.
- State-of-the-art results on dexterous manipulation with 10× sample efficiency and 2× convergence speed versus VLA baselines.

## Experimental Setup
Evaluation spans the SIMPLER benchmark (BridgeDataV2 Widow-X embodiment with system identification and visual matching), LIBERO (Goal, Object, and Spatial suites; 50 expert demonstrations per task across 10 tasks each; simulated Panda), and real-world dexterous bimanual manipulation with two 16-DoF "mimic" hands on Panda arms, including package sorting, measuring-tape handling, and stowing tasks. Comparisons cover multiple state-of-the-art baselines; full trial counts are not reproduced from the available evidence.

## Results
- State-of-the-art performance on simulated and real-world manipulation, including dexterous bimanual tasks, in the reported comparisons.
- 10× greater sample efficiency and 2× faster convergence compared to traditional VLA architectures.
- Ablations show near-perfect success rates when conditioning on oracle latents regardless of backbone fine-tuning, and improved performance when minimizing the domain gap via fine-tuning of predicted video.

## Limitations
Execution quality depends on the video model's generation fidelity; the domain gap between predicted and real observations remains a factor that fine-tuning mitigates but does not remove. The video backbone carries the dynamics prior, so behaviors outside its pre-training distribution may still require robot data. Full failure analyses are not reproduced from the available evidence.
