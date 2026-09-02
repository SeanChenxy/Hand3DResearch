# ViPRA: Video Prediction for Robot Actions

**Authors:** Sandeep Routray, Hengkai Pan, Unnat Jain, Shikhar Bahl, Deepak Pathak  
**Date:** 2025 (ICLR 2026)  
**Identifier:** [arXiv:2511.07732](https://arxiv.org/abs/2511.07732); DOI `10.48550/ARXIV.2511.07732`  
**Zotero item:** `BKE8GLDP` ([Zotero](zotero://select/library/items/BKE8GLDP))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
ViPRA turns a video prediction model into a robot policy. A video-language model is pre-trained on actionless human and robot videos to jointly predict future visual observations and fine-grained motion-centric latent actions (3–6 Hz), with perceptual losses and optical flow consistency keeping the latents physically grounded. Downstream, a chunked flow-matching decoder maps latent actions to continuous robot action sequences using only 100–200 teleoperated demonstrations, supporting smooth high-frequency control up to 22 Hz. ViPRA outperforms strong baselines with a 16% gain on the SIMPLER benchmark and a 13% improvement across real-world manipulation tasks.

## Background and Problem
Videos capture rich physical interaction but lack action labels, limiting their use in robot learning. Prior latent-action methods treat pre-training as autoregressive policy learning and use temporally coarse task-centric latents. The paper targets continuous robot control learned from actionless videos, explicitly modeling both what changes (video prediction) and how (fine-grained latent motion).

## Method
Pre-training trains a video-language model to predict future observations alongside sequences of motion-centric latent actions over short horizons; perceptual losses and optical flow consistency regularize the latents toward physically plausible behavior. For control, a chunked flow-matching decoder maps latent action chunks to embodiment-specific continuous actions, fine-tuned with only 100–200 teleoperated demonstrations, enabling cross-embodiment generalization and high-frequency (up to 22 Hz) closed-loop execution.

## Contributions
- A pretraining–finetuning framework converting video prediction into robot control via motion-centric latent actions.
- A chunked flow-matching action decoder achieving smooth, high-frequency continuous control from minimal demonstrations.
- Evidence that modeling state transitions with fine-grained latents outperforms autoregressive latent-action policy pre-training.

## Experimental Setup
Evaluation covers the SIMPLER benchmark suite (four Bridge tasks, reporting success and grasp rates) and three real-world manipulation tasks with full and partial success rates, including a bimanual setup. Baselines include methods that do not exploit video foundation models and prior latent-action approaches; ablations address multimodal pre-training priors, high-frequency control adaptation, and flow-matching decoding. Complete pre-training corpus statistics are not reproduced from the available evidence.

## Results
- SIMPLER benchmark: a 16% gain over strong baselines.
- Real-world manipulation tasks: a 13% improvement across the evaluated tasks.
- ViPRA-FM significantly outperforms baselines in the reported comparisons, and the framework adapts to high-frequency continuous control up to 22 Hz via chunked action decoding.

## Limitations
Downstream control still requires 100–200 teleoperated demonstrations per setting, so the action gap is reduced rather than eliminated. Latent actions are derived from visual motion, leaving forces and contact states outside the pre-training signal. Full per-task tables and failure cases are not reproduced from the available evidence.
