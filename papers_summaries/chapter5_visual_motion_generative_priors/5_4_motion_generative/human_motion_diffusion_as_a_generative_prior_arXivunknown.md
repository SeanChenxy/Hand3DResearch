# Human Motion Diffusion as a Generative Prior

**Authors:** Yonatan Shafir, Guy Tevet, Roy Kapon, Amit H. Bermano  
**Date:** 2023-03-02  
**Identifier:** [arXiv:2303.01418](https://arxiv.org/abs/2303.01418)  
**Zotero item:** `HW6WGANK` ([Zotero](zotero://select/library/items/HW6WGANK))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

This paper shows that a pretrained text-to-motion diffusion model (MDM) can be reused as a generative prior for three composition tasks that the underlying data cannot support on its own: zero-shot long-sequence generation (DoubleTake, up to 10-minute motions from a model trained only on clips of up to 10 seconds), few-shot two-person interaction generation (ComMDM, trained with as few as 10 sequences), and fine-grained joint/trajectory control via model composition (DiffusionBlending), which outperforms naive motion inpainting and dedicated baselines such as TEACH and MRT.

## Background and Problem

Human motion generation is bottlenecked by data: motion capture is expensive, and existing corpora consist almost exclusively of short, single-person sequences without transition annotations. As a result, long-sequence generation with per-interval text control, multi-person interaction synthesis, and elaborate control signals (trajectories, end-effectors) are poorly served. The paper asks whether a pretrained diffusion prior on short single-person motion can be composed — sequentially, in parallel, and across models — to address these out-of-domain tasks in zero-shot or few-shot regimes instead of training dedicated models.

## Method

All three methods build on a frozen MDM (DDPM-based, CLIP text conditioning, geometric losses). (1) DoubleTake (sequential, zero-shot): a two-take inference procedure. The first take denoises all intervals in one batch, forcing roughly one-second "handshakes" (overlapping prefix/suffix frames) of neighboring intervals to be frame-wise averaged at every denoising step. The second take partially noises each transition "sandwich" (interval, handshake, interval) for T'=700 steps and refines it under a soft inpainting mask (hard mask 0.85, soft mask 0.1, with a 10-frame linear blend), then unfolds the batch into the full sequence. (2) ComMDM (parallel, few-shot): a single-layer transformer block inserted after layer 8 of two frozen MDM instances that exchanges activations and outputs a symmetric correction term for each actor, plus learned initial root poses; only this slim block is trained. (3) Model composition: MDM is fine-tuned per control task (trajectory, a single joint) by masking noise on the control features during training and cleanly propagating them during sampling; DiffusionBlending then generalizes classifier-free guidance to interpolate between aligned fine-tuned models, G_s = G_a + s*(G_b - G_a), enabling cross-combinations of control signals (e.g., s=0.5 for equal weighting).

## Contributions

- A prior-based framing of motion diffusion with three composition paradigms (sequential, parallel, model), enabling new tasks with little to no new data.
- DoubleTake: arbitrarily long motions with per-interval text control and realistic transitions, without transition-annotated training data.
- ComMDM: the first textually driven two-person motion generation, learned from about a dozen examples; the authors also contribute 5 textual annotations for 14 two-person 3DPW motions.
- DiffusionBlending: composition of separately fine-tuned control models for flexible joint and trajectory control, substantially improving over MDM inpainting.

## Experimental Setup

Datasets: HumanML3D (joint positions/rotations/velocities/foot contacts) and BABEL (SMPL representation) for long-sequence generation; 3DPW for two-person experiments (27 two-person sequences; after discarding the noisy test split, 10 training and 4 validation sequences, augmented by mirroring and cropping). Metrics: HumanML3D evaluators — top-3 R-precision, FID, Diversity, MultiModal-Dist; root and joint mean L2 error for prefix completion; user studies with 30 participants. Hardware/training: single NVIDIA GeForce RTX 2080 Ti; MDM retrained 1.25M steps on BABEL to match TEACH; DoubleTake uses 1-second transitions; ComMDM trained 240K steps (prefix completion) and 100K steps (text-to-motion); control fine-tuning 80K steps with batch size 64.

## Results

Long sequences (BABEL test): DoubleTake achieves motion FID 1.04 vs. 1.12 for TEACH, and clearly better transitions (FID 1.88 vs. 3.86 with 70-frame margins; 3.45 vs. 7.93 with 30-frame margins), without TEACH's post-processing. On HumanML3D, the full two-take method reaches FID 0.60 vs. 1.00 for the first take alone. Ten-minute coherent motions are demonstrated qualitatively. Two-person prefix completion (3DPW): ComMDM root L2 error is 0.19/0.26/0.30 m at 1/2/3 s vs. MRT's 0.13/0.21/0.25 m — MRT wins on error but produces static, unrealistic poses; in a 30-user study ComMDM is preferred over MRT for quality (79.2%), completion (64.2%), and interaction (76.7%), and over plain MDM (65.0% quality). Control (HumanML3D test): fine-tuned models cut FID from 0.98 to 0.54 (trajectory) and from 0.82 to 0.34 (left wrist); DiffusionBlending yields FID 0.22 vs. 1.18 for MDM inpainting on left wrist + trajectory and 0.18 vs. 0.81 on left wrist + right foot.

## Limitations

The authors state that long-sequence quality remains bounded by the underlying prior, with possible inconsistencies between distant intervals, and that interacting with rich environments is not addressed. ComMDM synchronizes motions well but generalizes only to interaction types seen during training (limited by roughly 10 samples), and valid inter-person contacts are not yet modeled. The techniques are presented as domain-agnostic but were validated only on human motion. The paper also notes methodologically that joint-error metrics favor dull, low-frequency motion, which motivates its preference for perceptual/user-study evaluation of two-person results.
