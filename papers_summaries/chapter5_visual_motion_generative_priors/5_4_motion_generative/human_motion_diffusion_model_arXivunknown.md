# Human Motion Diffusion Model

**Authors:** Guy Tevet, Sigal Raab, Brian Gordon, Yonatan Shafir, Daniel Cohen-Or, Amit H. Bermano  
**Date:** 2022-09-29  
**Identifier:** [arXiv:2209.14916](https://arxiv.org/abs/2209.14916)  
**Zotero item:** `C6P5ITAY` ([Zotero](zotero://select/library/items/C6P5ITAY))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Motion Diffusion Model (MDM) is a lightweight, transformer-encoder-based classifier-free diffusion model for human motion generation that predicts the clean sample x0 rather than the noise at each denoising step, enabling direct use of geometric losses (position, velocity, foot contact). Trained in about 3 days on a single NVIDIA RTX 2080 Ti, MDM achieves state-of-the-art text-to-motion FID on HumanML3D (0.544) and KIT (0.497), outperforms action-to-motion specialists on HumanAct12 and UESTC, and supports in-betweening and body-part editing through inference-time inpainting.

## Background and Problem

Human motion generation from text or action labels is hard because the mapping is inherently many-to-many (one prompt admits many valid motions), data acquisition is costly, and humans are highly sensitive to motion artifacts. Prior text-to-motion approaches built on auto-encoders or VAEs constrain the modeled distribution, while diffusion models suit the many-to-many setting but are resource hungry and hard to control. The paper defines the problem as building a diffusion generator for motion sequences conditioned on text, action classes, or nothing at all, that is cheap to train, controllable, and benefits from established motion-domain geometric regularization.

## Method

MDM represents motion as a sequence of joint positions and/or rotations and denoises it with an encoder-only transformer instead of a U-net, matching the temporal, non-spatial structure of motion data. The time step and condition code are projected to tokens, and the network directly predicts the clean sample (following DALL-E 2) with the L_simple objective, which makes geometric losses — position (via forward kinematics when predicting rotations), velocity, and foot-contact losses that nullify foot velocity on ground contact — directly applicable. Text conditions come from a frozen CLIP ViT-B/32 encoder; actions use learned class embeddings; training uses classifier-free guidance (condition dropped for 10% of samples, guidance scale s=2.5 at sampling) to trade off diversity and fidelity. Editing (temporal in-betweening and spatial body-part editing) is adapted from diffusion image inpainting and applied only at sampling time by overwriting the known portion of x0 each step.

## Contributions

- A motion-specific diffusion architecture: transformer encoder backbone with sample prediction instead of noise prediction, enabling geometric losses in the diffusion loop.
- A generic conditioning framework covering text-to-motion, action-to-motion, and unconditioned generation with one model design.
- State-of-the-art results on leading benchmarks while training on lightweight hardware (about 3 days on a single mid-range GPU).
- Training-free motion editing applications: text-conditioned or unconditional in-betweening and body-part re-synthesis via joint-space inpainting.

## Experimental Setup

Text-to-motion uses HumanML3D (14,616 motions, 44,970 text descriptions, from AMASS and HumanAct12) and KIT (3,911 samples), evaluated with R-precision (top-3), FID, MultiModal-Dist, Diversity, and Multimodality (20 evaluation runs, 95% confidence intervals). Action-to-motion uses HumanAct12 (about 1,200 clips, 12 classes) and UESTC (40 classes, 40 subjects, 25K samples, cross-subject protocol) with FID, accuracy, diversity, and multimodality. A 31-user side-by-side study on KIT compares MDM against prior work and ground truth. Models use T=1000 noising steps with a cosine schedule, 8 transformer layers, latent dimension 512, batch size 64, trained 500K steps (text-to-motion), 750K (HumanAct12), and 2M (UESTC); the reported checkpoint minimizes FID.

## Results

On HumanML3D, MDM reaches FID 0.544, R-precision 0.611, MultiModal-Dist 5.566, and Multimodality 2.799, versus T2M's FID 1.067 and R-precision 0.740 — best FID, Diversity, and Multimodality, though R-precision trails T2M. On KIT, MDM's FID 0.497 is far below T2M's 2.770 (R-precision 0.396 vs. 0.693). On HumanAct12, MDM attains FID 0.100 with 0.990 accuracy, versus ACTOR 0.120/0.955 and INR 0.088/0.973; on UESTC, MDM's FIDtest 12.81 beats INR 15.00 and ACTOR 23.43 with 0.950 accuracy. The foot-contact loss slightly worsens FID (0.080 without) but visibly removes shakiness and foot sliding. In unconstrained synthesis on HumanAct12, MDM scores FID 31.92 versus dedicated unconditional model MoDi's 13.03 and ACTOR's 48.80. In the user study, MDM was preferred over JL2P 90.4%, TEMOS 59.4%, T2M 54.8%, and even over real ground-truth motions 42.3% of the time.

## Limitations

The paper itself notes that the diffusion approach has a long inference time, requiring about 1000 forward passes per result; although the small motion model keeps this to roughly a minute per sequence, it remains a compromise relative to single-pass generators. The authors also flag that better control mechanisms integrated into the generation process and a wider range of applications are open directions, and the results indicate a fidelity-diversity trade-off managed through the guidance scale (R-precision below T2M on both text benchmarks despite much better FID).
