# H2R-Grounder: A Paired-Data-Free Paradigm for Translating Human Interaction Videos into Physically Grounded Robot Videos

**Authors:** Hai Ci, Xiaokang Liu, Pei Yang, Yiren Song, Mike Zheng Shou  
**Date:** 2025-12-10  
**Identifier:** [arXiv:2512.09406](https://arxiv.org/abs/2512.09406); DOI `10.48550/arXiv.2512.09406`  
**Zotero item:** `YFS6IY7N` ([Zotero](zotero://select/library/items/YFS6IY7N))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
H2R-Grounder converts ordinary third-person human-object interaction videos into motion-consistent robot manipulation videos with physically grounded contacts, using only unpaired robot videos for training. Its shared abstraction, H2Rep, represents any manipulator video as an inpainted background video plus a minimal 2D pose cue (a red dot and blue arrow for gripper position and orientation); a Wan 2.2 5B video diffusion model is LoRA-fine-tuned in-context to map H2Rep back to realistic robot videos, learning contacts and occlusions from real Franka data. On out-of-distribution DexYCB human videos it is ranked first by humans and a VLM across motion consistency, background consistency, visual quality, and physical plausibility, offering a scalable route to synthesize robot learning data from unlabeled human videos.

## Background and Problem
Robot demonstration data is scarce and lab-bound while human interaction videos are abundant, but the visual embodiment gap prevents direct reuse, and prior "robotization" pipelines that render robot arms onto inpainted human frames produce floating or misaligned arms and demand accurate camera-robot calibration unavailable in the wild. Paired human-robot video collection is prohibitively expensive, and existing generative translators either lose background consistency or generate geometrically implausible arms. The paper targets a translation framework that requires no paired human-robot videos and no test-time calibration, yet yields robot videos whose contacts, occlusions, and background are physically consistent.

## Method
The data engine builds H2Rep from unpaired robot videos: Droid clips are segmented with Grounded-SAM2, the 6-DoF end-effector trajectory is projected to 2D with calibrated cameras and rendered as a dot-arrow overlay, the arm is removed with Minimax-Remover video inpainting, and the pose cue is alpha-blended (alpha = 0.4) onto the clean background. A Wan 2.2 TI2V-5B generator is then fine-tuned in-context-both the H2Rep condition and the target robot video are encoded by the same VAE and fused via self-attention, training only Q/K/V LoRA adapters under a flow-matching objective with a fixed text prompt. At transfer time, the identical pipeline is applied to human videos (Grounded-SAM 2.1 person masks, ViTPose body pose, HaMeR hand pose, with the index-thumb midpoint and thumb direction as a surrogate gripper pose), and the frozen-base generator with lightweight LoRA produces the robot video; the authors show that in-context conditioning outperforms ControlNet-style VACE conditioning for motion-background coherence.

## Contributions
- A paired-data-free human-to-robot video translation paradigm trained only on unpaired robot videos.
- H2Rep, a simple embodiment-agnostic intermediate representation (inpainting-removed background plus 2D pose overlay) that unifies human and robot domains without calibration or 3D alignment.
- An in-context LoRA fine-tuning scheme for large video diffusion models that improves realism and temporal consistency over VACE-style conditioning for physically grounded robot video generation.

## Experimental Setup
Training uses about 76K third-person Franka manipulation videos from Droid (50 held out for validation), standardized to 1280x720 at 10 fps with clips up to 49 frames; fine-tuning runs 200 steps with mini-batch 4 on 8 NVIDIA H200 GPUs with gradient accumulation 2. Out-of-distribution evaluation uses 100 DexYCB videos (subject 01, top-down camera view) processed through the automatic annotation pipeline, plus qualitative comparisons on internet videos; rendering-based baselines (Phantom, Masquerade) are excluded because their required calibration is unavailable for in-the-wild data, leaving adapted RoboMaster and commercial editors Kling and Runway Aleph as baselines. Evaluation combines a 22-participant human study and Gemini VLM scoring (1-5) on motion consistency, background consistency, visual quality, and physical plausibility, with SSIM/LPIPS ablations on Droid; no downstream policy training is reported in the available evidence.

## Results
In the human study H2R-Grounder is the top-ranked method on all four aspects, with 54.5% first-rank preference for motion consistency, 56.8% for background consistency, 61.4% for visual quality, and 63.6% for physical plausibility, versus 2.3-18.2% for RoboMaster and at most 40.9% (visual quality) for Kling. VLM scoring agrees: 3.7 motion, 4.9 background, 4.0 visual quality, and 4.4 physical plausibility, with Kling slightly ahead only on visual quality (4.1). On Droid ablations the 5B in-context model reaches SSIM 0.82 and LPIPS 0.22, versus 0.68/0.30 and 0.71/0.27 for VACE 1.3B/14B; removing the pose indicator or LoRA drops SSIM to 0.80, and a 14B backbone gives no quality gain while cutting the generable length from 49 to 17 frames. The deployed 5B model takes about 13 seconds per frame (roughly 648 seconds for a 49-frame 704x1280 video on one H200 with 63 GB peak memory).

## Limitations
The framework supports only single-hand to single-arm translation, with bimanual extension left as future work. Because training data contains only the Franka arm, outputs are Franka-style only, and adapting to other embodiments would require per-robot fine-tuning or additional LoRA adapters. Generation is slow (seconds per frame), and the evidence reports video-translation quality rather than measured gains from training policies on the synthesized data.
