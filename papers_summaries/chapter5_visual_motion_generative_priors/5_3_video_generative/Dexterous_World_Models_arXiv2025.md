# Dexterous World Models

**Authors:** Byungjun Kim, Taeksoo Kim, Junyoung Lee, Hanbyul Joo  
**Date:** 2025-12-19  
**Identifier:** [arXiv:2512.17907](https://arxiv.org/abs/2512.17907)  
**Zotero item:** `2NH94RGS` ([Zotero](zotero://select/library/items/2NH94RGS))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Dexterous World Model (DWM) is a scene-action-conditioned video diffusion framework that converts static 3D scene reconstructions into interactive digital twins: given a static scene rendering along a camera trajectory and an egocentric hand motion sequence, it generates temporally coherent videos of plausible dexterous interactions such as grasping, opening, and moving objects. The key design is to model only the residual visual dynamics induced by hand actions while preserving unaltered scene regions, and DWM further supports simulation-based action evaluation by ranking candidate actions against text or image goals.

## Background and Problem

Advances in 3D reconstruction make it easy to build realistic digital twins of everyday environments, but these twins remain static and support only navigation and view synthesis without embodied interactivity. Existing video world models either condition on camera motion or text, which cannot precisely specify hand configurations and fine-grained temporal control, or hallucinate both the scene and its evolution from a single frame, entangling background synthesis with action-driven dynamics and breaking causal consistency. The paper formulates world modeling as predicting the environment state induced by actions while conditioning on the known static scene, so that the model learns a dynamics term driven by dexterous hand manipulation and an observation model determined by the camera trajectory, rather than re-synthesizing the entire scene.

## Method

- DWM is a latent video diffusion transformer conditioned on two egocentric signals concatenated with the noisy latent: a static-scene video rendered along the specified camera trajectory for spatial consistency, and a hand-mesh rendering video encoding geometry and motion cues, plus an optional text prompt for semantic guidance.
- It is initialized from a pretrained video inpainting diffusion model (CogVideoX-Fun); with a full mask such a model behaves as a near-identity operator with a generative prior, so training steers the model to preserve static appearance and camera motion while synthesizing only the interaction-induced residual change.
- Because no real dataset provides paired static scene renderings and interaction videos under identical camera trajectories, the authors build a hybrid dataset: synthetic egocentric interactions from TRUMANS (SMPL-X actors, head-mounted virtual camera) yield exactly aligned interaction, static-scene, and hand-mesh videos, while fixed-camera real-world videos from TASTE-Rob pair the repeated first frame as the static-scene video with hand meshes predicted by HaMeR.
- A dedicated evaluation protocol uses Aria Glasses with millimeter-level SLAM and a 3D Gaussian reconstruction from pre-action frames to create paired static/interaction videos under dynamic egocentric viewpoints.

## Contributions

- A new world-modeling formulation that separates the static world from its dynamics and conditions video diffusion jointly on scene renderings and dexterous hand actions, enabling embodied simulation in known 3D environments.
- A residual-dynamics conditioning scheme that leverages inpainting priors as an identity-preserving initialization, and pixel-aligned hand-mesh conditioning instead of pose parameters or hand masks.
- A hybrid training dataset construction combining synthetic aligned egocentric interactions with fixed-camera real-world interaction videos, extending supervision beyond purely synthetic dynamics.
- A simulation-based action evaluation protocol that scores candidate actions by VideoCLIP similarity (text goals) or LPIPS distance (image goals), enabling goal-driven action selection without reward functions or real-world trials.

## Experimental Setup

- A benchmark of 144 samples pairs static scene videos, hand-mesh videos, and ground-truth interaction videos: 48 held-out synthetic TRUMANS sequences with dynamic cameras, 48 real-world static-camera TASTE-Rob videos, and 48 custom-captured real-world dynamic-camera samples (pick-and-place, articulated manipulation such as opening a washing machine or folding a chair, and counterfactual dynamics such as elevator buttons and faucets).
- Baselines are CVX SDEdit (CogVideoX denoising, noise strength 0.75, 50 steps), a fine-tuned CogVideoX-Fun without hand conditioning, and InterDyn for the static-camera setting.
- Metrics are PSNR, SSIM, LPIPS, and DreamSim against ground truth, averaged over three random-seed generations per sample.
- Implementation: CogVideoX-Fun-V1.5-5B-InP base model, 720x480 resolution, 49 frames, LoRA fine-tuning (rank 64), roughly 10 days on 4 NVIDIA A100 GPUs.

## Results

- DWM outperforms all baselines on every metric and setting: on synthetic dynamic-camera data it reaches 25.031 PSNR, 0.844 SSIM, 0.289 LPIPS, and 0.086 DreamSim versus 20.541/0.767/0.370/0.175 for the strongest fine-tuned baseline; on real-world static-camera data 21.547/0.816/0.227/0.057; and on the unseen real-world dynamic-camera data 21.654/0.550/0.557/0.225.
- Ablations show that adding fixed-camera real-world training data improves real-world dynamic-camera DreamSim from 0.273 to 0.225, that rendered hand-mesh conditioning (24.151 PSNR on synthetic data) clearly beats AdaLN pose injection (21.962 global, 22.789 per-frame) and binary hand masks (22.876), and that inpainting-based initialization converges to better DreamSim (0.088) than image-to-video initialization (0.103) after 4000 iterations.
- Qualitatively, DWM disentangles navigation from manipulation (without hand input it behaves as a pure navigator), accurately targets objects specified by hand trajectories, generalizes to unseen real-world scenes and interaction types such as opening a sliding window, and can convert generated human interaction videos into robot-arm videos for downstream robot data generation.

## Limitations

- The framework still relies on text prompts for the best visual quality, and distilling such semantic priors into purely action-based control remains future work.
- The model struggles with non-rigid or highly deformable objects and occasionally fails to maintain object rigidity and visual consistency during complex manipulations, which the authors attribute to limited deformable-interaction training data.
- The real-world dynamic-view data construction pipeline requires manual scene exploration and reconstruction, so it is scalable enough for evaluation but not yet for training.
- DWM does not explicitly reason about 3D structure, depth, or physical contact, restricting its ability to enforce strict physical constraints for applications such as robotic policy learning.
