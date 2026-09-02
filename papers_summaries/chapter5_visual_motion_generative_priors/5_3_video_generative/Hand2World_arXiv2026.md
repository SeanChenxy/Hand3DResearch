# Hand2World: Autoregressive Egocentric Interaction Generation via Free-Space Hand Gestures

**Authors:** Yuxi Wang, Wenqi Ouyang, Tianyi Wei, Yi Dong, Zhiqi Shen, Xingang Pan  
**Date:** 2026-02-10  
**Identifier:** [arXiv:2602.09600](https://arxiv.org/abs/2602.09600)  
**Zotero item:** `NXDGVN7C` ([Zotero](zotero://select/library/items/NXDGVN7C))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Hand2World synthesizes photorealistic egocentric hand-object interaction videos from a single scene image driven by free-space (mid-air) hand gestures captured from a monocular stream, with explicit camera control and arbitrary-length autoregressive generation for streaming deployment. It replaces 2D hand masks with an occlusion-invariant control signal built from projected 3D hand meshes, injects camera geometry through per-pixel Plücker-ray embeddings, and distills its bidirectional diffusion teacher into a causal generator, yielding a 76% reduction in FVD and a 42% reduction in camera trajectory error over state-of-the-art baselines on egocentric interaction benchmarks.

## Background and Problem

Interactive world models for augmented reality and embodied AI must respond to user input with low latency, geometric consistency, and long-term stability, and free-space hand gestures from head-mounted devices are one of the most natural control interfaces. The paper studies generating interaction videos in which hands enter a depicted scene, form plausible contacts, and induce object responses, while occlusion ordering stays correct and the background evolves coherently under head-induced ego-motion. Existing monocular mask-conditioned methods fail here for three reasons: training masks from contact-heavy data are partially occluded while free-space gestures produce complete masks (a distribution shift that hallucinates phantom occluders), viewpoint changes inferred from appearance alone cause background drift, and no causal end-to-end pipeline exists for online, user-driven interaction.

## Method

- Hand gestures are parameterized as MANO shape, pose, and translation, and the complete 3D hand meshes are projected to the image plane and rendered as a two-layer composite of filled silhouette plus wireframe overlay with color-coded hands; because the signal encodes intended geometry rather than visible pixels, visibility and occlusion are left for the generator to infer from scene context, eliminating the mask distribution shift.
- Camera motion is conditioned explicitly through per-pixel Plücker-ray embeddings computed from the estimated camera trajectory and injected additively by a lightweight adapter, decoupling viewpoint change from hand motion and preventing background drift.
- The backbone is Wan2.1-1.3B-Control, a flow-matching video diffusion transformer; the noisy video latent, hand-control latent, and first-frame-encoded scene latent are channel-concatenated, and training uses the rectified flow objective with a two-stage schedule (frozen-backbone camera-adapter pretraining, then joint LoRA fine-tuning).
- A fully automated monocular annotation pipeline extracts temporally stabilized hand meshes (YOLO detection, rule-based filtering and interpolation, HaMeR MANO estimation) and camera trajectories (Depth Anything V3 in streaming mode) from raw in-the-wild videos as pseudo-labels, avoiding multi-view rigs and manual annotation.
- The bidirectional teacher is distilled into a causal autoregressive generator following CausVid with self-forcing (ODE pretraining, distribution matching distillation, annealed student-history forcing), and infers block-wise with KV caching for arbitrary-length rollouts.

## Contributions

- The first monocular framework that synthesizes egocentric hand-object interaction videos from a single scene image under unconstrained free-space gestures, combining robust gesture control, explicit camera control, autoregressive arbitrary-length synthesis, generative scene dynamics, and monocular training and inference scalability in one system.
- Occlusion-invariant hand conditioning via silhouette-and-wireframe projections of complete 3D hand meshes, which keeps the control format consistent across occlusion states and delegates occlusion reasoning to scene context.
- Explicit camera modeling through Plücker-ray embeddings injected via a lightweight additive adapter, disentangling camera ego-motion from hand motion.
- A scalable training-and-deployment recipe: automated monocular annotation of hand geometry and camera pose plus autoregressive distillation validated across three egocentric benchmarks.

## Experimental Setup

- Evaluation uses 81-frame test clips from three egocentric datasets: ARCTIC (primary, bimanual manipulation of articulated objects with state changes), HOT3D (large-amplitude head motion from AR glasses), and HOI4D (800+ objects across 610 rooms).
- Baselines are CosHand, InterDyn, Mask2IV, and the Wan2.1-1.3B-Control backbone without the proposed conditioning; mask-based baselines receive SAM3-derived masks to isolate the effect of the occlusion-invariant representation.
- Metrics cover distributional realism (FVD), semantic alignment (DINO similarity), frame-level fidelity (PSNR, SSIM, LPIPS), temporal coherence (Flow-ERR via RAFT), and 3D/viewpoint consistency (Depth-ERR and Cam-ERR estimated with Depth Anything V3).
- Implementation: 10,000 adapter-pretraining steps then 100,000 joint LoRA (rank 256) fine-tuning steps on 8x A100-80GB GPUs, 41-frame 480p training clips, 384p for the AR student, and 50-step flow-matching sampling for the bidirectional teacher.

## Results

- On ARCTIC, Hand2World reduces FVD from 908.32 (best baseline InterDyn) to 218.76, raises DINO similarity from 0.80 to 0.88, and lowers Cam-ERR from the 0.12-0.14 baseline range to 0.07; the distilled Hand2World-AR stays close (FVD 232.40, DINO 0.88).
- On HOT3D, FVD drops from 349.89 (best baseline) to 106.20 while Cam-ERR falls from 0.33-0.38 across baselines to 0.13; on HOI4D, FVD reaches 251.05 with LPIPS 0.19, Cam-ERR 0.04, and Depth-ERR 7.98, all best among compared methods.
- Ablations identify the camera adapter as the dominant factor: removing it multiplies FVD by roughly 3.7x on ARCTIC (218.76 to 815.14), 4.1x on HOT3D, and 2.5x on HOI4D, while the wireframe overlay and annotation-time temporal stabilization contribute smaller complementary gains.
- The AR generator degrades gracefully with rollout length (ARCTIC FVD 232.40 at 81 frames, 264.20 at 162, 331.71 at 324) and runs at 8.9 FPS on a single A100 at 544x384 resolution, with hand reconstruction and camera estimation running in parallel at 25 FPS.

## Limitations

- Free-space gestures carry no physical contact constraints, so users can specify infeasible motions such as penetrating solid objects, which can produce implausible interactions; the authors suggest force-feedback devices as a remedy.
- Autoregressive quality degrades gradually as rollout length grows beyond the 81-frame training horizon, with accumulating errors in FVD, PSNR, and Cam-ERR at 162 and 324 frames.
- Camera conditioning depends on monocular pose estimation quality (Depth Anything V3), and the method is evaluated on benchmark clips rather than demonstrated in a fully closed-loop user study.
