# ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping

**Authors:** Youxin Pang, Ruizhi Shao, Jiajun Zhang, Hanzhang Tu, Yun Liu, Boyao Zhou, Hongwen Zhang, Yebin Liu  
**Date:** 2024-12-18  
**Identifier:** [arXiv:2412.16212](https://arxiv.org/abs/2412.16212)  
**Zotero item:** `YEWTRCJ6` ([Zotero](zotero://select/library/items/YEWTRCJ6))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

ManiVideo generates temporally coherent bimanual hand-object manipulation videos driven by MANO hand and object motion sequences, using a multi-layer occlusion (MLO) representation to learn 3D occlusion relationships and Objaverse 3D data to make object generation generalizable, and it outperforms HOGAN, Affordance Diffusion, and ControlNet baselines on DexYCB and a newly collected dataset.

## Background and Problem

The paper targets generation of consistent, temporally coherent bimanual hand-object manipulation videos from given motion sequences of hands (MANO parameters) and objects (3D models). Existing pose-driven HOI image generators condition on 2D signals such as depth maps, normal maps, topology maps, hand masks, or bounding boxes, but the authors identify two limitations: (1) 2D conditions only constrain visible regions, ignoring the plausibility of occluded parts — a serious problem in dynamic manipulation where finger self-occlusion, hand-object mutual occlusion, and invisible bent fingers are common; (2) training HOI datasets contain roughly 10 object categories, so models fail to generalize to unseen objects. Scarcity of HOI video data compounds both problems, making HOI video generation largely unexplored.

## Method

ManiVideo, built on the Animate Anyone architecture (AppearanceNet plus denoising UNet), introduces a multi-layer occlusion (MLO) representation with two components: (1) occlusion-free normal maps that render the object, palm, and each finger as separate layers from far to near (inspired by multi-plane images), compensating for hidden regions invisible to 2D signals; (2) occlusion confidence maps (depth-based) encoding how severely each part is occluded. The MLO structure is embedded into the UNet two ways: a lightweight pose guider (four conv layers) adds its features to the initial noise latent for spatial alignment, and a conv+MLP embedding is injected into added transformer blocks via cross-attention to capture 3D occlusion relationships. For object generalization, each Objaverse object provides appearance references from six viewpoints plus a background/human reference processed through AppearanceNet, and geometry (rendered normal maps plus a 2048x3 point cloud) is injected via cross-attention. Training proceeds in two stages — an image stage (~20,000 iterations) followed by a temporal stage (~30,000 iterations, adding temporal layers) — sampling equally from object-only, HOI video, and human datasets; for Objaverse, hand-related MLO layers are zeroed and only simulated object motions are used. Fine-tuning on human-centric data enables human-based HOI video generation with optional skeleton control.

## Contributions

1. The first framework supporting hand-object manipulation video generation with dexterous and generalizable grasping.
2. A multi-layer occlusion representation that learns articulated occlusion relationships from occlusion-free normal maps and occlusion confidence maps.
3. A training strategy integrating large-scale object datasets (Objaverse) with HOI video data to improve dynamic consistency and object generalization, additionally supporting human-centric HOI video generation.

## Experimental Setup

Training data: Objaverse (800K+ 3D models with simulated trajectories), DexYCB, a newly collected third-person bimanual dataset of 722 videos (376K frames, 15 objects, 10 views, 8 participants, daily tool use), and Human4DiT for human-centric fine-tuning; 3% of each dataset is held out for testing. Training used a single A800 GPU with Adam (learning rate 1e-5 image stage, 8e-6 temporal stage), batch size 4 then 1 with 24-frame sequences. Baselines: HOGAN (GAN-based, trained from scratch, evaluated only on DexYCB as it is single-handed), Affordance Diffusion (ContentNet with depth-map inputs), and ControlNet+Stable Diffusion v1.5 (CDiff). Metrics: FID, LPIPS, PSNR, SSIM (hand-object areas only), and MPJPE over 21 hand joints.

## Results

On DexYCB, ManiVideo achieves FID 49.96, LPIPS 0.079, PSNR 30.10, SSIM 0.913, and MPJPE 57.30, beating HOGAN (FID 64.74, MPJPE 60.95), ADiff (FID 53.95, MPJPE 59.12), and CDiff (FID 84.74, MPJPE 68.01). On the collected bimanual dataset it reaches FID 37.70, LPIPS 0.113, PSNR 29.59, SSIM 0.905, MPJPE 32.89 versus ADiff (FID 39.91, MPJPE 37.45) and CDiff (FID 45.50, MPJPE 42.89). Qualitatively, MLO handles finger self-occlusion, mutual occlusion, and invisible bent fingers where baselines produce wrong finger counts or artifacts. Ablations on the collected data show the full model's FID 37.70 degrades to 61.60 without Objaverse, 46.67 with MLO replaced by depth only, and 40.60 with MLO injected only into the initial noise. The model also generates unseen Objaverse objects and human-centric manipulation videos.

## Limitations

The paper states that performance is constrained by the accuracy of the driving signals, and that generalization to complex object textures is still limited by the domain gap between synthetic and realistic data; the authors suggest that incorporating object appearance into viewpoint and time via a more meticulous 4D representation could improve this.
