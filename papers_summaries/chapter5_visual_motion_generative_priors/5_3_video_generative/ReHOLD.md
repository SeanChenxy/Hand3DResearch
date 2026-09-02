# Re-HOLD: Video Hand Object Interaction Reenactment via adaptive Layout-instructed Diffusion Model

**Authors:** Yingying Fan, Quanwei Yang, Kaisiyuan Wang, Hang Zhou, Yingying Li, Haocheng Feng, Errui Ding, Yu Wu, Jingdong Wang  
**Date:** 2025-03-21  
**Identifier:** [arXiv:2503.16942](https://arxiv.org/abs/2503.16942)  
**Zotero item:** `AHDEZUCP` ([Zotero](zotero://select/library/items/AHDEZUCP))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Re-HOLD is a CVPR 2025 human-centric video reenactment framework that synthesizes hand-object interaction (HOI) videos from a driving motion sequence and a target object image, using specialized hand/object layout boxes for disentanglement, a restoration module with two texture memory banks, and an adaptive layout-adjustment strategy that keeps interactions physically plausible when the target object differs strongly in size or shape.

## Background and Problem

Digital-human research focused on lip-sync and body movement does not cover interaction with real-world objects, and HOI video synthesis remains sparse despite its industrial demand. The paper defines HOI reenactment: given sequential motion signals (layout guidance and reconstructed 3D hand meshes) and a target object image, generate a video reconstructing the source interaction, or — in cross-object reenactment — transfer the interaction to a different object. Three challenges motivate the design: (1) hand-object occlusion entanglement causes artifacts at the interface; (2) hands and objects have high degrees of freedom yet occupy few pixels, making detailed texture recovery hard; (3) shape and size differences between objects degrade interaction realism if the source motion is kept unchanged. Prior work HOI-Swap produces object-centric single-hand swapping and ignores hand/object size-position changes.

## Method

Re-HOLD follows a two-branch Stable Diffusion v1.5 architecture: a Reference U-Net encodes object texture from the reference image, and a Denoising U-Net predicts noise from the latent plus layout guidance, with temporal attention layers and cross-branch attention. The layout representation uses three bounding boxes: two fixed-size square hand boxes (detected via DWPose keypoints, pose- and size-invariant to disentangle hand position from mesh-driven motion) and one variable-size object box (from LISA segmentation), encoded by a lightweight 4-conv layout encoder with zero-conv projection added to the noisy latent. The HOI Restoration Module reshapes structure and refines texture: HaMeR-reconstructed 3D hand meshes (with random positional augmentation during training to reduce over-reliance on hand positions) are encoded by a ControlNet, while two independent learnable memory banks (hand and object, 512 entries each) restore diverse hand poses and fine-grained object textures through Hand-Attention and Object-Attention layers masked to the respective regions. For cross-object reenactment, an adaptive layout-adjustment strategy runs at inference in four steps: identify hand-object contact via H2O distance against a threshold, resize the object box to the target object's adaptive ratio around its center, shift hand boxes horizontally to preserve the original H2O distance, and align the object box bottom — preventing floating objects and unreasonable contact.

## Contributions

1. The first HOI reenactment framework for human-centric video generation achieving realistic and reasonable HOI synthesis.
2. Specialized hand and object layout representations plus an HOI Restoration Module enabling disentanglement and improved HOI modeling.
3. A layout-adjustment strategy compatible with objects differing significantly in shape and size, generating reasonable interactions.

## Experimental Setup

Training uses a collected dataset of 9 subjects with 14 objects (5-second clips; two unseen objects per subject held out for testing) plus single-object HOI4D videos. Data are processed at 25 FPS, 512x512; DWPose provides hand boxes and crops, HaMeR the MANO meshes, LISA the object masks. Training on 4 A800s (learning rate 1e-5) has two stages: image-level HOI modeling (batch 48, 100K steps, about 3 days, with L1 loss focused on hand/object regions every 10 iterations near the end) and temporal modeling (24 frames, batch 1, 50K steps, about 2 days, temporal layers only). Inference uses DDIM with 30 steps. Baselines: AnyV2V, VideoSwap, AnimateAnyone, RealisDance, HOI-Swap. Metrics: hand fidelity, subject consistency, motion smoothness; hand agreement, PSNR, FID for self-reenactment only; contact agreement was excluded due to incorrect object detection.

## Results

On the collected dataset, Re-HOLD reaches FID 19.021, PSNR 33.451, hand agreement 0.773, hand fidelity 0.993, subject consistency 0.953, and motion smoothness 0.995 in self-reenactment — best FID/PSNR among all baselines (AnimateAnyone: FID 26.361; RealisDance: 26.337; HOI-Swap: 30.932) — and best cross-reenactment subject consistency (0.955) with hand fidelity 0.994. On HOI4D self-reenactment it attains FID 26.583 and hand agreement 0.826 versus HOI-Swap's 30.152 and 0.754. A 15-voter user study over 20 samples rates Re-HOLD highest: HOI consistency 0.92, object consistency 0.92, temporal consistency 0.88 (HOI-Swap: 0.76, 0.40, 0.44). Ablations confirm each component: removing layout guidance drops self-reenactment hand agreement to 0.753, and removing hand/object attention or the adaptive strategy also degrades results.

## Limitations

The paper states that its dataset was designed to capture fundamental hand movements for object display in live-streaming scenarios, so the framework produces less satisfactory results for 3D object manipulation cases, such as generating a multi-view video of an object, which the authors leave as future work.
