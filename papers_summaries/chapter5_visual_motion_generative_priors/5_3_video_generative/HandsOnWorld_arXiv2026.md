# HandsOnWorld: Unconstrained Egocentric Video Generation with Camera-Disentangled Hand Control

**Authors:** Yushuo Chen, Xiaoyu Shi, Xiaoshi Wu, Xintao Wang, Pengfei Wan, Yebin Liu  
**Date:** 2026-07-02  
**Identifier:** [arXiv:2607.02075](https://arxiv.org/abs/2607.02075)  
**Zotero item:** `ER33I2YG` ([Zotero](zotero://select/library/items/ER33I2YG))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HandsOnWorld trains hand-controlled egocentric video generation directly on unconstrained monocular video instead of multi-view or marker-based motion capture, via two components: EgoVid-Pro, a protagonist-centered annotation pipeline that distills clean 3D hand trajectories from in-the-wild footage into a 103K-clip, roughly 12M-frame dataset spanning diverse everyday scenes, and the Pluecker Hand Map, a world-frame control signal that extends Pluecker rays from camera geometry to the hand surface. By disentangling hand motion from camera ego-motion at the representation level, the method surpasses camera-space baselines on every metric under substantial ego-motion and generalizes to out-of-distribution lab datasets and fully in-the-wild generations.

## Background and Problem

Hands are the primary interface for manipulating generated worlds, but hand-controlled egocentric generators have been confined by a data annotation pyramid in which hand-supervision fidelity is inversely coupled to scene diversity: in-the-wild corpora offer only coarse body pose, fixed multi-camera sites cover staged activities, and instrumented headsets yield precise but tabletop-only captures. Moving to unconstrained monocular video raises two challenges absent from lab data: protagonist hand identification, since off-the-shelf reconstruction returns bystander hands, hand-like false positives, and unstable detections; and camera-hand motion entanglement, because camera-space control signals such as projected 2D joints or rendered mesh images encode only camera-relative pose, so identical signals can correspond to very different absolute 3D hand motions once substantial head ego-motion is present.

## Method

- The protagonist-centered annotation pipeline processes EgoVid-5M (a curated Ego4D subset with captions) through three progressive filters: semantic filtering against a 16-verb vocabulary of concrete manipulation actions (removing roughly 90% of clips), image-level filtering that tightens HaWoR detection confidence from 0.2 to 0.4 and keeps clips with at least 80 valid detections of 120 frames, and 3D-geometry filtering that jointly fits a single SMPL body - head anchored to the egocentric camera, wrists pulled toward detected hand anchors, pose regularized by VPoser - rejecting any tracklet the first-person body cannot reach.
- The resulting EgoVid-Pro comprises 103,032 clips of 120 frames (about 12M annotated frames) with protagonist-only single-hand and bimanual trajectories, comparable in scale to the largest existing 3D-hand-annotated egocentric dataset (Ego-Exo4D) but spanning far more diverse everyday scenes; a clean subset of 34,078 videos with complete bimanual annotations over the first 81 frames is used for training.
- The Pluecker Hand Map rasterizes the posed MANO mesh with nvdiffrast and assigns each hand-covered pixel a 6D surface-normal ray (outward normal plus position-cross-normal moment) in the world frame, concatenated with the per-pixel camera Pluecker ray into a 12-channel map; a formal analysis shows the hand ray is invariant to pure camera translation yet shifts by the hand's displacement under hand motion, disentangling the two sources at the input level.
- The 12-channel map is encoded by a 4-layer convolutional encoder and added as a residual to the noisy video latent at each denoising step of Wan2.2-I2V (5B and 14B DiT backbones), fine-tuned with LoRA (1,000 iterations, AdamW, batch size 16, 480x640) and dual classifier-free guidance over text and geometry; causal-forcing distillation of the 5B backbone yields a 9.64 FPS autoregressive variant for long-video rollout.

## Contributions

- An unconstrained egocentric hand-controlled video generation framework that removes reliance on multi-view or marker-based capture, enabling training on everyday monocular video and generalization to scenes, objects, and imaginary worlds far beyond tabletop settings.
- EgoVid-Pro, the first large-scale egocentric dataset of clean, protagonist-only 3D hand trajectories curated from in-the-wild video, which avoids the scale-versus-label-quality tradeoff that raw unfiltered footage imposes.
- The Pluecker Hand Map, a unified world-space control signal pairing camera rays with hand surface-normal rays that disentangles absolute 3D hand motion from camera ego-motion at the representation level, resolving the ambiguity of camera-space signals under ego-motion.

## Experimental Setup

- Training-data conditions compare the pretrained Wan2.2-I2V-14B zero-control reference, fine-tuning on ARCTIC alone, ARCTIC plus raw EgoVid clips, and fine-tuning on EgoVid-Pro, evaluated on 300 held-out EgoVid-Pro validation videos (three non-overlapping 81-frame clips each) and on ARCTIC's standard 267/34 split center-cropped to 800x600.
- Control-signal comparisons reimplement Hand2World and Generated Reality, adapt FMC's 6D object-pose encoding to 16 hand joints (FMC*), and augment the static-camera JointControl with a camera-control branch (JointControl*), on both ARCTIC and EgoVid-Pro; an out-of-distribution test evaluates all methods trained on the same EgoVid-Pro data directly on H2O with independent multi-view annotations.
- Metrics cover visual quality (PSNR, SSIM, LPIPS, FVD, VBench subject and background consistency), camera pose (RotErr, TransErr from HaWoR-reconstructed trajectories), and hand pose (2D/3D L2Err, PA-JPE, detection recall, plus world-space W-JPE, WA-JPE, and RTE for absolute 3D trajectory accuracy).
- Additional ablations compare protagonist-filtering strategies (camera-outward and MCP-up heuristics, EgoSim) and world-space signal variants (position, depth, and normal maps).

## Results

- On EgoVid-Pro, HandsOnWorld beats all baselines on every metric: PSNR 17.42, SSIM 0.5367, LPIPS 0.2923, FVD 274.51, RotErr 3.75, TransErr 3.33, L2Err 11.32, PA-JPE 6.37, and world-space hand errors of W-JPE 67.46, WA-JPE 33.02, and RTE 4.12, versus for example L2Err 21.84 for Hand2World and 44.26 for EgoSim-style annotation.
- On tabletop ARCTIC the method merely matches the strongest baselines (FVD 174.23 versus 166.97, L2Err 14.36 versus 15.20), which the authors explain as expected: with the camera frame nearly aligned to the world frame, camera-space and world-space encodings carry equivalent information, so the representation only diverges under substantial ego-motion - precisely the regime EgoVid-Pro provides.
- The out-of-distribution H2O evaluation shows large margins despite domain shift (PSNR 17.98, LPIPS 0.2270, L2Err 9.98, W-JPE 59.51 versus 84.22-139.16 for baselines), and data-condition comparisons reveal that ARCTIC-trained baselines leak the lab distribution (mocap markers on hands, tone shift) and confuse objects, while raw unfiltered EgoVid pretraining improves visual quality but degrades control accuracy.
- The distillation study reports 0.13 FPS for the 14B model, 2.39 FPS for the 5B model, and 9.64 FPS for the distilled 5B autoregressive variant with modest quality loss (PSNR 16.31 versus 16.69), and signal ablations confirm the Pluecker encoding outperforms position, depth, and normal map alternatives on all five hand metrics.

## Limitations

- Because the control signal rasterizes only the visible hand surface, self-occluded hand regions carry no explicit pose information, and the authors suggest layered representations such as multi-plane images as a possible remedy.
- The annotation pipeline depends on an off-the-shelf hand detector, and when a frame has no detection the system cannot distinguish whether the hand left the image or was simply missed; such ambiguous gaps are currently discarded rather than supervised, with mask-based training or uncertainty-aware conditioning proposed as future work.
- The paper's reported inference quality comes from the bidirectional diffusion model, and the faster distilled autoregressive variant trades a measurable drop in PSNR and FVD for its 9.64 FPS throughput.
