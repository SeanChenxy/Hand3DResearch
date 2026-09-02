# EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video

**Authors:** Ryan Hoque, Peide Huang, David J. Yoon, Mouli Sivapurapu, Jian Zhang  
**Date:** 2026 (ICLR 2026)  
**Identifier:** [arXiv:2505.11709](https://arxiv.org/abs/2505.11709)  
**Zotero item:** `6LDCN9MU` ([Zotero](zotero://select/library/items/6LDCN9MU))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

EgoDex is the largest dataset of dexterous human manipulation to date: 829 hours of 1080p, 30 FPS egocentric video with paired 3D skeletal annotations captured natively at recording time using Apple Vision Pro, totaling 90 million frames and 338,000 episodes across 194 tabletop tasks ranging from tying shoelaces to folding laundry. Unlike Ego4D or EPIC-KITCHENS, every frame carries precise 3D poses for the head, arms, wrists, and all 25 joints of each hand. The authors train and systematically evaluate 14 imitation-learning policies for hand trajectory prediction, introducing reproducible benchmarks with a best-of-K distance metric.

## Background and Motivation

Imitation learning for manipulation lacks an Internet-scale data corpus; teleoperation datasets are bottlenecked by robot hardware, and unstructured Internet video lacks precise annotations. Egocentric human video with paired 3D hand pose is a passively scalable middle path. Existing egocentric datasets do not focus on manipulation and have no native dexterous annotations, while existing HOI datasets with 3D hands are orders of magnitude smaller and emphasize grasping rather than long-horizon manipulation. The closest effort, EgoMimic, collects about 4 hours with wrist-only tracking, versus EgoDex's 829 hours with full upper-body and per-joint finger annotations.

## Dataset Construction

All data is collected with Apple Vision Pro running visionOS 2, using ARKit pose tracking and calibrated cameras so collectors record bare-handed with no external apparatus; sessions run 10-15 minutes with episode boundaries marked by pause/resume. Modalities per frame are egocentric RGB (1920 x 1080 at 30 Hz), camera intrinsics and extrinsics, 3D position and orientation of all upper-body joints including 25 joints per hand, per-joint confidence values, and language annotations (collector metadata fused by GPT-4 into a single detailed description). Tasks are organized as reversible pairs, reset-free, and reset types to maximize collection yield. Behavioral diversity goes far beyond pick-and-place (unscrewing bottle caps, flipping pages, slotting batteries, plus FurnitureBench assembly tasks); most verbs have more than 1,000 demonstrations, versus DROID where many verbs have fewer than 10.

## Evaluation Protocol

The action at each timestep is a 48-dimensional vector: per hand, 3D wrist position, 6D wrist orientation, and 3D positions of five fingertips, predicted as relative chunks in the current camera frame. Two benchmarks are defined: dexterous trajectory prediction (from images, skeletal history, and language, predict the next chunk) and inverse dynamics (additionally conditioned on a goal image). A fixed 1% held-out test set is sampled per task; evaluation uses a best-of-K metric — sample K trajectories and take the minimum Euclidean distance to ground truth, averaged over timesteps and the 12 wrist/fingertip keypoints. Policies from the X-IL framework combine encoder-decoder and decoder-only Transformers with behavior cloning, denoising diffusion, and flow matching, trained for 50,000 steps at batch size 2048 on 8 A100 GPUs.

## Findings and Analysis

With a 2-second horizon, encoder-decoder models consistently beat decoder-only ones; encoder-decoder flow matching is best at K=5 and K=10 (average distance down to 0.038 m, final 0.041 m, up to 34% better), while behavior cloning is about 15% better at K=1 (0.044 m), showing BC's average prediction is better but diffusion/flow capture modes better. Accuracy degrades with horizon (1 s: 0.031 m versus 3 s: 0.053 m average distance), and visual goal-conditioning cuts average distance 22% and final distance 53%. A 500M-parameter model matches the 200M default, and performance scales with dataset size on a log axis.

## Contributions

The largest and most diverse dexterous manipulation dataset (338K episodes, 194 tasks, 90M frames) with native 3D hand, finger, and upper-body tracking; structured task taxonomy with language and camera-pose annotations; and two reproducible trajectory-prediction benchmarks with best-of-K metrics and a 14-model baseline study covering architecture, policy representation, horizon, goal-conditioning, and data scaling.

## Limitations

Scene and background diversity is limited to tabletop environments, though behavior diversity is broad. The skeletal annotations are themselves model predictions and can be imperfect during heavy occlusion (e.g., towel folding) or very fast motions. The benchmarks evaluate hand trajectory prediction only, without object pose or contact supervision, and out-of-distribution tasks are treated separately in the appendix rather than in the main test set.
