# Masquerade: Learning from In-the-wild Human Videos using Data-Editing

**Authors:** Marion Lepert, Jiaying Fang, Jeannette Bohg  
**Date:** 2025-08-13  
**Identifier:** [arXiv:2508.09976](https://arxiv.org/abs/2508.09976)  
**Zotero item:** `WU4Z3SKC` ([Zotero](zotero://select/library/items/WU4Z3SKC))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Masquerade edits in-the-wild egocentric human videos into "robotized" demonstrations to explicitly close the visual embodiment gap between humans and robots. Each Epic Kitchens clip is processed by estimating 3D hand poses, inpainting away the human arms, and overlaying a rendered bimanual robot that tracks the recovered end-effector trajectories. A ViT-Base vision encoder pre-trained to predict future 2D robot keypoints on 675K edited frames, then co-trained with a diffusion policy head on only 50 robot demonstrations per task, yields policies that outperform ImageNet, DINOv2, and HRP baselines by an average of 62 percentage points (12% to 74%) on three long-horizon bimanual kitchen tasks deployed zero-shot in unseen scenes.

## Background and Problem
Robot datasets remain orders of magnitude smaller and less diverse than the corpora behind language and vision breakthroughs, while in-the-wild human videos offer massive scale without action labels. Prior uses of human videos—visual pre-training, reward inference, world models—assume the model will implicitly bridge the embodiment gap between human hands and robot grippers, which is difficult for vision-based policies that are brittle under out-of-distribution appearance shifts. The paper asks whether explicitly closing that gap, even imperfectly, unlocks more signal: the task is to convert uncurated egocentric human videos into usable robot-training data and combine them with a small single-scene robot dataset to obtain policies that transfer to novel environments.

## Method
The pipeline extends Phantom's data-editing approach to in-the-wild footage. HaMeR estimates 21 keypoints per hand, which are mapped to a temporally smoothed robot end-effector pose (position, orientation, normalized gripper width); Detectron2 and SAM2 segment the arms, E2FGVI inpaints them away, and a virtual bimanual robot rendered with known camera intrinsics and extrinsics is composited into the original view. Supervisory labels are 2D waypoints projected onto the image plane over a horizon of H future frames, homography-warped into the first frame's view to compensate for egocentric camera motion. Frames with camera motion above 5 cm translation or 0.5 rad rotation per timestep, or invalid actions, are filtered out. A ViT-Base encoder conditioned on DistilBERT clip descriptions via FiLM is pre-trained with the 2D keypoint loss, then co-trained with a Diffusion Policy head on real demos (L = L2D + lambda Lpolicy, lambda = 10), with rendered robots also inpainted over the real robot footage. The hardware is two Kinova Gen3 7-DoF arms with Robotiq 2F-85 grippers and a rigidly mounted ZED mini camera.

## Contributions
- Extension of data-editing (pose estimation, inpainting, robot overlay) from curated single-hand videos to large-scale in-the-wild egocentric human videos.
- A pre-training plus co-training recipe that preserves the human-video keypoint objective during imitation fine-tuning, which the authors show is essential for out-of-distribution robustness.
- Evidence that even imperfect 2D overlays yield large cross-embodiment gains, with logarithmic-style scaling in the amount of edited video.

## Experimental Setup
Training uses 10K Epic Kitchens clips (675,713 frames) and 50 bimanual demos per task collected with an Oculus headset in a single scene. Three long-horizon tasks are evaluated—Stack Pots, Scrape Potato, Sweep Chilis, each scored over three subtasks—in three out-of-distribution scenes with 10 rollouts per scene (30 per task). All methods share the ViT-Base architecture; baselines are ImageNet initialization, DINOv2, and HRP affordance pre-training.

## Results
Masquerade beats all baselines in every out-of-distribution scene, averaging 74% success versus roughly 12% for baselines (5-6x). Removing robot overlays or removing co-training causes steep performance drops, showing both components are indispensable. Scaling co-training data on Stack Pots raises success monotonically: 2% at 0%, 26% at 10%, 47% at 50%, and 68% at 100% of the edited corpus. Masquerade also shows the smallest drop between in-distribution and out-of-distribution scenes on Sweep Chilis, where baselines degrade sharply.

## Limitations
The authors note dependence on hand-pose estimators that fail under fast motion or heavy occlusion, forcing frame discards; missing depth prevents occlusion-correct overlay compositing, so robot pixels can wrongly appear over scene objects; egocentric camera motion requires filtering for a stationary robot; and retargeting dexterous human grasps onto parallel-jaw grippers is imperfect.
