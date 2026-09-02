# You Only Teach Once: Learn One-Shot Bimanual Robotic Manipulation from Video Demonstrations

**Authors:** Huayi Zhou, Ruixiang Wang, Yunxin Tai, Yueci Deng, Guiliang Liu, Kui Jia  
**Date:** 2025-04-27 (RSS 2025)  
**Identifier:** [arXiv:2501.14208](https://arxiv.org/abs/2501.14208)  
**Zotero item:** `JUL46LHA` ([Zotero](zotero://select/library/items/JUL46LHA))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
YOTO (You Only Teach Once) teaches dual robot arms long-horizon bimanual manipulation from a single third-person binocular video of a human demonstration. Vision models extract fine-grained hand motion, which is simplified into discrete keyframes plus a motion mask encoding left-right coordination, then injected into the robots and proliferated into large training sets via real-robot auto-rollout and point-cloud geometric augmentation. A customized bimanual diffusion policy (BiDP) trained on this data achieves a 76.8% average success rate across five complex tasks, versus 23.4% for the strongest baseline, and shows the best out-of-distribution generalization.

## Background and Problem
Bimanual manipulation is difficult because the two arms must coordinate alternately or synchronously while avoiding collisions, and because two arms plus end effectors span a higher-dimensional action space. Existing approaches—predefined task taxonomies and teleoperation-based imitation—lack simplicity, versatility, and scalability; teleoperated demonstrations are also non-stationary and despatialized. The authors instead leverage human demonstration videos, where spatial-temporal positions, dynamic postures, interaction states, and dexterous transitions are available almost for free. The task is to extract bimanual action patterns from as few as one binocular observation and convert them into robust visuomotor policies.

## Method
A human demonstrates the task on the dual-arm workbench (two contralateral Aubo i5 arms, parallel-jaw grippers, DexSense binocular stereo camera). WiLoR detects hands and estimates MANO 3D shapes; rather than trusting monocular 3D trajectories directly, YOTO projects hand center points onto the 2D image and lifts them back to 3D via stereo matching for camera-space stability. Gripper open/close is inferred from hand-object contact detection, and a gripper orientation is constructed from index-wrist and ring-wrist vectors. Continuous trajectories are simplified into about 10 keyframes per task (K much smaller than the 100-200 frames), verified and corrected on the real robot in about three minutes, and paired with a binary motion mask recording which arm moves at each keyframe. Demonstration proliferation combines real-world auto-rollout—editing keyposes to shift or replace objects, collecting roughly 300 demonstrations in 8 hours—and geometric transformations of segmented object point clouds (Florence-2 detection, SAM2 segmentation, stereo matching), expanding data about 100 times to 5K-24K trajectories per task. BiDP consumes only manipulated-object point clouds (1024 points) with a SIM(3)-equivariant PointNet++ encoder and FiLM-conditioned convolutional U-Net denoiser (DDIM, 8 steps), predicts all keyposes from the initial one-shot observation, and reorganizes the bimanual action space into a time-ordered single-arm sequence using the motion mask.

## Contributions
- A paradigm for extracting and injecting dual-arm movements from one-shot human hand demonstrations.
- A rapid demonstration proliferation solution combining auto-rollout verification with point-cloud geometric transformation, cheaper and more reliable than teleoperation.
- BiDP, a bimanual diffusion policy with object-only observations, keypose prediction, and motion-mask-based action space reorganization.

## Experimental Setup
Five real-world tasks are evaluated: pull drawer, pour water, and unscrew bottle (strictly asynchronous, 10-12 keyframes each) plus uncover lid and open box (synchronous, 12-16 keyframes), with 36-243 auto-rollout demonstrations per task and multiple everyday objects per category. Baselines are ACT, Diffusion Policy, DP3, and EquiBot, trained identically and evaluated with randomized object placements; success is scored substep by substep with the average completed length. OOD tests use held-out unseen objects on pull drawer and uncover lid.

## Results
BiDP averages 76.8% success over the five tasks versus 23.4% for EquiBot, 19.4% for DP3, 15.8% for DP, and 5.7% for ACT, finishing 43/54 on pull drawer, 28/36 on pour water, 23/30 on unscrew bottle, 20/25 on uncover lid, and 14/20 on open box. Ablations show each component helps: object-only observations and sparse keyframes raise the pull-drawer rate from 24.1% to 57.4%, motion-mask action reorganization to 61.1%, and geometric augmentation to 79.6%, with performance rising monotonically as expansion grows from none to 500 times. Under OOD object shifts, BiDP retains 35.0% average success versus 8.8% for EquiBot and 0% for the other baselines.

## Limitations
The authors note that vision-based hand trajectory extraction has inherent errors requiring careful human verification on real robots; the fixed workbench limits flexibility; parallel grippers restrict functionality compared with dexterous hands or tactile sensing; and ultra-difficult tasks such as tool-based manipulation, highly dynamic actions, and human-robot collaboration remain unexplored.
