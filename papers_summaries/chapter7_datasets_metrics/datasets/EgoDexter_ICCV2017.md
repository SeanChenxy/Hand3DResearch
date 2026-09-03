# Real-time Hand Tracking under Occlusion from an Egocentric RGB-D Sensor (EgoDexter)

**Authors:** Franziska Mueller, Dushyant Mehta, Oleksandr Sotnychenko, Srinath Sridhar (with Dan Casas, Christian Theobalt)
**Date:** 2017 (ICCV 2017)
**Identifier:** [arXiv:1704.02201](https://arxiv.org/abs/1704.02201)
**Zotero item:** No record found in the Zotero library; identity verified against arXiv metadata.
**Evidence status:** No Zotero record; verified against full-text PDF extraction (arXiv 1704.02201).

## Summary
This paper targets hand pose estimation from moving egocentric RGB-D cameras in cluttered real environments, a setting where existing methods fail because of object occlusions, background clutter, and first-person viewpoints typical of VR/AR. The authors present a two-stage CNN method (hand localization followed by 3D joint regression) refined by a kinematic pose tracking energy, and introduce two datasets to support it: SynthHands, a photorealistic synthetic training corpus of roughly 220,000 annotated RGB-D images built with a merged-reality capture strategy, and EgoDexter, a real annotated benchmark of 3190 frames of natural hand-object interaction in cluttered egocentric scenes (1485 manually annotated). Evaluation uses 2D pixel error and 3D Euclidean joint/fingertip error; the full method achieves an average fingertip error of 32.6 mm on EgoDexter while running in real time. Comparisons show that commercial systems (LeapMotion Orion) and third-person mid-air methods fail under the occlusions present in EgoDexter.

## Background and Motivation
Most prior hand pose estimation work addressed free hand motion in mid-air from third-person viewpoints in uncluttered scenes, where occlusions are rare. Egocentric scenarios — cameras on the head, shoulder, or chest — introduce object occlusions, cluttered backgrounds, manipulated objects, and field-of-view limitations, and this setting remained unsolved. CNN-based approaches require large amounts of annotated data, but markerless multi-view capture and manual annotation are infeasible at egocentric scale because of occlusions, cost, and time; even semi-automatic annotation fails when large parts of the hand are occluded. Prior synthetic datasets showed unnatural mid-air poses, no complex hand-object interactions, and no realistic background clutter or noise. The paper fills both gaps: a training data generation strategy that samples natural interaction, and a real benchmark for evaluation.

## Dataset Construction
The paper contributes two datasets.

**SynthHands (synthetic training data).**
- Source: real, non-occluded hand motion captured in mid-air from a third-person viewpoint with a real-time markerless tracker, retargeted onto an artist-rigged photorealistic hand model (Leap Motion hand model, Unity engine); merged-reality playback lets users perform grasping and manipulation motions against virtual on-screen objects.
- Scale: roughly 220,000 RGB-D images (63,530 frames of real hand motion sampled every 5th frame as poses).
- Variation: 12 skin tones (2 x 6 female/male textures), hand shape scaling beta in [0.8, 1.2] on male and female meshes, wrist rotation from a 70-degree range and arm rotation from a 180-degree range, 5 virtual egocentric viewpoints with simulated sensor noise and calibration, 7 virtual object shapes with 145 randomized object textures, and 10,000 real background images (desktops, offices, corridors, kitchens) composited via chroma keying.
- Annotations: accurate 3D joint positions by construction; no 3D scans of the evaluation objects are included, so the test objects are unseen.

**EgoDexter (real benchmark).**
- Source and sensor: real RGB-D recordings (Intel RealSense SR300, 640x480 at 30 Hz) in real cluttered environments with natural lighting and camera motion.
- Scale: 4 sequences (Rotunda, Desk, Kitchen, Fruits); 3190 frames in total, of which 1485 frames were manually annotated.
- Subjects: 4 users (2 female) with skin color and hand shape variation.
- Objects and actions: natural hand-object interactions with objects distinct from the SynthHands training objects (e.g., grabbing bottles, opening books, kitchen tasks).
- Annotations: 2D and 3D fingertip positions marked with a custom annotation tool, following common practice in free hand tracking; occluded fingertips are not annotated.
- A held-out synthetic test set of 5120 fully annotated SynthHands frames is also used for component evaluation.

## Evaluation Protocol
- Task: per-frame hand localization (2D root heatmap) and root-relative 3D regression of 21 joints from a single moving RGB-D camera, followed by kinematic skeleton fitting (26 DOF) for temporal smoothness; output is full 3D hand pose per frame.
- Metrics: 2D Euclidean pixel error for root localization; 3D Euclidean distance (mm) for all joints and, as a stricter measure, for the 5 fingertips.
- Evaluation sets: component and architecture ablations on the held-out synthetic test set; end-to-end evaluation on the real annotated EgoDexter sequences (average fingertip error per sequence).
- Baselines/comparisons: HALNet initialized with ground-truth versus predicted crops; a depth-only variant; a single combined CNN regressing global 3D pose directly; energy ablations (2D-only, 3D-only, combined); qualitative comparisons with LeapMotion Orion and with third-person mid-air methods (Sridhar et al.; Rogez et al. via reenacted sequences, since sensor differences preclude direct quantitative evaluation).

## Findings and Analysis
- On the synthetic test set, HALNet localizes the hand root with 2.2 px average error (standard deviation 1.5 px); initializing JORNet with predicted localization crops instead of ground-truth crops does not substantially degrade joint accuracy.
- On the real EgoDexter benchmark, the full formulation (combined 2D and 3D energy terms with kinematic tracking) yields the lowest average fingertip error of 32.6 mm; the 2D-only energy fails catastrophically on all sequences, and the 3D-only tracking term matches the raw 3D predictions.
- The two-step architecture (localize, then regress) outperforms a single CNN that directly regresses global 3D pose in cluttered, occluded scenes, and using RGB-D input outperforms depth-only input.
- Qualitatively, LeapMotion Orion fails under severe object occlusions, and the third-person mid-air method of Sridhar et al. produces catastrophic failures on EgoDexter; the authors report that quantitative comparison to the only other egocentric dataset (Rogez et al.) was not possible due to an unsupported sensor.
- Runtime is real-time: hand localization 11 ms, 3D joint regression 6 ms, kinematic tracking 1 ms on an Intel Xeon E5-2637 with an Nvidia Titan X (Pascal).

## Contributions
- A real-time two-CNN method for egocentric hand pose estimation in clutter and under strong occlusion, refined by a kinematic pose tracking energy.
- A photorealistic data generation framework (SynthHands) that uses merged reality to synthesize large amounts of fully annotated RGB-D data of natural hand-object interaction with diverse hands, viewpoints, objects, and real cluttered backgrounds.
- EgoDexter, a new annotated real benchmark with moving egocentric viewpoints, cluttered scenes, and hand-object interaction (3190 frames, 1485 annotated, 4 subjects, 4 sequences).

## Limitations
The authors report failure cases under fast motion that causes misalignment in the colored depth image or failures of the hand localization step, extreme self-occlusions, severe hand-object occlusions, and hands leaving the camera field of view. Training relied on synthetic data with sensor noise simulated for one specific camera, limiting generalization to other sensors; the authors suggest deep domain adaptation as future work. They were unable to quantitatively compare against the only prior egocentric dataset because of sensor incompatibility. Regarding the benchmark itself, annotations cover only fingertip positions (not full 21-joint hands or object pose), the sequences are few (4), and per-frame annotation excludes occluded fingertips, facts evident from the construction though not framed as limitations by the authors.
