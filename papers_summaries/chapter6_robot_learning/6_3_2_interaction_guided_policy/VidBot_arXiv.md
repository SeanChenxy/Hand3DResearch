# VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation

**Authors:** Hanzhi Chen, Boyang Sun, Anran Zhang, Marc Pollefeys, Stefan Leutenegger  
**Date:** 2025 (CVPR; the PDF is the CVF open-access version, no explicit date recorded in Zotero)  
**Identifier:** no identifier recorded in Zotero metadata or PDF text  
**Zotero item:** `CZ7IBBLR` ([Zotero](zotero://select/library/items/CZ7IBBLR))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Teleoperated robot demonstrations do not scale, so VidBot learns manipulation from in-the-wild monocular RGB-only human videos. A reconstruction pipeline combining SfM with a metric-depth foundation model extracts temporally consistent, metric-scale 3D hand trajectories and contact points — embodiment-agnostic 3D affordances — from everyday videos. A coarse-to-fine affordance model first infers contact and goal points from RGB-D observations and language instructions, then generates fine-grained 3D interaction trajectories with a diffusion model guided at test time by differentiable cost functions. Without robot data for training, VidBot reaches an 88.2% average success rate across 13 simulated household tasks, roughly 20% above the runner-up, and 80.0% success over 55 real-robot trials.

## Background and Problem
Imitation-learning approaches depend on costly teleoperated data, and scaling robot datasets such as Open X-Embodiment and DROID remains hard due to the combinatorial explosion of embodiments, tasks, and environments. Prior human-to-robot methods require static cameras, depth sensors, or MoCap, while web-video approaches learn only visual representations or 2D pixel-plane motion cues. The paper argues that 3D affordance — contact points plus interaction trajectories with spatial awareness — is the key embodiment-agnostic interface. The task input is an RGB-D frame (depth from a sensor or a metric-depth foundation model) and a language instruction; the output is contact points and a 3D interaction trajectory in the observation camera's frame, executable zero-shot on novel robots in novel scenes.

## Method
Data extraction runs on raw human videos: SfM provides camera intrinsics, scale-unaware poses, and sparse landmarks; a metric-depth foundation model predicts dense depth; hand/object detection and segmentation yield masks, and video inpainting produces hand-less frames. Consistent pose optimization fits a global metric scale by projecting landmarks onto predicted depth, then refines per-frame poses and scales for cross-view consistency. Hand centers across frames yield interaction trajectories, and downsampled hand points in the first and last frames give contact and goal points (demonstrated on Epic-Kitchens-100 with EpicFields SfM). The affordance model factorizes into a coarse stage — goal and contact predictors outputting pixel-space probability heatmaps, fused with CLIP language and bounding-box features via Perceiver modules — and a fine stage, a 1D U-Net diffusion model that directly predicts the unnoised trajectory conditioned on coarse points, language, object features, and TSDF spatial features from a 3D U-Net. Test-time guidance injects gradients of differentiable multi-goal, collision-avoidance, and contact-normal costs into each denoising step, and the final cost value ranks candidate trajectories.

## Contributions
- A gradient-based pipeline extracting metric-scale 3D hand trajectories and contact points from in-the-wild RGB-only human videos.
- A coarse-to-fine affordance model predicting 3D contact/goal points and diffusion-generated interaction trajectories conditioned on language and scene context.
- Test-time differentiable cost guidance adapting trajectories to novel scenes and robot morphologies without retraining.

## Experimental Setup
Simulation uses IsaacGym with 13 everyday household tasks from FrankaKitchen, PartManip, and ManiSkill, each evaluated from three viewpoints with five trajectories per viewpoint (15 trials per task); success requires exceeding the object's degree-of-freedom threshold without collisions. Baselines include GAPartNet and Where2Act (simulator-trained), Octo (fine-tuned on the extracted affordance data), VRB (same human-video source, 2D affordance lifted to 3D), and GFlow (given ground-truth depth and poses). Ablations test six variants on six tasks, plus visual goal-reaching and exploration applications. Real deployment uses Hello Robot Stretch 3 and Boston Dynamics Spot.

## Results
VidBot attains an 88.2% average success rate over the 13 tasks versus 69.2% for Octo, 61.0% for GFlow, 59.0% for VRB, 58.5% for Where2Act, and 51.1% for GAPartNet, reporting roughly a 30% improvement over VRB despite identical human-video supervision. Ablations show the full model at 85.6% on six tasks: removing coarse goal prediction drops it to 57.8%, multi-goal guidance to 73.3%, contact-normal guidance to 76.7%, collision avoidance to 77.8% (a 26.7% drop on picking up cans), and cost-informed plan selection to 74.5%. The model also converges faster and higher than human-video baselines on goal-reaching and exploration, and real deployment succeeds in 80.0% of 55 trials across three environments.

## Limitations
The authors state that data quality is constrained by the accuracy of the depth foundation model and the SfM pipeline, despite filtering low-quality labels via the final optimization loss, and that precise tasks such as unscrewing caps remain challenging; they propose multimodal affordance extraction from wearable devices as future work.
