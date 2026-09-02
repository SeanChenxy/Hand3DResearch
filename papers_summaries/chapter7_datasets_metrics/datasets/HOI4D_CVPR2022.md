# HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction

**Authors:** Yunze Liu, Yun Liu, Che Jiang, Kangbo Lyu, Weikang Wan, Hao Shen, Boqiang Liang, Zhoujie Fu, He Wang, Li Yi  
**Date:** CVPR 2022 (Zotero record dated 2024-01-03, corresponding to arXiv v4)  
**Identifier:** [arXiv:2203.01577](https://arxiv.org/abs/2203.01577)  
**Zotero item:** `TVPFM82J` ([Zotero](zotero://select/library/items/TVPFM82J))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HOI4D is a large-scale egocentric RGB-D dataset for category-level human-object interaction: 2.4M frames over 4,000 sequences (15 fps, 20 s each) of participants manipulating 800 object instances from 16 categories (7 rigid, 9 articulated) in 610 indoor scenes. It provides frame-wise 4D panoptic segmentation, motion segmentation, 3D hand pose (MANO), category-level object and part pose, fine-grained action labels, and reconstructed object meshes and scene point clouds. Three benchmarks are defined: category-level object/part pose tracking, semantic segmentation of 4D point cloud videos, and fine-grained egocentric action segmentation; state-of-the-art methods degrade sharply on all three.

## Background and Motivation

- Applications such as assistant robots and augmented reality require understanding interaction from 4D egocentric inputs (temporal streams of colored point clouds), unifying dynamic-scene semantics, 3D hand pose under object occlusion, pose and functionality of novel objects, and human action/intent. Existing egocentric datasets mostly offer 2D features without the 3D hand and object pose annotations needed for this.
- Prior HOI datasets are largely instance-level: objects come from a tiny pool whose exact CAD models and sizes are known beforehand (e.g., H2O covers only 8 instances), limiting transfer to the diversity of daily objects. Most also ignore articulated objects, and marker- or mocap-based capture biases natural motion and appearance.
- Synthetic data is an alternative, but simulating natural human motion and functional grasping for generic objects remains unsolved, limiting sim-to-real realism. Existing 4D point cloud segmentation work is dominated by outdoor driving data and does not transfer to cluttered, egocentric indoor scenes.

## Dataset Construction

- **Hardware and capture:** a head-mounted rig (bicycle helmet) carrying two precalibrated, synchronized RGB-D sensors: Kinect v2 (time-of-flight, better at long range) and Intel RealSense D455 (structured light, better at short range around 1 m), giving a natural testbed for cross-sensor transfer. Videos are 15 fps, 20 s long, totaling 2.4M frames over 4,000 sequences in 610 indoor rooms. The abstract states 4 participants while the introduction states 9 participants (both figures appear in the paper).
- **Objects and tasks:** 16 categories selected mainly from ShapeNet and SAPIEN Assets (linking HOI4D to synthetic 3D datasets), 7 rigid and 9 articulated, 50 unique instances per category (800 total), each with a CAD mesh reconstructed from multi-view high-resolution images (objects decorated with stickers to enrich texture and hide specular areas; RealityCapture / Agisoft Metashape used for reconstruction; PartNet-style part labels for articulated objects). 54 functionality-oriented tasks are defined across categories in the main text (the supplementary task table lists 76), spanning pick-and-place plus functionality-based actions such as opening a drawer, pouring water, cutting, and switching on a lamp. Tasks come in a simple level (pick-and-place, clean background) and a complex level (10-20 objects placed in clutter), supporting pose tracking/robot learning and 4D panoptic segmentation respectively.
- **4D panoptic segmentation:** moving content is separated from static content. 2D motion segmentation masks are manually annotated on 10% of frames and propagated to the rest with an interactive video-object-segmentation tool with human refinement; static content is reconstructed via SLAM, manually labeled in 3D, and projected back per frame; the two are merged into 4D dynamic panoptic labels.
- **3D hand pose:** MANO with per-capturer fixed shape beta and pose theta (45 joint coefficients plus global rotation and translation). Annotators label 2D positions of 11 of the 21 standard hand keypoints (wrist, 5 fingertips, second knuckles) on 20% of frames, estimating occluded keypoints; an optimization with joint-angle, 2D, depth, point-cloud, and mask losses initializes annotated frames; poses are propagated by linear interpolation and refined with additional contact and temporal-consistency losses over batches of 6-11 frames; failure frames are detected and corrected manually.
- **Category-level object and part pose:** annotators fit tight amodal oriented bounding boxes (physically measured per object) to objects or individual parts every 10 frames, giving 9D poses consistent across each category. Poses are propagated to intermediate frames by interpolation in the world frame, then optimized with the differentiable renderer SoftRas plus HOnnotate-style losses (silhouette, depth, chamfer against the scanned mesh point cloud, temporal consistency), with joint-angle limits for articulated objects.
- **Action annotation:** frame-wise fine-grained action category labels defined for interactive scenes (more fine-grained than existing action datasets).

## Evaluation Protocol

- **Split:** sequences are randomly divided 7:3 into training and test sets.
- **Category-level object and part pose tracking:** input is an RGB-D sequence with a perturbed ground-truth pose for initialization; evaluation follows prior category-level tracking protocols. Metrics: 5deg5cm (percentage of estimates with orientation error < 5 degrees and translation error < 5 cm), mean orientation error (degrees), and mean translation error (cm). Benchmarked on 4 rigid categories (toy car, mug, bottle, bowl) and 1 articulated category (laptop, with keyboard and display parts tracked separately). Baselines: BundleTrack (a model-free 6D tracker) and a point-to-plane ICP baseline (Open3D).
- **Semantic segmentation of 4D point cloud videos:** 376 videos spanning 14 semantic categories (7 object, 7 background); each frame is sampled to 4,096 points without color; metrics are mean IoU over object categories, background categories, and all. Baselines: PSTNet and P4Transformer, both state of the art on outdoor 4D segmentation.
- **Fine-grained action segmentation:** frame-wise accuracy, segmental edit distance, and segmental F1 at 10%, 25%, and 50% IoU thresholds, using I3D features (2048-d at 15 fps). Baselines: MS-TCN, MS-TCN++, and ASFormer.
- **Cross-dataset evaluation:** hand pose (Mesh Graphormer) vs. H2O; category-level tracking (CAPTRA, bottle category) vs. NOCS; action segmentation (ASFormer, 5 shared classes) vs. GTEA; each method is trained and tested on both datasets and their union.

## Findings and Analysis

- **Interaction makes category-level tracking much harder:** BundleTrack reaches 86.5 (5deg5cm) on the synthetic, interaction-free NOCS bottle category but only 19.3 on HOI4D bottles (ICP: 2.1). Across rigid categories, BundleTrack achieves 9.7-22.6 (5deg5cm) with 13.9-28.4 cm translation error. On the laptop, part-pose tracking reaches only 24.2 (keyboard) and 12.2 (display). Qualitative analysis attributes failures to hand occlusion and fast object motion during interaction, whereas camera ego-motion alone is handled.
- **Outdoor 4D segmentation fails indoors:** P4Transformer reaches 61.2 mIoU overall (objects 44.6, background 77.7) and PSTNet 52.0 (objects 31.4, background 72.6). Object categories are far worse than background due to small size, flexible movement, and severe egocentric occlusion.
- **Fine-grained actions are poorly perceived:** ASFormer attains 46.8 frame-wise accuracy on HOI4D versus 85.6 on 50Salads; MS-TCN 44.2 and MS-TCN++ 42.2 are similar. Failure analysis shows predicted action order is often correct while the current-action prediction is wrong, suggesting networks learn action sequencing rather than per-frame action perception.
- **Cross-dataset generalization favors HOI4D:** for hand pose, the HOI4D-trained model degrades 2.2x on H2O (22.3 to 48.9 mm root-relative MPJPE) while the H2O-trained model degrades 3.5x on HOI4D (19.9 to 70.4 mm); joint training helps H2O (15.9 mm) but slightly hurts HOI4D (24.3 mm). Similar patterns hold for CAPTRA tracking (NOCS-trained drops from 70.5 to 50.4 on HOI4D; HOI4D-trained drops from 55.3 to 34.2 on NOCS) and ASFormer action segmentation (GTEA-trained drops from 77.4 to 14.1 on HOI4D).
- **Downstream utility (supplementary):** converting HOI4D trajectories into demonstrations for a SAPIEN-based Adroit Hand "pick up" task raises success rate from 3.5% (SAC reinforcement learning) to 17.4% (GAIL imitation with 12 demonstrations), indicating value for robot imitation learning.

## Contributions

- The first large-scale 4D egocentric dataset for category-level HOI, covering 800 instances across 16 rigid and articulated categories with both instance-level meshes (enabling instance-level research and sim-to-real transfer) and category-level pose annotations.
- A data collection and annotation pipeline combining targeted human annotation (motion segmentation, keypoints, amodal boxes) with automatic propagation and optimization (mask propagation, SLAM, interpolation, differentiable-rendering pose optimization), scaling to 2.4M annotated frames.
- Three benchmarks for category-level HOI from 4D signals: object/part pose tracking, 4D point cloud semantic segmentation, and fine-grained action segmentation, with baselines showing large headroom.

## Limitations

Stated in the paper (Section 7) and derivable from its benchmarks:

- Two-handed manipulation is not covered; all tasks are single-hand, since single-hand manipulation was already considered challenging enough. Bimanual cooperation is deferred to future work.
- Reported baseline performance on HOI4D is far below that on prior synthetic or cleaner datasets (e.g., BundleTrack 19.3 vs. 86.5 5deg5cm on bottle; ASFormer 46.8 vs. 85.6 accuracy), showing the data is considerably harder than what current methods were designed for, and none of the tested methods comes close to solving the tasks.
- Hand pose annotation is derived from single-viewpoint egocentric observations with optimization over 20% annotated frames (with occluded keypoints estimated by annotators), and object pose labels rely on interpolation plus optimization every 10 annotated frames; the paper reports that failure frames from ambiguous poses or bad initialization must be detected and rectified manually.
