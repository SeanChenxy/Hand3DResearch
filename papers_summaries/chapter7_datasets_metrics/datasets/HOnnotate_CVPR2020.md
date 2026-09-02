# HOnnotate: A Method for 3D Annotation of Hand and Object Poses

**Authors:** Shreyas Hampali, Mahdi Rad, Markus Oberweger, Vincent Lepetit  
**Date:** 2019-07-02  
**Identifier:** [arXiv:1907.01481](https://arxiv.org/abs/1907.01481); DOI `10.1109/CVPR42600.2020.00326`  
**Zotero item:** `V9JIHJUS` ([Zotero](zotero://select/library/items/V9JIHJUS))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HOnnotate (CVPR 2020) is an optimization-based annotation method that labels RGB-D sequences of a hand manipulating an object with accurate 3D poses of both, even under heavy mutual occlusion, by jointly optimizing all frame poses of a sequence instead of tracking frame by frame. Applying it produced HO-3D, the first markerless dataset of real color images with 3D hand and object annotations (77,558 frames, 68 sequences, 10 subjects, 10 YCB objects), whose hand annotations reach 0.77 cm mean joint error (AUC 0.79). The paper also demonstrates the dataset's value with a single-RGB hand pose estimation baseline that beats direct MANO parameter prediction and transfers to unseen objects.

## Background and Motivation

Progress in single-image 3D hand and object pose estimation had not transferred to interacting hand-object pairs, mainly because of severe mutual occlusions and the lack of suitable real training data. Existing routes to annotated data all had drawbacks: automatic tracking-based annotation pipelines were noisy, marker- or sensor-based capture (for example the magnetic sensors in FPHA) alters hand appearance and biases learning, green-screen human-in-the-loop capture (FreiHAND) constrains the background and requires manual effort, and synthetic datasets such as ObMan cannot yet render realistic complex manipulation while remaining necessary for unbiased evaluation on real images. The authors instead cast annotation as a global energy minimization over the whole sequence, analogizing it to bundle adjustment in SLAM: temporal constraints are exploited globally, differentiable rendering allows gradient-based optimization, and multi-camera capture improves robustness while a single-camera mode keeps the setup practical for other laboratories.

## Dataset Construction

The method represents the hand with MANO (45 joint angles plus 6 wrist degrees of freedom and per-subject fixed shape parameters) and objects with 10 YCB-Video meshes. Sequences are captured by up to 5 synchronized RGB-D cameras (640x480, 5 ms synchronization). The cost function has data terms for silhouette discrepancy (against DeepLabv3 segmentations trained on synthetically composited hand-YCB images), Tukey-robust depth residuals, 2D hand joint heatmaps from a CPM-based network (trained on a 15,000-frame semi-automatically annotated seed dataset augmented with Panoptic Studio data), and an optional 3D point-cloud alignment term that accelerates convergence. Constraint terms enforce joint-angle limits defined directly on the 45 axis-angle parameters (MANO's PCA pose space proved insufficiently expressive for complex grasps), physical plausibility through a repulsion term that penalizes hand-object interpenetration, and temporal consistency via zeroth- and first-order motion models. Optimization proceeds in stages per setup: the multi-camera pipeline initializes, tracks single frames, then refines all poses jointly in batches of 20 frames (Adam, learning rate 0.01, 100 iterations); the single-camera pipeline additionally estimates an approximately constant grasp pose over the sequence before relaxing this assumption in a final multi-frame refinement.

## Evaluation Protocol

Annotation accuracy is measured by manually annotating 3D joint locations in randomly chosen frames on the consolidated point cloud from all five cameras and comparing against the automatic annotations; per-term contributions are isolated by re-running the optimization with subsets of energy terms enabled. The single-camera pipeline is validated by treating multi-camera annotations as reference and reporting average hand and object mesh differences at each stage. The learned hand pose baseline is trained on the 66,034-frame training split and evaluated on 13 test sequences (11,524 frames) containing unseen subjects and objects, reporting mean joint error (root- and scale-aligned), Procrustes-aligned mesh error, and F-scores at 5 mm and 15 mm.

## Findings and Analysis

With all terms enabled, the multi-camera pipeline achieves 0.77 cm mean hand-joint error and AUC 0.79, on par with FreiHAND's 0.791 despite larger objects causing stronger occlusion and the absence of a controlled background. Silhouette and depth terms individually fail but jointly provide better optima; the 3D term gives minor accuracy gains but speeds convergence; the physical plausibility term yields more natural grasps without improving numerical accuracy; and multi-frame multi-camera optimization improves accuracy by about 15% relative to single-frame optimization. The single-camera variant agrees with the multi-camera reference within 0.77 cm (hand) and 0.45 cm (object) average mesh error, with the final refinement adding 15% improvement. The hand pose baseline reaches 1.06 cm mesh error, 3.04 cm joint error, and F@5mm/F@15mm of 0.51/0.94 when it predicts joint directions in addition to 2D keypoints (1.14/3.14 and 0.49/0.93 without), clearly ahead of the directly regressing Hasson et al. model retrained on the same data (1.30 cm mesh error, 8.31 cm joint error), and it produces plausible poses on objects never seen in training.

## Contributions

- A joint multi-frame optimization method (HOnnotate) for automatic 3D annotation of hand and object poses in RGB-D sequences, combining segmentation, depth, 2D keypoint, and 3D alignment data terms with anatomical, physical, and temporal constraints.
- HO-3D, the first markerless color-image dataset with 3D annotations of both hand and object, enabling data-driven training and real-image evaluation for interacting hand-object pose estimation.
- Evidence that predicting 2D keypoints with joint directions and fitting MANO outperforms direct MANO parameter regression under occlusion.
- A single-camera annotation mode that lets other groups extend the dataset cheaply, supporting community-driven scaling.

## Limitations

Segmentation of the hand is a recognized weak point: the segmentation model was trained on synthetic data, and imperfect segmentations sometimes reduce annotation accuracy. The physical plausibility term improves realism but not measured accuracy, and the paper stops short of single-RGB joint hand-object pose estimation, listing it as future work. Finally, because the ground truth is itself algorithmically generated, its residual millimeter-level error (quantified at 0.77 cm mean joint error) is inherited by every downstream evaluation performed on the dataset.
