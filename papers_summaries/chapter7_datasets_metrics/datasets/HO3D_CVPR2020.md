# HOnnotate: A Method for 3D Annotation of Hand and Object Poses

**Authors:** Shreyas Hampali, Mahdi Rad, Markus Oberweger, Vincent Lepetit  
**Date:** 2019-07-02  
**Identifier:** [arXiv:1907.01481](https://arxiv.org/abs/1907.01481); DOI `10.1109/CVPR42600.2020.00326`  
**Zotero item:** `V9JIHJUS` ([Zotero](zotero://select/library/items/V9JIHJUS))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

This CVPR 2020 paper introduces HOnnotate, a fully automatic annotation method that recovers the 3D poses of a hand and a manipulated object jointly over entire RGB-D sequences, and uses it to build HO-3D, the first markerless dataset of real color images with 3D annotations for both the hand and the object. The dataset comprises 77,558 frames from 68 sequences with 10 subjects manipulating 10 YCB objects, and an annotation accuracy of 0.77 cm mean hand-joint error (AUC 0.79) validated against manual annotations. The authors also train a single-RGB-image hand pose predictor on HO-3D that outperforms direct MANO parameter regression under severe hand-object occlusion and generalizes to objects never seen during training.

## Background and Motivation

Data-driven estimation of 3D hand and object poses from a single image was blocked by the absence of real, annotated hand-object interaction data. Annotating real images is difficult because the hand and the object mutually occlude each other, and earlier annotation pipelines were either noisy tracking-based systems, required visible instruments (for example, the magnetic sensors of the FPHA dataset that bias the hand's appearance), or depended on green screens and human-in-the-loop refinement as in FreiHAND. Synthetic alternatives such as ObMan provide perfect poses but suffer from a domain gap and cannot simulate complex manipulation realistically, and evaluation on real images remains necessary. The authors frame their solution as the equivalent of bundle adjustment for hand-object capture: rather than tracking poses frame by frame with drift and manual initialization, they jointly optimize all hand and object poses over a whole sequence, exploiting temporal consistency in a stronger way and using differentiable rendering so that modern gradient-based optimizers can minimize a complex objective.

## Dataset Construction

HO-3D sequences are captured with one or several synchronized RGB-D cameras; the multi-camera rig uses 5 cameras at 640x480 pixels with 5 ms synchronization accuracy. Hands are represented with the MANO model (45 joint angles plus 6 wrist degrees of freedom, with per-subject shape parameters estimated separately), and the manipulated objects are 10 YCB objects chosen for their shape and material variety and commercial availability. Annotations are produced by minimizing an energy combining data terms and constraints: a silhouette discrepancy term against segmentations predicted by DeepLabv3 (trained on synthetic composites of hands and YCB objects), a Tukey-robust depth residual term, a 2D joint error term against heatmap predictions from a Convolutional Pose Machine, an optional 3D point-cloud distance term that mainly accelerates convergence, a joint-angle limit term applied directly to the 45 axis-angle parameters (deliberately avoiding MANO PCA pose space, which could not express some grasps), a physical plausibility repulsion term preventing interpenetration, and a temporal consistency term with zeroth- and first-order motion models. Optimization is multi-stage: automatic initialization, single-frame tracking, and a final multi-frame joint refinement over all poses simultaneously, run in batches of 20 frames with Adam (learning rate 0.01, 100 iterations). A single-camera variant additionally exploits the assumption that the grasp pose changes little across a sequence, first estimating a constant grasp pose and object trajectory before a relaxed joint refinement, which makes the pipeline easy to extend by other groups.

## Evaluation Protocol

Annotation quality is validated by manually marking 3D hand joint locations in randomly selected frames using the consolidated point cloud built from all five depth cameras, and comparing these to the automatic estimates. Hand pose prediction is evaluated on a test set of 13 sequences (11,524 frames) that contain subjects and objects absent from the 66,034-frame training split, using three standard metrics: mean joint error after root and scale alignment, mesh vertex error after Procrustes alignment, and the F-score at 5 mm and 15 mm thresholds.

## Findings and Analysis

The multi-camera annotation pipeline reaches 0.77 cm mean hand-joint error and an AUC of 0.79, comparable to the FreiHAND annotation accuracy (AUC 0.791) despite larger objects, heavier occlusion, and no green screen. The ablations show that silhouette and depth terms must be combined to escape local minima, the 3D distance term mainly speeds convergence, and the physical plausibility term improves grasp naturalness without improving accuracy, while multi-frame multi-camera optimization improves accuracy by roughly 15% over single-frame optimization. The single-camera pipeline is consistent with the multi-camera reference, with 0.77 cm and 0.45 cm average mesh error for hand and object, respectively, and its final refinement stage yields a further 15% improvement. For single-image hand pose prediction, a CNN that predicts 21 2D joints plus 20 root-relative joint directions before MANO fitting achieves 1.06 cm mesh error and 3.04 cm joint error, versus 1.14 cm and 3.14 cm without direction predictions and 1.30 cm and 8.31 cm for a retrained Hasson et al. baseline that directly regresses MANO parameters, demonstrating that keypoint-based lifting is more accurate under occlusion and that the dataset supports generalization to unseen objects.

## Contributions

- HOnnotate, a fully automatic method for annotating real RGB-D images of hand-object interaction with 3D hand and object poses, robust to large mutual occlusions through joint multi-frame optimization.
- HO-3D, the first markerless dataset of color images with 3D annotations for both hand (MANO) and object (6D pose), covering 77,558 frames, 68 sequences, 10 subjects, and 10 YCB objects.
- A single-RGB-image hand pose baseline trained on HO-3D that handles severe object-induced occlusions and generalizes to unseen objects.
- A reusable single-camera annotation setup that lowers the barrier for other researchers to extend the dataset with new objects and scenes.

## Limitations

The authors note that hand segmentation quality is a limiting factor, since the segmentation network had to be trained on synthetic composites, and that this sometimes affects annotation accuracy. The physical plausibility term does not measurably improve pose accuracy, and joint hand-object pose estimation from a single RGB frame is left as future work rather than being demonstrated. As with any automatically generated ground truth, the annotations carry residual error on the order of several millimeters, which the paper quantifies (0.77 cm mean joint error) but which should be considered when using HO-3D as a benchmark reference.
