# FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape From Single RGB Images

**Authors:** Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, Thomas Brox  
**Date:** 2019 (ICCV 2019)  
**Identifier:** DOI `10.1109/ICCV.2019.00090`  
**Zotero item:** `UPJ9BN4I` ([Zotero](zotero://select/library/items/UPJ9BN4I))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

FreiHAND is the first large-scale multi-view RGB hand dataset annotated with both 3D pose and shape, created to address the dataset bias the authors document in single-view 3D hand pose estimation: networks trained on existing datasets perform well in-domain but generalize poorly elsewhere (average cross-dataset ranks of 2.9-6.1 for seven prior datasets versus 2.2 for FreiHAND). The dataset covers 32 subjects and hand-object interactions, captured with 8 synchronized cameras and annotated through an iterative semi-automated human-in-the-loop procedure that fits the MANO model. Its 33K-sample training set (augmented to 132K with background compositing) and 4K-sample in-the-wild evaluation set enable supervised monocular articulated hand shape estimation for the first time.

## Background and Motivation

Synthetic datasets struggle to reproduce real image characteristics and often sample unrealistic poses, while controlled real datasets carry markers, gloves, or narrow variation. The paper's cross-dataset experiment — training the same architecture on each of STB, RHD, GANerated, Panoptic, LSMV, FPHA (FPA), and HO-3D, then testing on all evaluation splits — shows substantial drops out-of-domain: FPHA-trained networks exploit visible magnetic markers, GANerated suffers from texture and color artifacts, and strong STB results turn out not to predict generalization. Sparse manual annotation of all 21 keypoints across views (about 15 minutes per multi-view set) is too expensive at scale and yields no shape information, motivating a bootstrapped annotation strategy.

## Dataset Construction

Eight calibrated, time-synchronized RGB cameras at the corners of a cube record 32 subjects of different genders and ethnic backgrounds performing actions with and without household objects, demonstrating varied grasping techniques. Annotation starts from automatically extracted green-screen segmentation masks (refined to align the model wrist) plus six manually placed 2D keypoints (fingertips and wrist); a multi-term MANO fitting objective combines 2D/3D keypoint, segmentation, shape, and pose priors to produce pose and shape candidates. A multi-view network (MVNet) predicting 3D keypoints with confidence enables bootstrapping: heuristic criteria (confidence above 0.8, per-keypoint above 0.6, mask IoU at least 0.7, and keypoint distances under 0.5 cm) auto-accept samples, while annotators verify or refine the rest in about 5 seconds each. Four iterations grow verified fits from 302 to 993, 1449, 2609, and 4565 samples. All green-screen recordings (24 subjects) form the training set; the evaluation set has 11 subjects (3 shared) captured in 2 indoor and 1 outdoor location.

## Evaluation Protocol

Cross-dataset generalization is scored as the area under the percentage-of-correct-keypoints curve of a single-view pose network trained on each dataset and tested on all others, ranked by cumulative average rank. The shape estimation benchmark takes a single RGB image as input and requires predicting MANO pose and shape parameters; predicted meshes are Procrustes-aligned and scored with mean per-vertex mesh error and F-scores at 5 mm and 15 mm thresholds against the mean-shape baseline, a MANO fit to predicted 3D keypoints, and a Kanazawa-style MANO-parameter regression network (MANO CNN).

## Findings and Analysis

The network trained on FreiHAND ranks first across all seven external evaluation sets (AUC of 0.473 on STB, 0.518 on RHD, 0.562 on Panoptic, 0.537 on LSMV, 0.557 on FPHA, 0.217 on GANerated, and 0.678 on its own evaluation set), confirming that its pose, shape, viewpoint, and object variation reduce dataset bias. The bootstrapping loop improves monotonically: cross-dataset AUC on RHD rises from 0.244 to 0.518 and on Panoptic from 0.347 to 0.562 as accepted annotations grow. On shape estimation, MANO CNN outperforms both baselines with 1.16 mean mesh error and F-scores of 0.484 at 5 mm and 0.925 at 15 mm, versus 1.45/0.415/0.884 for fitting a MANO model to predicted keypoints and 1.78/0.300/0.808 for the mean shape, with the largest margin in the fine-grained regime.

## Contributions

The largest RGB hand dataset with paired pose and shape labels at publication time; a scalable iterative human-in-the-loop annotation pipeline combining sparse manual input, multi-view MANO fitting, MVNet confidence-based auto-acceptance, and human verification; a quantitative demonstration and mitigation of cross-dataset bias in single-view hand pose estimation; and the first real benchmark with training and evaluation protocols for monocular articulated hand shape estimation.

## Limitations

The training portion is restricted to green-screen recordings requiring background compositing (with harmonization and colorization to hide green bleeding), and although the evaluation set spans indoor and outdoor scenes, the overall scale (about 33K training samples) remains below synthetic alternatives; annotations are MANO fits rather than direct measurements, so fitting errors propagate into the ground truth; and objects are limited to household items allowing one-handed manipulation, without 3D object pose or shape annotations.
