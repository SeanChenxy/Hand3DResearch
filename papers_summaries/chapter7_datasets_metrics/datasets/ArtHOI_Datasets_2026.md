# ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions

**Authors:** Zikai Wang, Zhilu Zhang, Yiqing Wang, Hui Li, Wangmeng Zuo  
**Date:** 2026 (CVPR 2026; arXiv March 2026)  
**Identifier:** [arXiv:2603.25791](https://arxiv.org/abs/2603.25791)  
**Zotero item:** `ISKVHBWR` ([Zotero](zotero://select/library/items/ISKVHBWR))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

ArtHOI tackles 4D reconstruction of hands interacting with articulated objects from a single monocular RGB video, a setting prior HOI methods exclude because they assume rigid objects and prior articulated-object methods require pre-scanned objects or multi-view input. The optimization-based framework fuses foundation-model priors (depth, segmentation, image-to-3D, 6D pose, MLLM contact reasoning) and introduces Adaptive Sampling Refinement for metric scale/pose plus MLLM-guided hand-object alignment. Evaluation rests on two newly contributed benchmarks, ArtHOI-RGBD and ArtHOI-Wild, on which the method outperforms the pre-scan-dependent RSRD baseline.

## Background and Motivation

The paper identifies an evaluation gap: existing HOI reconstruction methods assume rigid objects, while 4D articulated-object reconstruction requires pre-scanned canonical geometry or multi-view videos, so uncontrolled interactions with objects like scissors and laptops remain unhandled. Foundation models provide geometric, motion, and semantic priors, but naive integration fails because generated meshes lack metric scale and separately reconstructed hands and objects interpenetrate or lose contact. The authors therefore build a benchmark suite spanning controlled and in-the-wild data to evaluate template-free, pre-scan-free reconstruction.

## Dataset Construction

ArtHOI-RGBD contains five demonstration sequences of common articulated objects (headphone, scissor, candy box, CD drive, stapler) captured with an Intel RealSense stereo camera at 1280 x 720, 30 FPS, with accurate metric depth. ArtHOI-Wild contains eight in-the-wild clips collected from internet sources and smartphone recordings. Ground truth is created with a 3D annotation tool built on Viser: part-wise object motions are labeled across frames for all five ArtHOI-RGBD videos and four RSRD videos, guided by depth maps; complete object geometry comes from an additional surrounding scan of each object; hand-object contact states are annotated for all used videos. The evaluation also reuses nine RSRD videos and a three-object ARCTIC subset (mixer, box, scissors).

## Evaluation Protocol

The method is compared against RSRD, which needs pre-scanned object sequences, and EasyHOI applied frame-by-frame. Articulated-object reconstruction is scored with Chamfer distance (mm), Maximum Symmetry-Aware Surface Distance, and F-scores at 5 mm and 10 mm thresholds. Hand-object alignment is scored with the Collision-Contact (Co2) metric from Open3DHOI on annotated contact frames. MLLM contact reasoning is evaluated by binary contact accuracy and contacting-finger accuracy, with predictions within 1-3 frames of an annotated contact window counted as correct. No existing method performs the same task without pre-scans, so RSRD's Gaussian part representation is replaced with meshes for fair comparison, and RSRD cannot process ArtHOI-Wild at all.

## Findings and Analysis

On the five ArtHOI-RGBD sequences, ArtHOI achieves the lowest errors on every object, e.g., CD drive CD 3.334 mm versus RSRD's 282.330 mm, and stapler 4.487 mm versus 288.704 mm. On RSRD's own dataset the results are comparable despite no pre-scanning (scissor CD 5.447 mm versus 68.564 mm), though RSRD is better on the bear object (8.739 mm versus 12.374 mm). RSRD fails to reconstruct objects on the ARCTIC subset, while ArtHOI reaches CDs of 12.1 mm (mixer) and 14.0 mm (box) with contact accuracies of 82.5% and 76.6%. MLLM-guided alignment lowers the Co2 score on ArtHOI-RGBD from 0.972 (unaligned) to 0.029, versus 0.392 for RSRD with WiLoR hands. The full prompting strategy raises MLLM contact accuracy to 88.58% on RSRD and 86.56% on ArtHOI-Wild while cutting false positives to 11.20% and 9.81%; a mask-intersection heuristic degrades to 76% accuracy on wild data. ASR attains a 100% scale/pose optimization success rate versus 57-78% for FoundationPose and Any6D.

## Contributions

The paper contributes the ArtHOI framework for pre-scan-free monocular 4D hand-articulated-object reconstruction, the ASR metric scale/pose optimization, MLLM-guided contact-constrained alignment, and two new annotated benchmarks (ArtHOI-RGBD, ArtHOI-Wild) with part-motion and contact annotations.

## Limitations

The newly contributed benchmarks are small (five controlled and eight in-the-wild sequences), and ground-truth meshes require additional surrounding scans that internet videos cannot provide, which is also why RSRD cannot be run on ArtHOI-Wild. Comparison with RSRD required substituting its Gaussian part representation with meshes, and evaluation tolerates 1-3 frames of slack in contact labeling. The pipeline takes roughly one hour for a 100-frame video on an NVIDIA A6000 GPU, dominated by part-wise motion reconstruction.
