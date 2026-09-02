# Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives

**Authors:** Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, et al.  
**Date:** 2024 (CVPR 2024)  
**Identifier:** DOI `10.1109/CVPR52733.2024.01834`  
**Zotero item:** `MXVR856Z` ([Zotero](zotero://select/library/items/MXVR856Z))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Ego-Exo4D is the largest public dataset of time-synchronized egocentric and exocentric video, focused on skilled human activity: 740 participants in 13 cities performed 43 activities across 8 domains (soccer, basketball, dance, bouldering, music, cooking, bike repair, health care) in 123 natural scene contexts, yielding 5,035 takes (1-42 minutes, averaging 2.6) and 1,286 hours of paired ego-exo video. It is multimodal (7-channel audio, IMU, eye gaze, SLAM cameras, 3D point clouds, camera poses) and carries three paired language corpora including novel expert commentary by coaches. Four benchmark families cover ego-exo relation, keystep recognition, proficiency estimation, and 3D body/hand pose.

## Background and Motivation

The paper argues that skill understanding requires both viewpoints: the ego view captures close hand-object interactions and attention, while the exo view captures full-body pose and environment context, and skill acquisition itself requires translating between them (actor-observer mapping). Existing ego-exo datasets (CMU-MMAC, Charades-Ego, Assembly101, H2O, Homage) are small (5-513 hours, 4-71 participants), unsynchronized, or lab-constrained; instructional video datasets lack the paired views. Ego-Exo4D targets real-world experts in authentic settings — professional chefs, athletes, bike technicians, dancers, competitive boulderers — with skill levels from novice to expert to support proficiency modeling.

## Dataset Construction

A low-cost (under $3,000) rig pairs Project Aria glasses (8 MP RGB, two grayscale SLAM cameras, IMU, 7 microphones, eye tracking) with 4-5 tripod-mounted GoPros, auto-synchronized and calibrated via a QR-code procedure into a metric, gravity-aligned frame. Participants wear Aria while exocentric cameras surround them; takes run up to about 60 minutes. Physical domains emphasize body pose and object interaction; procedural domains entail sequences of intricate hand-object manipulations. Language resources are: spoken expert commentary by 52 domain experts (117,812 time-stamped critiques, about 7 per minute, averaging 4 sentences, with 2-5 experts per video plus spatial drawings and skill ratings), participant narrate-and-act tutorials (about 10% of takes), and third-party atomic action descriptions totaling 432K sentences. Annotations required over 200,000 hours of annotator effort.

## Evaluation Protocol

Four task families are defined with baselines in the appendices and public challenges in 2024. Ego-exo relation: correspondence predicts, from a query object mask in one view, the matching mask per synchronized frame of the other view, both directions; translation is decomposed into ego track prediction and ego clip generation, each with and without known ego-camera pose. Keystep recognition: models train on paired ego-exo video but are tested on trimmed ego clips against a taxonomy of 689 keysteps across 17 procedural activities; a second task is online detection under a 20 mW energy budget choosing among audio/IMU/RGB; a third infers procedure structure (previous, optional, mistake, missing, next keysteps) under weak supervision. Proficiency estimation: demonstrator-level classification (novice, early, intermediate, late expert) and demonstration-level temporal localization of good execution versus needs improvement, from ego plus optionally M exo views. Ego pose: given ego video, output 17 3D body joints (MS COCO convention) and 21 3D joints per hand per timestep.

## Findings and Analysis

The ego-pose family provides, to the authors' knowledge, the largest manual ground-truth egocentric body and hand pose collection to date, with about 14M frames of 3D ground-truth and pseudo-ground-truth combined, spanning diverse real-world scenarios (expert musicians, bike mechanics) rather than lab grasps. Baseline results for all tasks are reported in the appendices, and annotations come in two public versions (v1 for the paper's baselines, larger v2 for future challenge leaderboards). The paper emphasizes the complementary failure modes the tasks expose: heavy occlusion and small objects in ego views versus small object size in exo views, and the need to synthesize unseen fingertips in exo-to-ego translation.

## Contributions

The first large-scale, time-synchronized, multimodal ego-exo dataset of skilled activity with metric 3D context; three novel time-indexed language corpora headlined by coach-provided expert commentary; four benchmark families with annotations, metrics, and baselines; and an open-source capture rig and protocol reproducible by other labs.

## Limitations

Capture is restricted to scenarios with closed environments and consented participants, so nearly all video is unblurred but social passers-by are limited; the expert commentary and narrate-and-act streams cover only part of the dataset (about 10% of takes for narrations); ego-pose ground truth relies partly on pseudo-ground-truth to reach the 14M frames; and baseline performance details are deferred to appendices, with the first leaderboards only starting in 2024.
