# FineGym: A Hierarchical Video Dataset for Fine-grained Action Understanding

**Authors:** Dian Shao, Yue Zhao, Bo Dai, Dahua Lin  
**Date:** 2020 (CVPR 2020)  
**Identifier:** [arXiv:2004.06704](https://arxiv.org/abs/2004.06704)  
**Zotero item:** No record found in the Zotero library; identity verified against arXiv metadata.  
**Evidence status:** No Zotero record; verified against full-text PDF extraction (arXiv 2004.06704).

## Summary
The paper introduces FineGym, a dataset built on professional gymnastics videos that targets fine-grained action understanding, where methods must both parse an activity into temporal phases and distinguish subtly different action classes. Unlike coarse-grained benchmarks (UCF101, Kinetics, ActivityNet) where background context can dominate recognition, FineGym annotates every routine with a three-level semantic hierarchy—event, set, and element—and a two-level temporal structure—actions and sub-actions. Version 1.0 covers 10 gymnastic events, 15 sets, and 530 element categories with 4,883 action instances and 32,697 sub-action instances drawn from 303 competition records totaling about 708 hours. Element labels are assigned by a trained annotation team using decision trees derived from the official FIG Code of Points rather than crowdsourcing. Systematic experiments show that state-of-the-art methods, which excel on coarse benchmarks, degrade sharply at element level (best mean accuracy 46.5% on the long-tailed Gym288 setting), and that sparse frame sampling, appearance-only cues, and Kinetics pretraining all fail to transfer to this regime.

## Background and Motivation
On coarse benchmarks the latest models achieve very high accuracy, but categories such as "hockey" versus "gymnastics" are often separable from background context alone, and a few frames sometimes suffice. Sport analytics instead requires fine-grained capability along two axes: temporally, decomposing an action into smaller elements; semantically, differentiating sibling sub-classes in the taxonomy. Existing fine-grained datasets (Breakfast, MPII-Cooking 2, Something-Something, Diving48) have far fewer finest-level classes (e.g., 48 in Diving48 versus 530 in FineGym) and lack multi-level structure. FineGym's gymnastics domain offers rapid movements and dramatic body deformations, consistent backgrounds that force models to attend to the actions themselves, and—crucially—official expert documentation (the FIG 2017-2020 Code of Points) that defines and organizes categories consistently, sidestepping the impracticality of manually designing such a taxonomy.

## Dataset Construction
- **Source:** Official Internet video records of top-level gymnastics competitions, selected for completeness, distinctiveness, and high resolution (over 95% at 720P/1080P); 303 competition records totaling roughly 708 hours, cut evenly into 10-minute chunks, with redundancy removed by manual checking.
- **Semantic hierarchy (three levels):** Events are routines such as vault (VT), floor exercise (FX), uneven bars (UB), and balance beam (BB)—10 event categories in total (6 male, 4 female); sets are mid-level categories grouping technically and visually similar elements (15 sets selected from the official codebook across 4 events); elements are the finest categories (530), e.g., "double salto backward tucked" within the beam-dismounts set.
- **Temporal hierarchy (two levels):** annotators first locate the start and end of every complete routine (an action instance) and assign its event label, discarding incomplete routines; they then decompose each routine into sub-action instances with temporal boundaries. Annotated sub-actions usually last under 2 seconds. Vault routines average 8 seconds, other events about 55 seconds.
- **Element labeling via decision trees:** Because element differences are too subtle for direct assignment, each set has a manually built decision tree of attribute-based queries; an annotator travels from the root (set label) to a leaf (element label). Tree paths also yield attribute sets and difficulty values beyond the label itself.
- **Annotation workforce and quality control:** Crowdsourcing (e.g., AMT) was rejected as infeasible for this expertise; instead a specially trained team was used, with domain-specific training, rigorous pretests, referential slides and demos, and cross-validation across annotators.
- **Statistics (v1.0):** VT: 1 set, 67 element classes, 2,034 instances / 2,034 sub-action instances; FX: 5 sets, 111 element classes, 912 / 8,929; BB: 5 sets, 135 element classes, 976 / 11,586; UB: 4 sets, 133 element classes, 961 / 10,148; total 4,883 instances and 32,697 sub-action instances. Instances per element category range from 1 to 1,648 (heavy-tailed); 354 of the 530 defined categories have at least one instance. Both a natural long-tailed setting (Gym288) and a balanced setting (Gym99, thresholding instance counts) are provided.

## Evaluation Protocol
- **Coarse-grained recognition (event and set level):** Temporal Segment Network (TSN) with 3 sparsely sampled frames, comparing RGB, optical flow, and two-stream inputs.
- **Element-level recognition (three sub-tasks):** (a) elements across all events in the Gym288 (long-tail) and Gym99 (balanced) settings; (b) elements within an event, on Vault (6 classes) and Floor Exercise (35 classes); (c) elements within a set, on FX-G1 (11 classes) and UB-G1 (15 classes). Methods span 2D-CNN pipelines (TSN, TRN, TRN-ms, TSM, ActionVLAD), 3D-CNNs (I3D, Non-local I3D, with and without Kinetics pretraining), and a skeleton-based method (ST-GCN). Metrics: mean class accuracy and top-1 accuracy.
- **Temporal action localization:** Structured Segment Network (SSN) localizes (i) actions (events) within untrimmed videos and (ii) sub-actions within action instances; metric is mAP at tIoU thresholds 0.5-0.95 (0.05 interval) and their average.
- **Diagnostic protocols:** varying the number of sampled frames for TSN (1, 3, 5, 7, 12) against UCF101 and ActivityNet v1.2; shuffling TRN's test frames; training TSM with 3 frames and testing with more; comparing Kinetics- versus ImageNet-pretrained I3D per class.

## Findings and Analysis
- **Event/set level resembles coarse recognition:** TSN reaches 99.86% top-1 (two-stream) at event level and 97.69% at set level; three frames—under 5% of all frames—already suffice, and appearance dominates at event level while motion dominates at set level.
- **Element level is far harder (Gym288/Gym99 mean accuracy):** TSM two-stream is best at 46.5%/83.1%; TRN-ms two-stream 43.3%/82.0%; TSN two-stream 37.6%/79.9%; ActionVLAD RGB only 16.5%/60.5%; I3D RGB 27.9%/66.7%; skeleton-based ST-GCN collapses to 11.0%/34.0%. All methods overfit to head classes under the long-tailed distribution.
- **Granularity effects:** within Vault (6 classes) the best methods reach only about 30-33% mean accuracy (TRN-ms two-stream 30.1%), whereas within-event FX (35 classes) reaches 78.2%; within-set tasks reach 75.8%/83.0% (TRN two-stream). Subtle differences (leg direction relative to turn, bent versus straight legs) dominate the errors.
- **Sparse sampling fails:** TSN on Gym99 rises steadily from 35.46% (1 frame) to 78.82% (12 frames, 30% of all frames), while UCF101 saturates around 85-86.7% with few frames.
- **Temporal dynamics are essential:** shuffling test frames significantly drops TRN performance; TSM trained with 3 frames degrades sharply when tested with more frames, while TSN's average pooling is robust.
- **Pretraining does not always help:** Kinetics-pretrained I3D lifts UCF101 from 84.5% to 97.9%, but on FineGym Kinetics and ImageNet pretraining perform similarly, attributed to the gap between coarse and fine-grained temporal patterns.
- **Localization:** SSN achieves 49.4% average mAP (tIoU 0.5-0.95; 60.0% at 0.5) for actions, but only 9.6% average (22.2% at 0.5, collapsing to 0.6% at 0.9) for sub-actions, whose boundaries require understanding the whole routine.
- **Intermediate representations fail:** person detection and pose estimation miss the gymnast in many frames of intense motion, explaining ST-GCN's poor results.

## Contributions
- FineGym v1.0: a high-quality, fine-grained action dataset with a three-level semantic hierarchy (10 events, 15 sets, 530 elements) and two-level temporal annotations (4,883 actions, 32,697 sub-actions), built from 303 official competition records (~708 hours), released at https://sdolivia.github.io/FineGym/.
- A decision-tree annotation methodology grounded in the official FIG Code of Points, with a trained annotator team and layered quality control, transferable to other domains requiring fine-grained labels.
- Systematic empirical studies isolating what breaks at fine granularity: sparse sampling, appearance reliance, temporal-model train/test mismatch, coarse-grained pretraining, and skeleton estimation.
- Benchmark protocols at three recognition granularities (across events, within event, within set) plus coarse- and fine-grained temporal localization.

## Limitations
- The dataset is restricted to gymnastics, with deliberately consistent backgrounds; the action-centric design aids fine-grained study but limits background/context diversity.
- The element-class distribution is heavily long-tailed (1 to 1,648 instances per category), and 176 of the 530 defined categories have no instances in v1.0.
- Annotation cannot be crowdsourced and depends on a specially trained team, expert documentation, and per-set decision trees, making extension to new domains costly; errors can propagate along the hierarchy, motivating the quality-control overhead.
- Current annotations cover recognition and temporal localization only; auto-scoring, action generation, and multi-attribute prediction are listed as future additions.
- Existing vision components (person detection, pose estimation) are unreliable on this data due to intense motion and rare poses, capping the performance of pose-based methods.
