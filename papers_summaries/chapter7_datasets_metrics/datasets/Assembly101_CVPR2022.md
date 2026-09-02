# Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities

**Authors:** Fadime Sener, Dibyadip Chatterjee, Daniel Shelepov, Kun He, Dipika Singhania, Robert Wang, Angela Yao  
**Date:** 2022 (CVPR 2022)  
**Identifier:** DOI `10.1109/CVPR52688.2022.02042`  
**Zotero item:** `JDZ3JI67` ([Zotero](zotero://select/library/items/JDZ3JI67))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Assembly101 addresses the scarcity of large, non-scripted, multi-step activity benchmarks outside the kitchen domain by capturing 4321 videos (513 hours) of 53 adults assembling and disassembling 101 take-apart toy vehicles from 12 synchronized viewpoints. It is annotated with over 1M fine-grained and more than 100K coarse action segments plus 18M 3D hand poses, mistake labels, and participant skill levels. Baselines are provided for action recognition, anticipation, temporal segmentation, and a newly proposed mistake detection task, exposing large gaps on egocentric views, tail classes, unseen toys, and pose-only recognition.

## Background and Motivation

The paper positions procedural activity understanding as dominated by curated instructional or kitchen datasets with strict ordering, few recorded multi-step alternatives, or small scale. Assembly introduces free-style, goal-oriented procedures with natural variation in action ordering, mistakes, and corrections, and its simultaneous static and egocentric recordings enable cross-view transfer and 3D hand-object interaction studies that existing egocentric datasets lack. The authors also note that detecting mistakes in procedural activities had not previously been studied as a dataset task.

## Dataset Construction

Fifty-three adults (28 male, 25 female) each worked on six toys in hour-long sessions, first disassembling a completed toy and then re-assembling it, guided only by a picture of the assembled vehicle with no step instructions. A desk rig with eight RGB cameras (1920 x 1080; five overhead, three side) and four monochrome cameras (640 x 480) on a custom headset records synchronized, calibrated views. The 362 disassembly-assembly sequences span 15 vehicle categories. Fine-grained actions combine 24 verbs and 90 objects (including 5 tools) into 1380 classes with an average duration of 1.7 s; coarse actions combine 11 verbs and 61 part-attach/detach events into 202 classes averaging 16.5 s. Both hands are tracked from the four egocentric cameras with a modified MegATrack, yielding 21 world-coordinate keypoints per hand. Coarse assembly segments carry mistake/correction/correct labels (15.9% and 6.7% of the 60K assembly segments) and each participant receives a 1-5 skill rating.

## Evaluation Protocol

Videos are split 60/15/25 into train/validation/test, with 25 of 101 toys shared across splits and held-out instances for zero-shot evaluation; test ground truth is withheld for online leaderboards. Recognition uses pre-trimmed clips scored by Top-1 verb/object/action accuracy (TSM for video, 2s-AGCN and MS-G3D for pose); anticipation predicts actions 1 second ahead with class-mean Top-5 recall (TempAgg); temporal segmentation assigns frame-wise labels with C2F-TCN and MS-TCN++, scored by MoF, edit distance, and F1@10/25/50; mistake detection classifies current coarse segments as correct/mistake/correction with precision and recall for full-segment and early (half-segment) prediction.

## Findings and Analysis

Fixed views beat egocentric views by 16.2% in action recognition Top-1 (39.2% vs 23.0% overall action accuracy), 4.9% in anticipation recall, and 6.5% MoF in segmentation, and models trained on one view type transfer poorly to the other. Head-tail imbalance is severe: recognition accuracy drops 37% from head to tail classes, and tail-class MoF is 7.2% versus 51.5% for head classes. Seen toys outperform unseen toys in all tasks, mainly on object labels. Pose-only recognition (MS-G3D with context, 28.7% action accuracy) trails fused egocentric video TSM (33.8%) on objects but exceeds it on verbs (65.7% vs 59.0%). TSM features pre-trained on EPIC-KITCHENS reach only 17.3% action accuracy versus 40.5% with in-domain pre-training. Mistake detection is hard even with oracle coarse labels (62.7% mistake recall), and TSM features reach only 46.6%, dropping further in early prediction.

## Contributions

The dataset itself: the largest procedural activity dataset at release, uniquely combining synchronized static and egocentric recordings, multi-granularity action labels, 3D hand poses, mistake and skill annotations; four benchmark tasks with baselines; and analyses of cross-view transfer, long-tail behavior, generalization to unseen toys, skill effects, and pose-versus-appearance recognition.

## Limitations

Hand poses are provided but no 6D object poses, and the paper leaves joint modeling of 3D objects and hand poses to future work. Baseline performances remain far from solving the tasks, especially on tail classes, egocentric views, and mistake early prediction, and the dataset's toy-vehicle domain may not transfer to real industrial assembly despite the procedural structure.
