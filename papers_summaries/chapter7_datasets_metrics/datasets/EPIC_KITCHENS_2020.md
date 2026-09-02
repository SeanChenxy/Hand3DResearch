# The EPIC-KITCHENS Dataset: Collection, Challenges and Baselines

**Authors:** Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, Michael Wray  
**Date:** 2020 (TPAMI; arXiv April 2020)  
**Identifier:** [arXiv:2005.00343](https://arxiv.org/abs/2005.00343)  
**Zotero item:** `PYNDZTKB` ([Zotero](zotero://select/library/items/PYNDZTKB))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

EPIC-KITCHENS is the largest egocentric video benchmark at its release, recorded by 32 participants in their native kitchens across 4 countries, yielding 55 hours (11.5M frames) of non-scripted daily activity densely annotated with 39.6K action segments and 454.2K active-object bounding boxes. Its annotation pipeline is unique in having participants narrate their own videos after recording, reflecting true intention. The paper defines object detection, action recognition, and action anticipation challenges over seen-kitchen and unseen-kitchen test splits, with baselines showing the value of multimodal inputs and explicit temporal modeling.

## Background and Motivation

Existing egocentric datasets are small, often captured in a single environment, and rely on scripted activities, which remove the multi-tasking, searching, and plan changes of real kitchen life. The authors target unscripted, untrimmed recordings of natural interactions so that the same data can support many downstream tasks. Compared to ADL, the closest native-environment egocentric dataset, EPIC-KITCHENS has 11.5M versus 1M frames, 90x more action segments, and 4x more object bounding boxes.

## Dataset Construction

Recordings use a head-mounted GoPro (Full HD, 59.94 fps, stereo audio) started on every kitchen entry over at least three consecutive days per participant; participants recorded alone after removing identity-disclosing items. Post-recording, each participant narrated their actions (17 in English, 7 Italian, 6 Spanish, 1 Greek, 1 Chinese), which were transcribed, timed via caption alignment, and refined by AMT workers who adjust start/end times with a union-of-best-agreements consensus over 4 annotators, yielding 39,564 segments (mean 3.7 s, 24% overlapping another segment). Active-object bounding boxes are collected for nouns within each segment at 2 fps, producing 454,158 boxes plus 125,375 true-negative (occluded/absent) labels. Free-form narrations are grouped into 125 verb and 331 noun classes; quality sampling reports error rates of 5.7% (segment boundaries), 6.3% (boxes), 3.3% (verbs), and 6.0% (nouns).

## Evaluation Protocol

Ground truth for 27% of the data is held out in two test protocols: S1 (seen kitchens, 80/20 sequence split per kitchen) and S2 (unseen kitchens, 4 participants fully held out, 7% of frames), each with zero-shot verb/noun/action classes (e.g., 220 zero-shot actions in S2). Object detection uses Faster R-CNN with mAP at IoU 0.05/0.5/0.75, separating many-shot (202 classes) and few-shot (78) groups. Action recognition classifies pre-trimmed segments into verb, noun, and action with top-1/top-5 accuracy and per-class precision/recall. Action anticipation predicts the action 1 second before it starts from 0.5-2 s of observation. Baselines include 2SCNN, TSN, TRN, and TSM over RGB, optical flow, and audio modalities, plus DMR and ED for anticipation.

## Findings and Analysis

Object detection is hard: all-class mAP at IoU 0.5 is 28.06 (S1) and 28.57 (S2), below 10% for few-shot classes, though seen and unseen splits perform comparably, suggesting cross-kitchen generalization. In recognition, fusion of RGB, flow, and audio gives the best results (TSN top-1 action 26.06% on S1, 15.61% on S2); flow is best for verbs, RGB for nouns, and audio outperforms RGB for verbs on unseen kitchens. Explicit temporal modeling lifts verb top-1 accuracy by 6.4-8.0% (S1), with TSM reaching 62.37% verb / 29.90% action on S1. Anticipation remains far harder (best action top-1 8.08% on S1), and no single anticipation method dominates.

## Contributions

The largest native-environment egocentric dataset with participant-narrated, densely annotated actions and active objects; a reproducible multi-lingual narration-to-annotation pipeline; three public challenges (object detection, action recognition, anticipation) with seen/unseen splits, zero-shot class accounting, and leaderboard baselines; and analyses showing the importance of temporal modeling and multimodal audio.

## Limitations

Narrations can be incomplete (e.g., 'open' is narrated more often than 'close') and temporally belated; bounding boxes cover only active objects; no 3D hand or object pose annotations are provided, so the benchmark addresses video-level action understanding rather than 3D hand-object interaction reconstruction; and baseline accuracies, especially for anticipation and few-shot detection, remain far from solving the tasks.
