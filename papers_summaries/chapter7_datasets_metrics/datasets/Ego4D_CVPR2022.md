# Ego4D: Around the World in 3,000 Hours of Egocentric Video

**Authors:** Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, et al.  
**Date:** 2022 (CVPR 2022)  
**Identifier:** [arXiv:2110.07058](https://arxiv.org/abs/2110.07058)  
**Zotero item:** `83EEQPQR` ([Zotero](zotero://select/library/items/83EEQPQR))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Ego4D is a massive-scale egocentric video dataset and benchmark suite offering 3,670 hours of unscripted daily-life video captured by 931 camera wearers across 74 worldwide locations in 9 countries, an order of magnitude larger than prior egocentric collections. Portions of the video carry audio, 3D environment scans, eye gaze, stereo, IMU, and synchronized multi-camera views. The paper contributes five benchmark challenges spanning the past (episodic memory), present (hand-object state changes, audio-visual diarization, social interactions), and future (activity forecasting) of first-person perception.

## Background and Motivation

The authors argue that Internet third-person datasets represent a limited notion of perception: they contain brief, curated clips, whereas robotics and AR require long, fluid first-person streams without a photographer's curation. Prior egocentric datasets either focus narrowly on kitchens, use scripted activities, or rely mostly on graduate-student wearers. Ego4D instead maximizes geographic, demographic, and scenario diversity with unscripted, long-form footage (typical raw clips of 8 minutes), recruiting wearers of varied occupations and ages (45% female, 96 over 50 years old) through 14 partner institutions on 5 continents.

## Dataset Construction

Seven head-mounted camera types (GoPro, Vuzix Blade, Pupil Labs, ZShades, OR-DRO EP6, iVue Rincon 1080, Weeview) are used to avoid overfitting to one device. Modalities include 3,670 hours of RGB video and narrations and precomputed features, 2,535 hours of audio, 836 hours of synchronized multi-camera video, 612 hours of unblurred-face consented video, 491 hours coupled to Matterport3D environment meshes, 224 hours of IMU, 80 hours stereo, and 45 hours of gaze. Every video passes a narration procedure: two independent annotators produce dense timestamped free-form sentences (13.2 sentences per minute on average, 3.85M sentences covering 1,772 unique verbs and 4,336 unique nouns), which seed taxonomies and benchmark sampling. Collection follows consent, de-identification, and blurring protocols.

## Evaluation Protocol

The five benchmarks are: (1) Episodic Memory over 1,000 hours with about 74K queries in three forms — natural language queries localized in time (top-k recall at tIoU thresholds), visual queries localizing a queried object temporally and spatially, and moment queries against a 110-activity taxonomy (mAP at tIoU, plus timeliness metrics). (2) Hands and Objects, the HOI-relevant suite: point-of-no-return temporal localization of object state changes (absolute temporal error), state-change object detection on pre/PNR/post frames (average precision), and state-change classification (accuracy), with hands, tools, and objects boxed in each frame. (3) Audio-Visual Diarization: face localization/tracking, active speaker detection, speech diarization, and transcription, evaluated with MOT metrics, diarization error rate, and word error rate. (4) Social Interactions: Looking-at-Me and Talking-to-Me classification (mAP, top-1). (5) Forecasting: locomotion and hand movement prediction (L2), short-term object interaction anticipation (Top-5 mAP with time-to-contact), and long-term action anticipation (edit distance).

## Findings and Analysis

The paper reports annotations produced by over 250,000 hours of annotator effort, with 48 to 1,000 annotated hours per benchmark on top of the fully narrated 3,670 hours. Baseline models built from state-of-the-art components are provided for all five tasks, with quantitative results in the appendices, and a formal CVPR 2022 challenge was launched to improve them. The narrations alone constitute, to the authors' knowledge, the largest repository of aligned video and language. Stated biases include urban/college-town skew, COVID-era stay-at-home emphasis, and narration language bias from the two annotation sites.

## Contributions

An unprecedented-scale, demographically and geographically diverse egocentric video corpus with rich multimodal streams and privacy safeguards; dense video-language narrations; and a five-benchmark evaluation suite spanning episodic memory, object state change understanding, audio-visual conversation, social interaction, and forecasting, each with task definitions, annotations, metrics, and baselines.

## Limitations

The authors acknowledge that 74 locations far from cover the globe, wearers skew urban, the pandemic limited large social events, battery life biases footage toward active portions of the day, and crowd-sourced narrations carry local language bias. The Hands and Objects benchmark annotates state changes with 2D boxes and timestamps rather than 3D hand or object poses, so it does not by itself support metric 3D hand-object reconstruction.
