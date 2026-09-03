# HACS: Human Action Clips and Segments Dataset for Recognition and Temporal Localization

**Authors:** Hang Zhao, Antonio Torralba, Lorenzo Torresani, Zhicheng Yan  
**Date:** 2019 (ICCV 2019)  
**Identifier:** [arXiv:1712.09374](https://arxiv.org/abs/1712.09374)  
**Zotero item:** No record found in the Zotero library; identity verified against arXiv metadata.  
**Evidence status:** No Zotero record; verified against full-text PDF extraction (arXiv 1712.09374).

## Summary
The paper presents HACS (Human Action Clips and Segments), a large-scale video dataset targeting both action recognition and temporal action localization, two areas whose dataset scales had diverged sharply. HACS consists of two annotation types sharing the 200-class ActivityNet-v1.3 taxonomy: HACS Clips, with 1.5M annotated 2-second clips sparsely sampled from 504K untrimmed YouTube videos (0.6M positive and 0.9M negative), and HACS Segments, with 139K densely annotated action segments (start, end, and label) in 50K untrimmed videos. The authors first run an empirical study of clip-sampling strategies—random sampling versus maximum-consensus and maximum-entropy image-classifier sampling—and use the winning strategy to concentrate human annotation effort. On HACS Clips they benchmark recognition models and show HACS beats Kinetics-600, Moments-in-Time, and Sports1M as a pretraining source; on HACS Segments they evaluate proposal generation and localization methods, showing that denser, shorter segments make localization substantially harder than on ActivityNet.

## Background and Motivation
Recognition datasets grew by two orders of magnitude (Sports1M, Kinetics, Moments-in-Time), but localization datasets did not: THUMOS 2014 has 2.7K untrimmed videos over 20 classes, ActivityNet 20K videos with 30K annotations, AVA 58K clips, and Charades 67K intervals. The authors argue this gap impedes sophisticated localization models. Dense manual annotation of untrimmed videos is expensive, so they instead sparsely sample short clips from many videos and have annotators rapidly verify whether the target action occurs—an efficient scheme that also yields a pretraining corpus for spatiotemporal feature learning. A further motivation is methodological: classifier-guided sampling (as in Kinetics, which used Google Image Search feedback classifiers) introduces hard-to-assess dataset bias, so the authors systematically compare sampling strategies before committing.

## Dataset Construction
- **Source and taxonomy:** YouTube videos retrieved by querying 200 action class labels (890K candidate videos, 1100-6600 per class); taxonomy identical to ActivityNet-v1.3. Videos are strictly shorter than 4 minutes with an average length of 2.6 minutes. De-duplication is performed both within HACS and against the validation/test sets of Kinetics, ActivityNet, UCF-101, and HMDB-51.
- **Preprocessing:** Shot detection via color histogram distance, then a Faster R-CNN person detector on two frames per shot removes shots without people.
- **Clip sampling study:** Two ResNet-50 image classifiers (a YouTube Frame Model trained on 600K person-verified frames from top-500 videos per class, and a Google Image Model trained on 304K searched images) predict action probabilities per shot center frame. Three sampling strategies are compared: Random, Maximum Entropy (ME—sample shots where the two classifiers disagree), and Maximum Consensus (MC—sample shots where both classifiers score the retrieval class highly). In a controlled "Train-mini" study (400 videos per class, 3 clips per video, human-verified 2-second clips), MC yields the most positives (100.3K positive versus 82.2K Random and 71.3K ME) and models trained on Train-mini-MC generalize best across all validation sets. Because ME-sampled validation clips are hardest and MC clips easiest, the final validation/test sets combine clips from all three sampling methods.
- **Clip annotation:** Each clip is labeled by three annotators with a detailed per-class guideline (definitions plus positive/negative examples); clips are kept only with consensus from at least two annotators, and clips of the same class are labeled by the same annotator group to remove inter-annotator noise.
- **HACS Clips scale:** 1.5M clips from 504K videos; splits of 1.4M train (492K videos), 50K validation (6K videos), 50K test (6K videos). Each video contributes 3 clips with a negative-to-positive ratio of roughly 1 to 2, so the set also contains many hard negatives (person and context present, action absent).
- **HACS Segments:** For a 50K-video subset (38K train, 6K validation, 6K test), annotators densely mark the start, end, and class of every action segment using a timeline tool. Guidelines separate foreground action segments from background (e.g., interview footage during Belly Dance, or explanations while a rider is visible for BMX, count as background). Result: on average 2.8 action segments per video (1.8x ActivityNet's 1.5) with average segment duration 40.6 seconds versus ActivityNet's 51.4 seconds.

## Evaluation Protocol
- **Recognition on HACS Clips:** classify a 2-second clip into one of 200 classes (plus background for the annotation task); metric is mean class accuracy (Class@1) on the validation set. Baseline I3D models are trained with RGB, flow (Farneback), and late fusion.
- **Transfer learning:** I3D models pretrained on HACS Clips, Kinetics-600, Moments-in-Time, or Sports1M are fine-tuned on UCF-101, HMDB-51 (split 1), and Kinetics-400 (validation Video@1, averaging predictions over 10 evenly sampled clips); two-stream I3D and R(2+1)D-34/101 are compared against prior work.
- **Proposal generation on HACS Segments:** BSN and TAG are trained with HACS supervision (TSN snippet features over 200 classes plus background) and evaluated with Average Recall@100 and AUC of the AR-AN curve, averaged over tIoU thresholds 0.5-0.95 at 0.05 increments. HACS Segments Mini (10K training videos, 50 per class) is introduced for a training-size-controlled comparison against ActivityNet.
- **Action localization:** Structured Segment Networks (SSN) is trained and tested on HACS Segments, reporting mAP at tIoU 0.5, 0.75, 0.95, and their average, with late fusion of RGB and flow scores.

## Findings and Analysis
- **Clip classification on HACS Clips validation:** I3D reaches 80.3% Class@1 with RGB, 72.2% with flow, and 83.5% with RGB+flow late fusion.
- **Pretraining comparison (I3D, RGB, fine-tuned):** HACS Clips beats Kinetics-600 on all three targets—UCF-101 95.1% versus 94.9%, HMDB-51 73.6% versus 73.4%, Kinetics-400 73.4% versus 72.9%—and clearly beats Sports1M and Moments-in-Time; the authors attribute this to 3x more training annotations than Kinetics-600 and a more fine-grained taxonomy than Moments.
- **State-of-the-art comparisons:** two-stream I3D pretrained on HACS Clips reaches 98.2% on UCF-101, 81.3% on HMDB-51, and 76.4% on Kinetics-400; two-stream R(2+1)D-101 reaches 77.0% on Kinetics-400. Pretraining HACS Clips also improves CDC localization by 8.6% average mAP on THUMOS 14 and 2.5% on ActivityNet over training from scratch.
- **Proposal generation:** BSN achieves AR@100 63.62 and AUC 53.41 on HACS Segments (TAG: 55.88 and 49.15); on the size-matched HACS Segments Mini, BSN drops to AR@100 61.85 and AUC 51.59, versus 74.16 and 66.17 on ActivityNet—showing HACS Segments is a harder localization benchmark with more, shorter segments per video.
- **Localization:** SSN reaches 28.82% mAP@0.5 and 18.97% average mAP (tIoU 0.5-0.95) on HACS Segments, versus 43.26% and 28.28% on ActivityNet; the 12.35% average-mAP gap on matched video scale is attributed to HACS's precise, dense segment annotations, and the 3.04% gap between full and Mini sets shows large training sets boost accuracy.

## Contributions
- HACS Clips: at 1.5M annotated clips, the largest manually verified action clip dataset (2.5x Kinetics-600's annotations), usable both as a recognition benchmark and as a pretraining source.
- A thorough empirical study of clip-sampling strategies showing maximum-consensus classifier sampling maximizes positive yield and downstream generalization, and that validation/test sets should mix all three sampling methods to avoid bias.
- HACS Segments: 2.5x more videos and 4.7x more action segments than ActivityNet, with guidelines that sharpen action-boundary definitions.
- Baselines and controlled comparisons demonstrating both the transfer value of HACS Clips and the increased localization difficulty posed by dense, short segments.
- Public release via http://hacs.csail.mit.edu.

## Limitations
- Classifier-guided clip sampling introduces dataset bias; the paper can measure its effects across validation sets but relies on mixing sampling methods rather than eliminating the bias.
- HACS Clips provides only clip-level positive/negative labels (2-second clips) without temporal boundaries, so it cannot by itself supervise localization; boundary supervision is limited to the 50K-video subset.
- Annotation quality is bounded by a two-of-three annotator consensus rule and per-class annotator grouping, which reduces but does not remove label noise.
- The 200-class taxonomy is inherited unchanged from ActivityNet-v1.3, inheriting its granularity and class-definition choices.
- Baselines show large absolute gaps remain (average localization mAP under 19%), indicating the benchmark is far from solved; the authors present this as a challenge rather than resolving it.
