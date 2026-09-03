# The Kinetics Human Action Video Dataset

**Authors:** Will Kay, João Carreira, Karen Simonyan, Brian Zhang  
**Date:** 2017 (arXiv)  
**Identifier:** [arXiv:1705.06950](https://arxiv.org/abs/1705.06950)  
**Zotero item:** No record found in the Zotero library; identity verified against arXiv metadata.  
**Evidence status:** No Zotero record; verified against full-text PDF extraction (arXiv 1705.06950).

## Summary
The paper introduces Kinetics, a DeepMind dataset of 400 human action classes with at least 400 clips per class, designed to be the large-scale successor to HMDB-51 and UCF-101 and the video analogue of ImageNet: large enough to train deep video networks from scratch and varied enough to separate the merits of different architectures. Each clip lasts around 10 seconds and is extracted from a unique YouTube video, avoiding the correlated clips that inflate performance on smaller datasets. The classes cover single-person actions, person-person interactions (e.g., shaking hands), and person-object interactions (e.g., playing instruments). The authors document the four-stage collection and de-noising pipeline, benchmark three standard ConvNet architectures trained from scratch, and report far lower accuracy on Kinetics than on UCF-101 (best baseline 61.0% top-1 versus 92.5%), establishing Kinetics as a substantially harder benchmark.

## Background and Motivation
HMDB-51 and UCF-101 served the community well but are no longer large enough or varied enough for the current generation of deep action classification models; UCF-101's 101 classes with 100+ clips each are drawn from only 2.5k distinct videos (e.g., 7 clips of the same person brushing their hair), so variation across performers, viewpoints, and lighting is far lower than the clip count suggests. The authors' goal is a classification dataset—deliberately excluding temporal localization—whose per-class clips each come from a different video, giving large variety in performers, speed, clothing, pose, age, and camera framing. Clips retain audio so the dataset can also support multi-modal analysis. The hope is to enable architectures with multiple input streams (RGB, flow, pose, objects), attention mechanisms, and a resolution of open questions such as recurrent versus convolutional temporal aggregation.

## Dataset Construction
- **Content:** 400 human action classes with 400-1150 clips per class, each clip around 10 seconds and taken from a unique YouTube video; 306,245 videos in total. Variable resolution and frame rate, resampled to 25 FPS for experiments.
- **Class structure:** No deep hierarchy; instead non-exclusive parent-child groupings such as Music, Personal Hygiene, Dancing, and Cooking. Classes include fine-grained temporal distinctions (swimming styles) and object-driven distinctions (wind instruments). Annotation is non-exhaustive: a clip may contain several Kinetics actions but is listed under only one, motivating top-5 evaluation.
- **Stage 1, action list:** Compiled from existing datasets (ActivityNet, HMDB, UCF101, MPII Human Pose, ACT), motion-capture file titles, and crowdsourced suggestions when a presented label was wrong.
- **Stage 2, candidate clips:** YouTube videos matched by title against the action list (verbs formatted as gerunds); image classifiers built from Google Image Search user relevance feedback were run at frame level, and 10-second clips were extracted around the top k = 2 responses (5 seconds either side).
- **Stage 3, AMT labeling:** Workers answered "Can you see a human performing the action class-name?" with Yes / No / Unsure / bad-video options; audio was withheld so classification is purely visual; workers also recorded whether the action lasts the whole clip. A clip was accepted with at least 3 confirmations out of 5 annotations; two of every 20 task slots were injected ground-truth clips and workers below 50% accuracy were warned; per-class quality gates started at 50% expected positive rate. The task drew more than 400 distinct workers per run.
- **Stage 4, cleanup:** One clip per YouTube link (filtering ~20% of Turker-approved examples); cross-link de-duplication via Inception-V1 averaged frame features with a cosine-similarity threshold of 0.97 (a further ~15% reduction); noisy or overlapping classes detected with repeatedly trained two-stream classifiers and merged, split, or removed; a final manual filtering pass sorted clips by two-stream confidence and removed the noisiest and residual duplicates.

## Evaluation Protocol
The task is human action classification: given a clip, predict one of the 400 classes. Because annotation is non-exhaustive, top-5 accuracy is considered more suitable than top-1. The dataset is split into training (250-1000 clips per class), validation (50 per class), and test (100 per class); baselines train on train/val and report on the held-out test set. Three architectures are benchmarked, all trained from scratch on Kinetics (ImageNet pretraining is used only for the small UCF-101 and HMDB-51 experiments): (a) ConvNet+LSTM, a ResNet-50 frame model with a 512-unit batch-normalized LSTM; (b) two-stream networks with ResNet-50 spatial and flow streams (10 stacked flow frames); (c) a 3D ConvNet, a modified C3D with batch normalization after all layers and a temporal-stride-2 first pooling, taking 16-frame 112x112 clips. Inputs are resampled to 25 FPS with the larger side scaled to 340 pixels (ResNet models) or 128 pixels (3D ConvNet); augmentation is random spatial cropping, temporal jitter of the start frame, and horizontal flipping.

## Findings and Analysis
- **Kinetics test set (top-1 / top-5):** ConvNet+LSTM RGB 57.0% / 79.0%; Two-Stream RGB 56.0% / 77.3%, Flow 49.5% / 71.9%, RGB+Flow 61.0% / 81.3%; 3D-ConvNet RGB 56.1% / 79.5%.
- **Cross-dataset comparison (RGB+Flow where applicable):** two-stream reaches 92.5% on UCF-101 split 1 but only 61.0% top-1 on Kinetics; on HMDB-51 split 1 it reaches 63.7%, i.e., HMDB-51 remains harder than Kinetics for appearance-heavy models despite its small training set. The parameter-rich, non-pretrained 3D-ConvNet performs poorly on UCF-101 (51.6%) and HMDB-51 (24.3%) but approaches the other models on Kinetics, supporting the claim that only a large dataset can train 3D ConvNets from scratch.
- **Class difficulty:** eating classes (distinguishing hotdogs, chips, doughnuts), dancing classes, and body-part-centered classes (e.g., "massaging feet", "shaking head") are hardest under the two-stream model.
- **Confusions:** the top confusions are fine-grained pairs, e.g., 'riding mule' vs 'riding or walking with horse' (40%), 'hockey stop' vs 'ice skating' (36%), 'swing dancing' vs 'salsa dancing' (36%), 'triple jump' vs 'long jump' (26%).
- **Dataset bias study:** in 340 of 400 classes the data is not dominated by a single gender or gender is indeterminable; for imbalanced classes (e.g., 'shaving beard', 'cheerleading') little evidence of classifier bias was found. One clear age-related bias exists in 'crying' (biased toward babies). The authors flag this analysis as preliminary.

## Contributions
- A 400-class, 306,245-clip action classification dataset with one clip per source video, an order of magnitude larger than prior action datasets, released via http://deepmind.com/kinetics.
- A documented large-scale collection pipeline combining image-classifier-driven clip localization with crowdsourced verification and multi-stage de-duplication and de-noising.
- From-scratch baselines showing Kinetics is much harder than UCF-101 while finally enabling 3D ConvNet training.
- A preliminary study of dataset imbalance and classifier bias along gender and age axes.
- Released trained baseline models for feature extraction on new action classes.

## Limitations
- The dataset targets classification only: clips are around 10 seconds, there are no untrimmed videos, and no temporal localization benchmark is provided.
- Annotation is not exhaustive: a clip can contain multiple Kinetics actions but carries a single label, which degrades top-1 evaluation.
- The class list was curated without a single authoritative source, and some classes overlap or are visually confusable (up to 40% cross-class confusion), requiring merges, splits, and removals during collection.
- Clips were positioned by image classifiers within the video, which the authors argue does not constrain visual variety but acknowledge as a pipeline-induced bias consideration.
- The bias analysis is explicitly preliminary; the authors state that a thorough investigation of dataset imbalance and classifier bias, with social scientists and critical humanists, is left to future work.
