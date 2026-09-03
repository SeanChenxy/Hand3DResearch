# ActivityNet: A Large-Scale Video Benchmark for Human Activity Understanding

**Authors:** Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, Juan Carlos Niebles  
**Date:** 2015 (CVPR 2015)  
**Identifier:** DOI `10.1109/CVPR.2015.7298698`  
**Zotero item:** No record found in the Zotero library; identity verified against the CVPR 2015 DOI.  
**Evidence status:** No Zotero record; verified against full-text PDF extraction (CVF open-access CVPR 2015 paper).

## Summary
The paper identifies a gap in video action recognition evaluation: existing benchmarks cover few classes, use short manually trimmed clips, and concentrate on sports or simple motions rather than the activities people perform in daily life. The authors introduce ActivityNet, a large-scale benchmark of 203 activity classes sampled from the American Time Use Survey (ATUS) hierarchy, with an average of 137 untrimmed videos per class, 1.41 activity instances per video, and 849 video hours in total. Videos are collected from online sharing sites through a semi-automatic crowdsourcing pipeline in which Amazon Mechanical Turk (AMT) workers both verify untrimmed videos and annotate the temporal boundaries of activity instances. The authors instantiate three benchmarking tasks—untrimmed video classification, trimmed activity classification, and activity detection—and evaluate a state-of-the-art improved-trajectories pipeline on all three. Baseline results are low (42.2% mAP untrimmed, 50.2% mAP trimmed, under 12.5% mAP detection), showing that ActivityNet is substantially harder than prior benchmarks.

## Background and Motivation
Prior action datasets (Hollywood, UCF Sports, Olympic Sports, UCF101, Thumos'14, HMDB51) are limited in one or more of: number of categories, samples per category, temporal length, and taxonomy depth. Sports-1M was the largest dataset, with roughly 500 sports-related classes, but its taxonomy is restricted to sports and its automatic tagging introduces undisclosed label noise. The authors argue that daily human activity is much broader than sports—the American Time Use Survey reports Americans average 1.7 hours per day on household activities versus 18 minutes on sports or recreation—yet benchmarks almost exclusively cover the latter. They therefore build the first activity dataset organized under a rich semantic taxonomy: the ATUS coding lexicon, which organizes more than 2000 activities along social interactions and where the activity takes place, with a hierarchy at least four levels deep (e.g., Filing nails → Washing, dressing and grooming → Grooming → Personal care).

## Dataset Construction
- **Source:** Web videos from online sharing sites (searched primarily on YouTube), downloaded at the best quality available; searches prioritize videos under 20 minutes, and in practice most videos are 5-10 minutes long. Around 50% of videos are HD (1280x720) and the majority run at 30 FPS.
- **Taxonomy:** 203 activity classes manually selected from the more than 2000 activities in the ATUS hierarchy, spanning 7 top-level categories: Personal Care, Eating and Drinking, Household, Caring and Helping, Working, Socializing and Leisure, and Sports and Exercises.
- **Collection and annotation pipeline (three stages):** (a) text-based web queries expanded with WordNet hyponyms, hypernyms, and synonyms retrieve candidate videos; (b) AMT workers verify whether the target activity is present, discarding false positives and yielding labeled untrimmed videos; (c) multiple expert turkers annotate the temporal boundaries of every activity instance, and annotations are clustered to obtain agreement, producing trimmed activity instances. An untrimmed video may contain more than one instance from more than one class.
- **Scale:** 203 classes, average 137 untrimmed videos per class, average 1.41 trimmed instances per untrimmed video, 849 total video hours. The per-class distribution of untrimmed videos and trimmed instances is close to uniform. The crowdsourced framework is designed for continuous, low-cost expansion.

## Evaluation Protocol
Three tasks are defined on the same pool of annotated videos. All use a one-vs-all linear SVM classifier over state-of-the-art features: improved trajectories (HOG, HOF, MBH encoded with Fisher vectors, GMM with 512 components), SIFT static context features (Fisher vectors, GMM with 1024 components), and AlexNet fc-6/fc-7 deep features computed every ten frames.

1. **Untrimmed video classification:** predict the set of activities in an untrimmed video; 27,801 videos across 203 classes, split 50% train, 25% validation, 25% test. Metric: mean average precision (mAP), since a video may carry multiple labels.
2. **Trimmed activity classification:** predict the label of a trimmed clip containing one activity instance; 203 classes with 193 samples per category on average. Instances from the same original video are constrained to the same subset to avoid data contamination. Metric: mAP over classes.
3. **Activity detection:** localize all activity instances (start and end frames) in untrimmed videos via sliding temporal windows (7 window lengths, step of 10 frames) with non-maximum suppression and five rounds of hard negative mining; 849 hours of video, of which 68.8 hours contain the 203 activities. A detection counts as true positive when the temporal intersection-over-union with ground truth exceeds a threshold alpha varied from 0.1 to 0.5. Metric: mAP over all classes.

## Findings and Analysis
- **Untrimmed classification:** combining motion, deep, and static features (MF+DF+SF) reaches 42.5% mAP on validation and 42.2% mAP on test; motion features alone reach 39.8%/39.2%, deep features alone 28.9%/28.7%, static features alone 24.7%/24.5%.
- **Trimmed classification:** MF+DF+SF reaches 50.5% mAP (validation) and 50.2% mAP (test); deep features alone reach 43.7%/43.0%, competitive with improved trajectories.
- **Detection:** mAP is low and drops as the overlap threshold tightens: MF+DF+SF scores 12.5% at alpha = 0.1 and 9.7% at alpha = 0.5; motion features alone score 11.7% at alpha = 0.1.
- **Cross-dataset difficulty:** with comparable state-of-the-art methods, untrimmed classification attains 71% mAP on Thumos'14 and 63.9% mAP on Sports-1M but only 42.2% mAP on ActivityNet; trimmed classification attains 85.9% accuracy on UCF101 and 66.7% on HMDB51 versus 45.9% on ActivityNet; detection attains 33.6% mAP on Thumos'14 versus 11.9% mAP on ActivityNet (alpha = 0.2).
- **Per-category analysis (trimmed classification):** Sports and exercises is easiest (66.6% validation / 66.1% test mAP) due to repetitive, structured temporal sequences; Household activities is hardest (34.2% / 33.9%) because of unstructured, temporally unconstrained execution. Hardest classes tend to be confused with activities sharing similar motions, objects, or context (e.g., Platform diving confused with Bungee jumping and Balance beam).

## Contributions
- The first large-scale human activity dataset organized under a deep semantic taxonomy (ATUS-derived), covering daily-life activities rather than only sports or short motions.
- A semi-automatic, human-in-the-loop crowdsourcing framework for continuous, low-cost collection of untrimmed videos with verified labels and temporally annotated activity instances.
- Three benchmarking protocols (untrimmed classification, trimmed classification, temporal activity detection) with baseline evaluations showing that existing state-of-the-art methods degrade sharply on ActivityNet relative to prior datasets.
- Public release of annotations, baseline models, and a toolkit at http://www.activity-net.org.

## Limitations
- Baseline performance is very low across all three tasks, and the paper attributes this to genuine dataset difficulty; it offers no method that comes close to solving the benchmark at release time.
- Temporal activity annotation is inherently ambiguous, which the authors handle only by sweeping the detection overlap threshold (alpha = 0.1 to 0.5) rather than resolving annotation uncertainty.
- Text-based web search is imprecise: the pipeline discards many retrieved videos that do not contain the intended activity, and storage constraints force the search toward videos shorter than 20 minutes.
- The first release covers 203 of the more than 2000 activities in the ATUS hierarchy, so most of the taxonomy remains unpopulated.
- Fully automatic collection alternatives were rejected for label noise, leaving the dataset dependent on paid crowdsourced verification and annotation for future expansion.
