# HumanNet: Scaling Human-centric Video Learning to One Million Hours

**Authors:** Yufan Deng, Daquan Zhou  
**Date:** 2026-05-07  
**Identifier:** [arXiv:2605.06747](https://arxiv.org/abs/2605.06747)  
**Zotero item:** `XHAEMSJE` ([Zotero](zotero://select/library/items/XHAEMSJE))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HumanNet is a human-centric video dataset of one million hours (967K hours in the headline statistics figure) of first-person and third-person footage, covering 150K+ objects and 720K+ tasks, collected and processed through a three-stage pipeline (collection, processing, annotation) that produces captions, hierarchical labels, motion descriptions, hand and body signals, and robot-ready subsets. The paper's central claim, validated with vision-language-action (VLA) post-training experiments on a LingBot-VLA architecture, is that 1,000 hours of egocentric HumanNet video can match or modestly surpass 100 hours of real-robot data for robot manipulation pretraining, positioning internet-scale human video as a scalable alternative to costly robot data collection.

## Background and Motivation

Robot learning is bottlenecked by the cost of collecting real-robot interaction data, while the internet contains vast amounts of video of humans performing manipulable tasks. Prior video datasets are orders of magnitude smaller and lack the annotations needed to convert human video into robot-actionable supervision. The authors define four design principles for embodied-ready human video: Scale (orders of magnitude more hours), Viewpoint diversity (egocentric and exocentric, since egocentric footage gives hand-centric manipulation signal and exocentric footage gives whole-body motion), Physical relevance (footage of interactions with the physical world rather than talking heads), and Pretraining readiness (annotations formatted for VLA and vision-language-model training). HumanNet is constructed to satisfy all four, with explicit bridges from human video to robot skills: egocentric video is converted into robot motion through 3D hand pose, while exocentric video is converted through motion retargeting.

## Dataset Construction

Construction follows three stages. Collection combines seed keyword expansion, keyword and channel crawling, platform and web search, integration of open datasets, and self-collection. Processing applies deduplication and normalization, content filtering (removing unsafe or privacy-sensitive material, with a privacy review embedded in the pipeline), quality filtering, scene splitting, and clipping into short clips. Annotation uses 3D hand and body pose detection, monocular SLAM, motion retargeting onto a unified humanoid skeleton, and LLM-assisted captioning that produces short and long video captions, hierarchical category labels (for example, Sports > Basketball > Defense), motion descriptions, hand and body metadata, and multi-label interaction categories. A clip is designated robot-ready when the retargeting error remains below 15 mm and valid-frame coverage exceeds 60%. The released statistics figure reports 967K hours of video, 150K+ objects, and 720K+ tasks, spanning both egocentric and exocentric viewpoints.

## Evaluation Protocol

The validation is a vision-language-action post-training study rather than a dataset benchmark. Four model variants sharing the LingBot-VLA architecture vary only the pretraining data source: a Qwen-based VLM without embodied pretraining; the same Qwen VLM adapted with 100 hours of real-robot CoBot data; a Qwen VLM adapted with 1,000 hours of egocentric video drawn from HumanNet; and LingBot, whose Qwen backbone is trained with 20,000 hours of real-robot data. All variants are then post-trained on the same protocol of 100 tasks with 20 episodes per task, totaling 34 hours of robot interaction data, and evaluated on validation task groups covering manipulation skills.

## Findings and Analysis

Continued training with 1,000 hours of egocentric HumanNet video surpasses continued training with 100 hours of real-robot Magic Cobot data, and the 1,000-hour egocentric initialization matches, and on several task groups slightly exceeds, the model initialized from 100 hours of real-robot CoBot data after identical post-training. The 1,000-hour egocentric model narrows but does not close the gap to the 20,000-hour real-robot LingBot baseline. The authors read these results as substantiating the central claim that egocentric human video is a more scalable pretraining source than real-robot data, with the remaining gap to large-scale robot data attributable to the residual embodiment difference between human video and robot embodiments.

## Contributions

A million-hour-scale human-centric video dataset unifying egocentric and exocentric sources with a reproducible collection-processing-annotation pipeline; an annotation suite targeted at embodied learning (captions, hierarchical labels, motion descriptions, hand and body signals, multi-label interaction categories); a retargeting-based definition of robot-ready video subsets with quantitative thresholds (retargeting error below 15 mm and valid-frame coverage above 60%); and a controlled VLA post-training study quantifying how egocentric pretraining substitutes for real-robot pretraining data.

## Limitations

The paper includes an explicit limitations discussion covering: the embodiment gap between human video and robot platforms, which caps transfer from human to robot motion; noise at scale, since web video and automated annotation introduce label and quality noise that filtering cannot fully remove; uneven coverage and bias across geography, socioeconomic context, and viewpoint; privacy and safety concerns inherent to large-scale human video, addressed through pipeline-level review but not eliminable; and dual-use impact of human-centric models trained on such data.
