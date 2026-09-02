# HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World

**Authors:** Xin Wang, Taein Kwon, Mahdi Rad, Bowen Pan, Ishani Chakraborty, Sean Andrist, Dan Bohus, Ashley Feniello, Bugra Tekin, Felipe Vieira Frujeri, Neel Joshi, Marc Pollefeys  
**Date:** 2023-09-29  
**Identifier:** [arXiv:2309.17024](https://arxiv.org/abs/2309.17024); DOI `10.1109/ICCV51070.2023.01854`  
**Zotero item:** `ALJS7VKI` ([Zotero](zotero://select/library/items/ALJS7VKI))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HoloAssist is a 166-hour egocentric dataset of 350 instructor-performer collaboration pairs (222 participants, 2,221 recording sessions) spanning 20 physical tasks, recorded with Microsoft HoloLens 2 and annotated with synchronized conversation, action, mistake, and intervention labels. Seven synchronized sensor streams (RGB, depth, head pose, 3D hand pose, eye gaze, audio, IMU) are captured per session. The authors derive benchmarks for action recognition, mistake detection, intervention type prediction, and 3D hand pose forecasting, establishing baselines for building AI assistants that monitor human activity and intervene at the right moment.

## Background and Motivation

Interactive AI assistants that help humans with physical tasks require understanding not only what the user is doing but also when the user errs and when the assistant should speak. Existing datasets either lack egocentric multi-modal sensing, lack physical task interactions, or contain no annotations of mistakes and conversational interventions. The authors therefore record natural instructor-performer collaborations in which one participant executes a task and the other guides, observes, and corrects, so that data contain rich examples of verbal guidance, error recognition, and timely intervention in the real world.

## Dataset Construction

Sessions were recorded with HoloLens 2 headsets on both collaborators and streamed through the Platform for Situated Intelligence, yielding seven synchronized modalities plus transcribed text. The corpus covers 20 physical tasks involving 16 objects grouped into small, medium, big, and rare categories, performed by 350 instructor-performer pairs (222 unique participants) over 2,221 sessions totaling 166 hours. Annotations include text summaries of each session; conversations segmented with purpose and initiator labels; action annotations at two granularities, 414 coarse actions (90 nouns, 39 verbs) and 1,887 fine-grained actions (165 nouns, 49 verbs); per-action mistake labels, with roughly 6% of actions containing a mistake; and three types of intervention labels for how the instructor responds to errors. Sessions are split 70/10/20 at the session level (1,545 / 213 / 463 sessions), ensuring participant-level separation between splits.

## Evaluation Protocol

Four benchmarks use TimeSformer-based baselines trained from scratch and fine-tuned from pre-trained weights. Action recognition is evaluated at coarse and fine granularity on trimmed clips, reporting top-1 and top-5 accuracy. Mistake detection evaluates whether a model can flag actions containing errors from the performer's egocentric stream, with variants using hands-only or multi-modal input, scored by F-score. Intervention type prediction classifies which of three intervention types follows a mistake, using combinations of RGB, hand, and eye-gaze modalities, and is scored by precision and recall. 3D hand pose forecasting conditions on 3 seconds of past hand poses and predicts 0.5, 1.0, and 1.5 seconds ahead, scored by mean joint error in centimeters.

## Findings and Analysis

Fine-grained action recognition from egocentric video is far from solved: the best fine-grained model reaches about 35% top-1 accuracy and about 50% on coarse actions. Mistake detection achieves an F-score of 40.19 with hands-only input, indicating that hand motion carries substantial but incomplete error signal. Intervention prediction reaches 48.31% precision and 37.59% recall with RGB, hand, and eye gaze combined, and the analysis of conversation data shows that instructors intervene immediately (within 5 seconds) for mistakes on linear tasks, whereas for non-linear tasks they often wait for self-correction or intervene lazily; spatial deixis ("here", "there") is frequent and challenging for grounding. Hand pose forecasting yields 9.80, 10.68, and 11.25 cm error at 0.5, 1.0, and 1.5 seconds respectively, establishing a reference point for anticipating manipulation from egocentric streams.

## Contributions

A large-scale, multi-modal, egocentric dataset of real-world instructor-performer collaboration with synchronized seven-stream sensing and unique mistake and intervention annotations; taxonomies of conversation purposes, coarse and fine actions, mistakes, and interventions; four benchmark tasks (action recognition, mistake detection, intervention prediction, hand pose forecasting) with TimeSformer baselines; and an analysis of when and how humans intervene during physical collaboration.

## Limitations

The paper states that object 6D pose is not annotated, which prevents full 3D scene reconstruction and object-centric reasoning benchmarks; the authors list object pose annotation as future work. Performance levels of the provided baselines (around 35-50% action recognition accuracy and moderate mistake detection scores) also demonstrate that the dataset exposes substantial open challenges rather than saturating them.
