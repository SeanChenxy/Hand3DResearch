# Ego4D (CVPR 2022)

> Grauman et al. (large consortium). *Ego4D: Around the World in 3,000 Hours of Egocentric Video.* CVPR 2022. Zotero Key: `83EEQPQR`.

## Summary
Ego4D is the largest-scale, multi-scene, multi-cultural first-person video dataset to date: 3,000+ hours, 74 locations, 9 countries, 9 daily-life scenarios (household, outdoor, workplace, etc.), annotated with 5 major tasks (episodic memory, forecasting, hand-object, social, narration). It is the "ImageNet" of egocentric video understanding.

## 1. Dataset Purpose
- Solves the fundamental problem of "small scale, single country, single scenario" of existing egocentric datasets. Ego4D is constructed as a new-generation visual foundation dataset for egocentric video.
- Tasks: 5 major benchmarks — (1) Episodic Memory; (2) Future Hand Prediction; (3) Hand-Object Interaction; (4) Social Interaction; (5) Audio-Visual Narration.
- Anchors "egocentric video understanding" as an independent sub-task, enabling "learning world models from first-person human data".
- Complements EPIC-KITCHENS and Assembly101: Ego4D's strength is scale and scene diversity, while EPIC's strength is the density of action / object labels.

## 2. Data Composition
- Source: real capture. 74 locations, 9 countries, 4 continents, 9 daily-life scenarios (home, workplace, outdoor, leisure, etc.).
- Viewpoint: first-person headset (multiple models: Pupil Labs, Vuzix, etc.).
- Scale: 3,000+ hours of video, 74 locations, 9 countries; about 200 subjects.
- Object and action: full range of daily scenarios (home, office, outdoor, supermarket, restaurant, etc.); full range of actions.
- No fine annotations such as 3D objects / hands / contact / 6D pose (focused on the "video + task" level).

## 3. Annotation and Supervision
- Video: 3,000+ hours of raw video + metadata (GPS, IMU, audio).
- Annotations: 5 major task labels — episodic memory queries, forecasting targets, HOI segments, social interaction, narration.
- 3D information: none (focused on 2D video).
- Hand: 2D hand mask provided in the future hand prediction sub-task.
- Object: object category / bounding box provided in the narration sub-task.
- Interaction: narration (natural language description) + action label.

## 4. Supported Evaluation
- Benchmark tasks: each of the 5 sub-tasks has its own specialized evaluation protocol.
- Key metrics: depending on the sub-task — retrieval mAP for episodic memory, hand mask AP for forecasting, mAP for HOI, IoU for social, BLEU / METEOR for narration.
- Provides standard train / val / test split.
- The core data source for egocentric foundation model training.

## 5. Why It Matters
- The largest-scale egocentric video dataset at the time (2022), 50× the scale of EPIC-KITCHENS.
- The coverage of 9 countries and 74 locations makes "cross-cultural egocentric behavior" evaluation possible.
- The 5-major-task design makes Ego4D the de facto platform for "egocentric pretraining + multi-task finetune".
- Inspired subsequent "extended Ego4D" datasets such as Ego-Exo4D and EgoSchema.
- The flagship data source of the "video generative prior" in Ch5 and the "video-based pretraining" in Ch6.

## 6. Limitations and Biases
- Lack of 3D annotation: no hand pose, 6D object pose, scene mesh, etc.; complements 3D benchmarks such as HO-3D v3, ARCTIC, and HOT3D but does not overlap.
- Sparse labels: narration etc. is crowdsourced, and the quality varies.
- The 9 countries are still dominated by North America and Europe, with weak coverage of Africa, South Asia, etc.
- Headset-specific: the camera quality of different headsets varies greatly, affecting the generalization of vision models.
- Not specifically oriented to "fine-grained hand-object interaction": the HOI task is coarse-grained (segmentation level).

## 7. Takeaway
Ego4D is best for demonstrating the capability of "large-scale egocentric video understanding + multi-task foundation model training". **Not suitable** for evaluating 3D hand pose, 6D object pose, joint hand-object reconstruction, articulated 4D, or in-studio high-precision tasks. In this survey, Ego4D plays the role of "egocentric video foundation dataset + multi-task HOI pretraining source" and serves as the core anchor of the "video generative prior" in Ch5 and the "video-based pretraining" in Ch6. **The abstract / DOI can be further refined after the Zotero metadata is supplemented.**
