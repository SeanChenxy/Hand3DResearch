# EPIC-KITCHENS (2020)

> Damen, Doughty, Farinella, Fidler, Furnari, Kazakos, Moltisanti, Munro, Perrett, Price, Wray. *The EPIC-KITCHENS Dataset: Collection, Challenges and Baselines.* arXiv 2005.00343, 2020. Zotero Key: `PYNDZTKB`.

## Summary
EPIC-KITCHENS is a large-scale egocentric kitchen video dataset: 55 hours, 11.5M frames, 32 subjects, 4 countries, 10 nationalities, 39.6K action segments, 454.2K object bounding boxes. It is focused on "natural kitchen" first-person action recognition / object detection / action anticipation, and is a long-term flagship benchmark for egocentric action understanding.

## 1. Dataset Purpose
- Solves the problem that "existing action datasets are scripted, lack ego perspective, and lack the richness of kitchen scenarios". EPIC-KITCHENS records in the "natural home kitchen" of the subjects, providing the densest action + object + narrative annotations.
- Tasks: (1) action recognition; (2) object detection / bounding box prediction; (3) action anticipation; (4) multi-task benchmark for egocentric video understanding.
- Anchors "first-person action recognition" and "action anticipation" as independent sub-tasks.
- Complements Ego4D: EPIC's strength is the kitchen scenario + dense annotation, while Ego4D's strength is scale and scene diversity.

## 2. Data Composition
- Source: real capture. 32 subjects record in the "natural home kitchen" of 4 countries (UK, US, Portugal, Mexico).
- Viewpoint: first-person headset (GoPro, etc.).
- Scale: 55 hours, 11.5M frames; 39.6K action segments; 454.2K object bounding boxes.
- Object and action: hundreds of nouns (kitchen knife, pot, refrigerator, spoon, etc.) + verbs (open, wash, cut, pour, etc.) covering kitchen scenarios.
- Dense, fine-grained actions ("close the tap" vs "open it up").
- No 3D annotation, no contact / hand pose annotation.

## 3. Annotation and Supervision
- Video: 55 hours of raw video.
- Annotations: verb + noun + start / end time per action segment; 0~many object bounding boxes per frame.
- Annotation method: subjects narrate their own actions after recording (to ensure "true intent"); crowdsource second-round verification.
- Language: narration text (for action recognition and text-video tasks).
- No 3D annotation, no tactile, no robot.

## 4. Supported Evaluation
- Benchmark tasks: (1) Action Recognition (verb / noun / action Top-1 / Top-5); (2) Action Anticipation (future verb / noun / action Top-1); (3) Object Detection (mAP).
- Key metrics: Top-1 / Top-5 / mean Top-5 recall, mAP.
- Provides seen / unseen kitchen split to test cross-kitchen generalization.
- Multiple baselines (TSN, SlowFast, Temporal Segment Networks, etc.) have long been used as SOTA references.

## 5. Why It Matters
- The pioneer of the "natural kitchen + narration annotation" paradigm, and has long been the standard evaluation for egocentric action recognition.
- 55 hours + 39.6K action segments / 454.2K bounding boxes were the largest scale in egocentric kitchen at the time.
- The multi-baseline protocol makes SOTA comparison reproducible.
- The coverage of 4 countries and 10 nationalities provides early "cross-cultural egocentric behavior" data.
- The core anchor of the "semantic prior" in Ch4 and the "video-based pretraining" in Ch6.
- Inspired subsequent large-scale egocentric datasets such as Ego4D, Ego-Exo4D, and EgoSchema.

## 6. Limitations and Biases
- No 3D annotation: not directly comparable to 3D benchmarks such as HO-3D v3 and ARCTIC.
- Single scene (kitchen only): narrow scene coverage, complementary to the multi-scene Ego4D.
- Fixed noun / verb vocabulary: limits open-vocabulary capability.
- Narration annotation is recalled after the fact and may deviate slightly from the real action.
- The 4 countries are still dominated by Europe and the US, with limited cultural coverage.
- No contact map, no affordance, no 6D object pose, no articulated 4D reconstruction.

## 7. Takeaway
EPIC-KITCHENS is best for demonstrating the capability of "egocentric kitchen action recognition / object detection / action anticipation", especially the recognition of fine granularity (verb-noun pair). **Not suitable** for evaluating 3D hand pose, 6D object pose, joint hand-object reconstruction, cross-scene tasks, or articulated HOI. In this survey, EPIC-KITCHENS plays the role of "egocentric kitchen action understanding flagship benchmark + video-based pretraining source" and serves as a hard anchor for multiple chapters including Ch4 and Ch6.
