# Something-Something (ICCV 2017)

> Goyal, Kahou, Michalski, Materzynska, Westphal, Kim, Haenel, Fruend, Yianilos, Mueller-Freitag, Hoppe, Thurau, Bax, Memisevic. *The "Something Something" Video Database for Learning and Evaluating Visual Common Sense.* ICCV 2017. DOI: 10.1109/ICCV.2017.622. Zotero Key: `EP38D6RH`.

## Summary
Something-Something is a large-scale "everyday physical action" egocentric video dataset: 100K+ videos, 174 action template classes ("pushing something from X to Y", "putting X next to Y", etc.), focused on "general physical common sense" visual understanding. It is a long-term flagship benchmark for video action recognition and temporal reasoning.

## 1. Dataset Purpose
- Solves the problem that "existing action recognition datasets are biased towards 'what is happening' rather than 'how it physically happens'". Something-Something takes "general physical action templates" as the core label space.
- Tasks: (1) action recognition (fine-grained physical actions); (2) temporal reasoning; (3) physical common sense learning; (4) video pretraining.
- Anchors "temporal physical reasoning" as an independent sub-task.
- It is one of the long-term standard evaluations in the video understanding field (alongside Kinetics and UCF-101).

## 2. Data Composition
- Source: crowdsourcing (Amazon Mechanical Turk). Subjects perform simple physical operations according to templates and record.
- Viewpoint: egocentric (subject hand-held / head-mounted recording).
- Scale: v2 contains 100K+ videos, 174 action classes.
- Object and action: everyday items (pen, paper, cup, ruler, etc.) with push, pull, drop, put, slide, fold, and other physical actions.
- Dense actions, focused on "temporal ordering" and "physical relations".

## 3. Annotation and Supervision
- Video: 100K+ egocentric videos.
- Annotations: each video has one class label (one of 174 template classes).
- 3D information: none.
- Hand: no hand annotation.
- Object: object category label (weak).
- Interaction: action class template.

## 4. Supported Evaluation
- Benchmark tasks: (1) action recognition (Top-1 / Top-5); (2) zero-shot action understanding; (3) video pretraining.
- Key metrics: Top-1 / Top-5 accuracy.
- Provides train / val / test split.
- It has long been used as the standard evaluation for video transformer / temporal CNN.

## 5. Why It Matters
- The pioneer of the "general physical action template" paradigm, promoting the shift of video understanding from object-centric to physics-centric.
- 100K+ videos + 174 classes were a large scale in video recognition at the time (2017).
- Inspired a large amount of video pretraining work (TimeSformer, Video Swin, etc.).
- Complements EPIC-KITCHENS: EPIC's strength is the kitchen scenario, while Something-Something's strength is "general physical common sense".
- A classic anchor of the "video generative prior" in Ch5 and the "video-based pretraining" in Ch6.

## 6. Limitations and Biases
- The object set is simple (pen, paper, cup, etc.): it cannot evaluate the fine-grained reconstruction of hand-object.
- No 3D annotation: not directly comparable to HO-3D v3, ARCTIC, etc.
- No hand pose, 6D object pose, or contact annotation.
- Object bounding box / segmentation is missing.
- The crowdsourcing quality varies, and some videos have problems such as the hand leaving the lens.
- No language instruction (only action labels).

## 7. Takeaway
Something-Something is best for demonstrating the capability of "general physical action recognition + temporal reasoning". **Not suitable** for evaluating 3D hand pose, 6D object pose, joint hand-object reconstruction, articulated 4D, bi-manual tool use, or in-the-wild egocentric complex tasks. In this survey, Something-Something plays the role of "general physical action understanding benchmark" and serves as the foundation anchor of Ch5 / Ch6 (complementary to EPIC-KITCHENS).
