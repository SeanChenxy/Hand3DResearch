# GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities

## Summary
GigaHands is a large-scale, markerless, multi-view bimanual hand-activity dataset (34 hours, 14k motion clips, 84k text descriptions, 183M images from 51 camera views, 56 subjects, 417 objects) with a procedural Instruct-to-Annotate pipeline that minimizes manual labeling while preserving verb diversity, paired with fully automatic 3D hand and object estimation, text-driven motion synthesis, motion captioning, and dynamic radiance field reconstruction.

## 1. Problem and Setting
- Task: Provide a diverse, large-scale, fully-annotated 3D bimanual hand-activity corpus for text-driven motion synthesis, motion captioning, hand-object reconstruction, and dynamic 3D reconstruction, overcoming the limited scale, coverage, and annotation depth of existing hand-activity datasets.
- Input: Multi-view RGB video (51 cameras, 1280×720, 30 fps) of bimanual hand activities; procedural activity instructions generated from a verb pool + scenarios; object templates (multi-view scans or single-view reconstructions) for 3D object tracking.
- Output: Per-frame 2D and 3D hand keypoints; MANO hand meshes (both hands); 3D object shape, 6D pose, and appearance; hand/object segmentation masks; camera pose (intrinsics + extrinsics from COLMAP with fiducial markers); atomic-level text descriptions (84k text annotations, 1,467 unique verbs).
- Span: Bimanual hand activities, including hand-object interaction (blue), gestures (orange), and self-interaction (red). Continuous 3D poses over time, captured from 51 camera views.
- Why difficult: In-the-wild data is sparse, hard to calibrate, and noisy; marker-based studio capture inhibits natural motions like self-contact; hand and object annotations are labor-intensive; capturing both hands and objects with high accuracy at scale is a long-standing bottleneck.

## 2. Core Method
GigaHands builds on a four-part pipeline.

**Procedural instruction elicitation.** Atomic actions are sourced from Ego4D, Ego-Exo4D, OakInk-2, TACO; verbs are extracted and grouped into a pool, then combined with objects into 5 scenarios (cooking/eating, office, crafting, entertainment, housework) and 25 scenes; an LLM organizes scenes into 191 activities and 1,370 instructions containing 533 verbs. Result: 5 scenarios → 25 scenes → 191 activities → 1,370 instructions.

**Filming with a markerless multi-camera rig.** 51 RGB cameras in a 3×3 grid per face of a cubic capture volume, LED illumination, transparent glass tabletop, software-synchronized at <3 ms temporal misalignment; COLMAP with fiducial markers for intrinsics/extrinsics. Instructions are played as audio sequentially; if the ending state of one action does not align with the next, a corrective instruction is recorded.

**Action annotation and augmentation.** The 13k motion sequences are split into 14k motion clips; annotators correct unscripted motions and LLM hallucinations. Each description is rephrased 5× by an LLM, expanding 14k clips into 84k motion-text pairs with 1,467 unique verbs.

**Hand and object motion estimation.**
- Hand pipeline (fully automatic): YOLOv8 hand bounding boxes; HaMeR for MANO mesh initialization; ViTPose for handedness; multi-view triangulation of 2D keypoints for 3D positions; one-euro filter for temporal smoothness; EasyMoCap MANO fitting.
- Object pipeline: DINOv2 detects salient objects at 1 fps; multi-view rendered template meshes select top-k boxes via Grounding DINO; OpenCLIP filters false positives; SAM2 segments object masks; differentiable rendering supervised by multi-view masks with Instant-NGP translation initialization and FoundPose-style multi-rotation DINOv2 rotation initialization, yielding precise 6D object pose.

**Demonstrated applications.**
- Text-driven hand motion synthesis: T2M-GPT backbone, R-Precision, MM Dist, FID, Diversity, Multimodality, motion VAE feature extractor.
- Motion captioning: TM2T backbone; evaluation with R-Precision, MM Dist, BLEU, ROUGE, BERTScore, distinct-n, Pairwise BLEU; transfer to in-the-wild datasets.
- Dynamic radiance field reconstruction: 2DGS fitted per frame with previous-frame initialization, 38 of 51 views used for training, 1 test view held out; segments hands and objects from Section 4.5.

## 3. Knowledge, Supervision, and Assumptions
- Training data: 56 subjects, 417 objects, 13k motion sequences, 14k clips, 51 camera views, 183M RGB frames (366M unique hand images).
- Supervision signals: 2D/3D hand keypoints from multi-view triangulation; MANO hand mesh parameters (shape + pose) from EasyMoCap fitting; 6D object pose and shape from differentiable rendering with multi-view masks; LLM-generated atomic-level text descriptions; procedural activity instructions.
- Domain knowledge: t-SNE pose/motion diversity analysis; verb-coverage diversity via UpSet plots; 2-cm contact threshold (in contact-region analysis via [101]); MANO parametric hand model; COLMAP for camera calibration; LLM-driven scenario structuring (GPT-4 mentioned in references).
- Foundation models used: YOLOv8 (detection), HaMeR (hand mesh init), ViTPose (handedness), DINOv2 (salient object detection, rotation init), Grounding DINO (box selection), OpenCLIP (false-positive filtering), SAM2 (segmentation), Instant-NGP (radiance field for translation init), GPT-4 (verb grouping, instruction generation, rephrasing), T2M-GPT (text-to-motion), TM2T (motion captioning), 2DGS (radiance field reconstruction).
- Assumptions: (i) a controlled studio + procedural instruct-to-annotate can approximate in-the-wild diversity; (ii) markerless capture is accurate enough for 3D hand+object reconstruction when dense multi-view is available; (iii) LLM-generated text descriptions are good enough for training text-driven motion models; (iv) text descriptions from one dataset (GigaHands) can transfer to captioning motions in other datasets (TACO, OakInk2).

## 4. Experiments and Findings
- Scale comparison (Table 1). GigaHands: 2,034 mins, 13.9k motion clips, 3.7M hand poses, 51 views, 183M frames, 56 subjects, 417 objects — vs AssemblyHands (630/62/203k/12/3.03M/34), HOI4D (1,333/4k/1.2M/1/2.4M/4/800), ARCTIC (121/339/218k/9/2.1M/10/11), TACO (202/2.3k/363k/13/4.7M/14/196), OakInk2 (557/2.8k/993k/4/4.01M/9/75), HOT3D (833/4.1k/1.7M/2-3/3.7M/19/33).
- Verb diversity (Figure 2 right). 1,467 unique verbs in GigaHands; 580 verbs exclusive to it, more than any other hand dataset, including in-the-wild ones.
- Text-driven motion synthesis (Table 2). T2M-GPT trained on GigaHands: Top-1 R-Prec 31.2, Top-2 44.7, Top-3 53.1, MM Dist 6.68, FID 4.70, Diversity 10.5, Multimodality 9.11 — outperforming TACO (Top-1 18.9, MM Dist 7.39, FID 11.0) and OakInk2 (Top-1 17.9, MM Dist 7.75, FID 19.6). GigaHands wins all metrics except MM Dist.
- Effect of data scale (Figure 6). T2M-GPT trained on 10/20/50/80/100% of training data — FID, MM Dist, Top-1, Top-3 all improve as data grows.
- Motion captioning (Tables 3, 4). TM2T trained on GigaHands: Top-1 R-Prec 57.0, Top-2 66.1, Top-3 69.8, MM Dist 5.37; BLEU@4 43.1, ROUGE 57.7, distinct-1 15.3, distinct-2 36.9, BERTScore 55.4, Pairwise-BLEU 0.916 — outperforms TACO and OakInk2 on most metrics, and successfully captions in-the-wild TACO / OakInk2 motions.
- Qualitative (Figures 5, 7, 8). GigaHands-trained models generate diverse motions from a single text; models trained on GigaHands also caption motions from TACO and OakInk2. 2DGS reconstructions on held-out test views (Figure 8) demonstrate dynamic radiance field fitting across timesteps.
- Contact diversity (Figure 3 right). Accumulated contact regions span both front and back of both hands (the right hand contacts objects more often, consistent with a majority of right-handed subjects).

## 5. Strengths and Limitations
### Strengths
- Largest 3D bimanual hand-activity dataset to date, with 51 camera views enabling dense multi-view reconstruction, hand+object tracking, and dynamic radiance field fitting.
- Instruct-to-Annotate pipeline reduces manual annotation cost while maximizing verb diversity (1,467 unique verbs, 580 exclusive).
- Fully automatic 3D hand and object estimation pipelines (YOLOv8 → HaMeR → ViPpose → multi-view triangulation → EasyMoCap for hands; DINOv2 → Grounding DINO → OpenCLIP → SAM2 → differentiable rendering with FoundPose-style initialization for objects).
- Empirical gains: better text-driven motion synthesis, better motion captioning, and transfer to in-the-wild datasets (TACO, OakInk2) without retraining; data-scale ablation shows monotonic improvement.

### Limitations
- Studio setting limits spatial coverage; motions requiring larger environments cannot be captured accurately.
- Fully automatic tracking works for rigid objects and object parts, but articulated and non-rigid objects (e.g., pants) remain difficult; the paper acknowledges this and shows 2DGS reconstruction as a workaround for some non-rigid cases.
- LLM-generated instructions can contain inconsistencies or hallucinations, requiring manual correction during annotation.
- Right-hand bias: a majority of subjects are right-handed, leading to asymmetric contact-region statistics.
- The studio does not reflect real-world context (e.g., varied lighting, backgrounds, occlusions) despite the procedural instruct-to-annotate design.
- Robotics and HCI applications are discussed as future work rather than evaluated directly.

## 6. Takeaway
GigaHands demonstrates that the long-standing bottleneck in scaling bimanual hand-activity data can be broken by combining a markerless dense multi-view capture system, a procedural Instruct-to-Annotate protocol, and a fully automatic hand+object estimation pipeline, producing a dataset whose scale, verb diversity, and 3D annotation depth simultaneously advance text-driven hand motion synthesis, motion captioning (including transfer to in-the-wild datasets), and dynamic 3D reconstruction — shifting bimanual activity research from object-centric narrowness to broad, text-annotated, multi-view 3D supervision.
