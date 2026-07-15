# H2O (ICCV 2021)

> Kwon, Tekin, Stuhmer, Bogo, Pollefeys. *H2O: Two Hands Manipulating Objects for First Person Interaction Recognition.* ICCV 2021. DOI: 10.1109/ICCV48922.2021.00998. Zotero Key: `6PAC4ZEZ`.

## Summary
H2O is the first 3D first-person HOI benchmark that provides bi-manual 3D pose + 6D object pose + egocentric RGB-D + interaction-category labels simultaneously: about 360K frames, 4 scenes, 8 everyday objects, and 11 bi-manual action categories. It is an early authoritative benchmark for egocentric bi-manual 3D HOI reconstruction and first-person action recognition.

## 1. Dataset Purpose
- Solves the lack of "bi-manual + object 3D pose" ground truth in first-person view. While HO-3D focuses on the third view, H2O directly targets AR/VR headsets.
- Tasks: (1) bi-manual 3D pose estimation from first-person view; (2) 6D object pose estimation from first-person view; (3) interaction recognition from first-person view (action classification).
- Anchors "first-person interaction" as an independent sub-task and provides a data foundation for AR/VR headset scenarios.
- A Graph Convolutional Network baseline is proposed that considers both intra-/inter-hand and hand-object dependencies.

## 2. Data Composition
- Source: real capture. Multiple subjects manipulate everyday objects bi-manually under the first-person view.
- Viewpoint: first-person multi-view (4 synchronized RGB-D cameras, with the subject's head fixed).
- Scale: about 360K frames, 4 scenes, 36 subjects.
- Object and action: 8 everyday objects (box, milk, bottle, can, bowl, cup, phone, sponge) + 11 bi-manual actions (pass, give, take, pour, etc.).
- Contains natural occlusion, bi-manual inter-occlusion, and the self-occlusion characteristic of the first-person view.
- No specialized articulated-object design (simple open/close of box, etc.), no bi-manual tool use.

## 3. Annotation and Supervision
- Hand: bi-manual 3D 21 joints (multi-view optimization + automatic refinement); MANO shape / pose.
- Object: 6D pose (jointly optimized with hands).
- Interaction: 11 bi-manual action labels, object class.
- Scene: 4-view RGB + depth, camera intrinsics / extrinsics, point-cloud fusion, object meshes.
- No language, no robot, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) hand pose (MPJPE / PA-MPJPE, per left / right hand); (2) object pose (ADD-S / AUC); (3) interaction recognition (Top-1 / Top-5 accuracy).
- Provides unique metrics such as "inter-hand distance / inter-hand contact".
- Key metrics: bi-manual hand MPJPE, object AUC-ADDS, action classification accuracy.
- Training + evaluation + held-out split are all public.

## 5. Why It Matters
- The first public "first-person bi-manual HOI" ground-truth dataset, establishing the egocentric HOI reconstruction paradigm.
- The "graph convolutional baseline" is the first-generation bi-manual reconstruction baseline and has been cited for a long time.
- The 11 interaction labels enable first-person action recognition benchmarks on this dataset.
- H2O + HO-3D form a "first-person vs third-person" comparison, becoming the de facto reference for follow-up AR/VR research.
- Promotes AR/VR-headset hand-object tracking from single-hand to bi-manual.

## 6. Limitations and Biases
- 8 everyday objects: low diversity, methods easily overfit specific object appearances.
- Bi-manual style of 36 subjects is still affected by culture / left-right dominance.
- 11 interactions: limited action categories; labels are "actions" rather than "task goals", language is missing.
- Joint mesh ground truth of hands + objects is missing (only 6D pose + joints), so a joint mesh reconstruction benchmark cannot be performed.
- No articulated object, no tool use, no long-horizon tasks, no contact-map ground truth.
- Under the first-person view, "the subject cannot see his own hand", leading to noisier annotations on some frames.

## 7. Takeaway
H2O is best for demonstrating the accuracy of "bi-manual + object 3D pose reconstruction + action recognition" in the first-person view, especially bi-manual tracking in AR/VR headset scenarios. **Not suitable** for evaluating articulated object, tool use, long-horizon task, language-conditioned generation, or dexterous embodiment. In this survey, H2O plays the role of "first-person bi-manual 3D HOI main benchmark + first-person action recognition benchmark" and serves as a hard anchor for evaluating "video-based headset-data pretraining" in Ch5.
