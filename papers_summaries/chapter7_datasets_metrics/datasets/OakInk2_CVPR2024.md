# OakInk2 (CVPR 2024)

> Zhan, Yang, Zhao, Mao, Xu, Lin, Li, Lu. *OakInk2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion.* CVPR 2024. DOI: 10.1109/CVPR52733.2024.00050. Zotero Key: `QVCEJ5AW`.

## Summary
OakInk2 is the bi-manual + complex-task expansion of OakInk: focused on 3D bi-manual HOI reconstruction and generation under "bi-manual + multi-step task" scenarios, providing task-level task labels, bi-manual affordance, and long-term sequences (each task contains multiple sub-steps on average). It is a composite benchmark for bi-manual complex-task HOI.

## 1. Dataset Purpose
- Solves the limitation of "OakInk v1 only has single-hand / simple grasping". OakInk2 directly addresses bi-manual + multi-step tasks.
- Tasks: (1) bi-manual 3D hand pose estimation; (2) bi-manual 6D object pose estimation; (3) task-level HOI generation (bi-manual affordance + task progress); (4) bi-manual handover and tool use.
- Provides task-level semantic labels (task name / sub-step / affordance chain), supporting "task-level HOI evaluation".
- Complements ARCTIC: ARCTIC's strength is articulated-object 4D mesh, while OakInk2's strength is task-level semantics.

## 2. Data Composition
- Source: mocap + object scanning, in the same studio as OakInk. Bi-manual subjects perform predefined tasks.
- Viewpoint: mocap coordinates, no RGB.
- Scale: significantly expanded compared to OakInk v1 (specific numbers to be confirmed in the paper; Zotero only has the title + DOI).
- Object and action: household, kitchen utensils, tools, and other categories; tasks cover open a box, use screwdriver, pour from bottle, fold cloth, cut food, etc., with 50+ task categories.
- Each task contains 5–15 sub-steps on average, and the sequence length can reach tens of seconds.

## 3. Annotation and Supervision
- Bi-manual: 3D 21 joints × 2 hands, MANO β / θ, bi-manual SMPL-X mesh.
- Object: 6D pose, 3D mesh, affordance region.
- Interaction: task-level label (50+ tasks), sub-step label, intent label, contact map (hand-object + object-object).
- Scene: mocap coordinates; no RGB.
- No language instruction (but task labels can serve as a language proxy), no robot, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) bi-manual 3D hand pose (MPJPE / Mesh Error); (2) bi-manual 6D object pose; (3) task-level HOI generation (task completion rate / step accuracy); (4) bi-manual affordance prediction.
- Key metrics: hand MPJPE, object pose error, task completion rate, step prediction Top-1.
- Provides task-level + sub-step-level split, evaluating long-horizon task completion.
- Cross-subject split: subjects can be left out for evaluation.

## 5. Why It Matters
- The first dataset that takes "bi-manual + complex task + multi-step" as a unified evaluation dimension.
- Task-level + sub-step-level labels inspire subsequent HOI generation papers to take "task progress" as the generation target.
- Promotes bi-manual HOI from "short-term grasping" to "long-term tasks".
- Can be used as "task-level HOI simulation ground truth" in the Ch6 "robot learning" section.
- Complements ARCTIC: ARCTIC measures 4D mesh, OakInk2 measures task-level semantics.

## 6. Limitations and Biases
- Still mocap-only: no RGB annotation, RGB-based vision models cannot be directly trained.
- Limited task set (~50 tasks): long-horizon task diversity is limited.
- No specialized articulated-object joint-angle design (only basic actions such as "open/close").
- No language instruction (only task labels), limiting direct VLA application.
- No tactile, no force, no dynamic contact annotation.
- The annotation pipeline shares with OakInk v1, and some systematic biases remain.

## 7. Takeaway
OakInk2 is best for demonstrating the capability of "bi-manual + multi-step task HOI generation", especially task-level completion rate and sub-step prediction. **Not suitable** for evaluating RGB-based vision tasks, articulated 4D mesh reconstruction, language-conditioned, or in-the-wild egocentric tasks. In this survey, OakInk2 plays the role of "task-level bi-manual HOI main benchmark" and serves as the hard anchor for evaluating "structured HOI supervision" in Ch6 for imitation learning / policy learning.
