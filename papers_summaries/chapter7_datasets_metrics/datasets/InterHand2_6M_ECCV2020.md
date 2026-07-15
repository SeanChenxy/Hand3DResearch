# InterHand2.6M (ECCV 2020)

> Moon, Yu, Wen, Shiratori, Lee. *InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image.* ECCV 2020. DOI: 10.1007/978-3-030-58565-5_33. Zotero Key: `NRYI9CR8`.

## Summary
InterHand2.6M is the first large-scale, single-RGB-image, bi-manual interacting (interacting hands) 3D hand pose dataset: 2.6M annotated frames covering a variety of two-person / one-person two-hand / hand-self / hand-other poses. It is the de facto standard benchmark for monocular interacting hand pose estimation.

## 1. Dataset Purpose
- Solves the problem that "existing hand pose datasets are dominated by single hands, and the data scale of interacting hands is very small".
- Tasks: 3D hand pose estimation of (a) single hand; (b) bi-manual interacting hand from a single RGB image.
- Does not contain objects; focused on hand-only bi-manual interaction ("shaking hands", "one hand holding the other", "clapping", etc.).
- The proposed InterNet baseline supports both single-hand and interacting-hand tasks.

## 2. Data Composition
- Source: real capture. Multiple people perform various hand poses under a controlled green screen + multiple RGB cameras.
- Viewpoint: 80 calibrated RGB cameras surrounding the capture space; a monocular in-the-wild evaluation set is also provided.
- Scale: 2.6M annotated frames, dozens of subjects; divided into three subsets: H (Hands, single hand) / IH (Interacting Hands) / IHO (Interacting Hands + Object).
- Action coverage: shaking, holding, crossing, pointing at, interlocking fingers, self-touching, etc.
- Object coverage: the IHO subset introduces simple everyday objects (cube, bottle, cup, etc.) but only to assist hand-pose variation.
- No 6D object pose annotation, no articulated object.

## 3. Annotation and Supervision
- Hand: bi-manual 3D 21 joints (multi-view + automatic pipeline); hand mesh obtained by MANO fitting.
- Object: the IHO subset provides 3D object mesh / 6D pose (simplified), but mainly used to assist hand localization.
- Interaction: interaction labels (self / other person / object / single hand).
- Scene: multi-view RGB, camera intrinsics / extrinsics, in-the-wild test set for cross-domain evaluation.
- No language, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) single-hand 3D pose (MPJPE / PA-MPJPE); (2) interacting-hand 3D pose (left / right hand independent MPJPE / relative MPJPE); (3) mesh error.
- Key metrics: MPJPE / PA-MPJPE / Mesh Error.
- Provides single-hand split, interacting-hand split, and in-the-wild test set; the cross-domain evaluation protocol is complete.
- The de facto evaluation standard for hand-only bi-manual interaction.

## 5. Why It Matters
- Systematizes "interacting hands" as an independent sub-task for the first time, establishing the paradigm of the direction (Moon et al.'s later InterWild and ReInterHand are all based on this).
- The 2.6M-frame scale + 80-view multi-view scheme becomes a standard reference for follow-up dual-hand datasets.
- The InterNet baseline has long been used as the reference on InterHand2.6M.
- Standard training / evaluation source for all hand-only bi-manual models (especially ReFormer, TCP-Net, HRNet-InterHand, etc.).
- In this survey, it complements FreiHAND: FreiHAND evaluates single-hand mesh, InterHand2.6M evaluates bi-manual joint / mesh.

## 6. Limitations and Biases
- Still hand-hand dominated, with no real hand-object manipulation: cannot directly evaluate "what the hand is doing".
- Green-screen background: a domain gap to in-the-wild RGB.
- The IHO subset has a small number of objects and simplified pose ground truth.
- No contact map, no affordance, no scene context.
- Annotation depends on MANO, which has limited expressiveness for severe hand-hand occlusion or special hand shapes.
- No language; the action label is weak.

## 7. Takeaway
InterHand2.6M is best for demonstrating the recovery accuracy of "bi-manual interacting pose" from monocular RGB. **Not suitable** for evaluating hand-object manipulation, bi-manual tool use, articulated objects, or in-the-wild egocentric tasks. In this survey, it plays the role of "hand-only bi-manual interaction main benchmark" and serves as the unified evaluation source for all "dual-hand mesh reconstruction methods" in Ch2 on hand-only tasks. Together with FreiHAND, it forms the two hand-only anchors of "single hand vs two hands".
