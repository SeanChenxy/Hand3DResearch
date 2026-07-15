# AssemblyHands (CVPR 2023)

> Ohkawa, He, Sener, Hodan, Tran, Keskin. *AssemblyHands: Towards Egocentric Activity Understanding via 3D Hand Pose Estimation.* CVPR 2023. DOI: 10.1109/CVPR52729.2023.01249. Zotero Key: `MBPGMKXZ`.

## Summary
AssemblyHands re-annotates 3.0M images (490K of which are first-person) of egocentric 3D hand poses from the Assembly101 videos, focused on egocentric hand-pose estimation under long-horizon, bi-manual, and complex actions (disassembling toys). It is a large-scale benchmark for egocentric 3D hand pose + action recognition.

## 1. Dataset Purpose
- Solves the bottleneck of "small scale and simple actions (mostly pick-and-place)" of existing first-person 3D hand-pose datasets.
- Tasks: (1) egocentric 3D hand pose estimation (single RGB); (2) egocentric action classification (based on hand pose).
- Emphasizes "long-horizon, complex, bi-manual" actions (disassembling toy vehicles, frequently switching hand responsibilities).
- Provides multi-view synchronization (8 external static + 4 egocentric headsets) as a strong supervision source.

## 2. Data Composition
- Source: based on Assembly101 (4,321 videos, 101 take-apart toys) re-annotated. The original Assembly101 only provides 2D keypoints, while AssemblyHands redo 3D.
- Viewpoint: 4 egocentric (head-mounted) + 8 external static cameras synchronized.
- Scale: 3.0M annotated images (of which 490K are egocentric) — the largest egocentric 3D hand pose dataset at the time.
- Action coverage: the full process of disassembling take-apart toys (vehicle models) — pressing buttons, pressing, tearing, pushing, pulling, rotating, connecting.
- Bi-manual interaction is dense; no bi-manual articulated-object manipulation (the toys themselves have no active joints).
- No contact map, no 6D object pose, no object mesh.

## 3. Annotation and Supervision
- Hand: 3D 21 joints (based on multi-view + iterative training-refinement, mean keypoint error 4.20 mm, 85% lower than the original Assembly101 annotation).
- Object: no 3D object annotation.
- Interaction: action categories come from the hierarchical action labels of Assembly101.
- Scene: multi-view RGB, camera intrinsics / extrinsics.
- No language instruction, no tactile, no robot.

## 4. Supported Evaluation
- Benchmark tasks: (1) egocentric 3D hand pose estimation (MPJPE / PA-MPJPE); (2) action classification (based on predicted hand pose).
- Key metrics: MPJPE, PA-MPJPE, action Top-1 / Top-5.
- Provides train / val / test split (based on subject + video split).
- The paper shows that "better hand pose → better action classification", proving that 3D hand pose is a key intermediate representation for action understanding.

## 5. Why It Matters
- The largest egocentric 3D hand pose dataset at the time (2023), 6× the 490K frames of H2O.
- Argues that "3D hand pose is a strong intermediate representation for action recognition" — an empirical basis for "semantic prior" in Ch4 and "motion prior" in Ch5.
- Provides a replicable pipeline for re-annotating 3D hand pose from 2D video at scale, influencing subsequent large-scale 3D hand-pose annotations such as Ego4D-Hand and HoloAssist-Hand.
- The application on long-horizon assembly / disassembly tasks inspires Ch6 research on "long-horizon robot tasks from egocentric video".

## 6. Limitations and Biases
- A single set of objects (101 take-apart toys): cross-object generalization is limited.
- No 6D object pose, no mesh, no contact map: a joint hand-object reconstruction benchmark cannot be performed.
- No articulated object, tool use, or cooking actions.
- Annotation is hand-only, with no hand-object contact ground truth.
- The training set is dominated by bi-manual, with fewer single-hand frames, so the usability for single-hand-only tasks is weak.
- Still "semi-scripted" actions of subjects executing take-apart toys, with less naturalness than in-the-wild egocentric video.

## 7. Takeaway
AssemblyHands is best for demonstrating "large-scale egocentric 3D hand pose estimation" and the transfer effect of "3D hand pose fed into action classification". **Not suitable** for evaluating hand-object 3D mesh reconstruction, 6D object pose, bi-manual contact, or language-conditioned tasks. In this survey, AssemblyHands plays the role of "egocentric 3D hand pose large-scale benchmark" and serves as a hard anchor for evaluating "semantic prior" in Ch4 and "motion prior" in Ch5 on action recognition.
