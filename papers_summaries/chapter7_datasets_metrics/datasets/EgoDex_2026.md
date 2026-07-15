# EgoDex (2026)

> Hoque, Huang, Yoon, Sivapurapu, Zhang. *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video.* arXiv 2505.11709, 2026. Zotero Key: `6LDCN9MU`.

## Summary
EgoDex is a "largest scale + high-precision 3D finger tracking" real human-hand operation dataset collected by Apple with Vision Pro: 829 hours of egocentric video, 3D finger joint poses per frame, 194 desktop tasks. It is the flagship dataset for imitation learning of dexterous manipulation.

## 1. Dataset Purpose
- Solves the fundamental problem that "existing egocentric data lacks 3D hand / finger ground truth". Apple Vision Pro's built-in SLAM + multi-camera + infrared finger tracking enables large-scale high-precision finger-tracking annotation.
- Tasks: imitation learning for hand trajectory prediction; dexterous manipulation policy learning; hand pose estimation; 3D hand mesh reconstruction.
- Directly oriented to the downstream application of "training robot / virtual hand manipulation policies with egocentric video".
- Anchors "Apple Vision Pro + egocentric + dexterous" as a new-generation HOI data-collection paradigm.

## 2. Data Composition
- Source: real capture. Multiple subjects use Apple Vision Pro headsets to perform desktop operation tasks.
- Viewpoint: first-person multi-camera (multiple SLAM cameras + infrared finger sensors built into the Vision Pro headset).
- Scale: 829 hours of egocentric video, 3D finger joint poses (26+ DOF × 2 hands) synchronized per frame.
- Object and action: 194 desktop tasks (tying shoelaces, folding laundry, stacking cups, writing, etc.); objects include daily household items.
- Contains natural 26-DOF hand motion, bi-manual cooperation, and in-hand manipulation.

## 3. Annotation and Supervision
- Hand: 3D finger joint poses (26+ DOF × 2 hands per frame), from the Vision Pro built-in sensor (sub-cm accuracy).
- Object: object category labels; no 6D pose annotation.
- Interaction: task label (194 categories), sub-step label (partial).
- Scene: first-person RGB-D (from Vision Pro sensors); scene mesh (partial).
- Some sequences provide language instruction (sub-task description).

## 4. Supported Evaluation
- Benchmark tasks: (1) hand trajectory prediction (next-frame 3D finger joints); (2) imitation learning policy (successfully executing 194 desktop tasks); (3) hand pose estimation (MPJPE / PA-MPJPE).
- Key metrics: trajectory MSE, policy success rate, hand MPJPE.
- Provides standard train / val / test split (by subject + task).
- Introduces metrics and benchmark protocols, making the evaluation of "egocentric → manipulation policy" reproducible.

## 5. Why It Matters
- 829 hours + 3D finger tracking = the largest-scale egocentric + high-precision hand-tracking dataset at the time.
- For the first time, "commercial headset (Vision Pro) → large-scale 3D HOI data" is established as a reproducible paradigm, expected to drive the release of similar datasets from Meta Quest 3, Pico, and other headsets.
- Directly provides a "high-quality human demonstration" source for imitation learning / VLA training, connecting the HOI community and the robot learning community.
- The diversity of 194 tasks makes "generalist manipulation policy" evaluation possible.
- The flagship anchor of "structured HOI supervision" in Ch6 "robot learning".

## 6. Limitations and Biases
- Desktop tasks dominate: no mobile manipulation, whole-body, or navigation.
- Although 194 tasks are many, the average sample per task is still limited relative to the 829-hour magnitude.
- Under the Vision Pro headset view, "the subject cannot see his own hand", causing partial hand occlusion; however, the Vision Pro's built-in infrared can improve this from the finger side.
- The object category label is coarse, with no 6D pose / mesh / contact annotation.
- Language instruction is only provided in a subset and does not run through the whole dataset.
- No tactile, no force, no specialized articulated-object design.
- Headset-specific (Vision Pro), and the transferability to other headsets (Quest, Aria) needs evaluation.

## 7. Takeaway
EgoDex is best for demonstrating the capability of "egocentric video → dexterous manipulation policy", especially the imitation-learning performance under large-scale + high-precision 3D hand tracking. **Not suitable** for evaluating 6D object pose estimation, joint hand-object mesh reconstruction, articulated HOI, in-the-wild complex tasks, or language-conditioned VLA (only partial coverage). In this survey, EgoDex plays the role of "egocentric dexterous manipulation flagship dataset" and serves as the hard anchor for evaluating "embodied learning data sources" in Ch6 and "motion generative prior" in Ch5.
