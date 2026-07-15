# ARCTIC (CVPR 2023)

> Fan, Taheri, Tzionas, Kocabas, Kaufmann, Black, Hilliges. *ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation.* CVPR 2023. DOI: 10.1109/CVPR52729.2023.01244. Zotero Key: `P7QSDW4P`.

## Summary
ARCTIC is the first benchmark focused on "bi-manual + articulated object" 3D reconstruction: 2.1M video frames, 11 subjects, 11 articulated everyday objects (scissors, laptop, drill, cup, etc.), with 3D bi-manual mesh + object mesh + dynamic contact ground truth. It provides an evaluation anchor for bi-manual / articulated reconstruction.

## 1. Dataset Purpose
- Solves the fundamental limitation that "existing HOI benchmarks are single-hand + rigid object". ARCTIC directly serves bi-manual + articulated reconstruction.
- Tasks: (1) Consistent motion reconstruction — reconstructing bi-manual + articulated objects from monocular video so that their motions are spatio-temporally consistent; (2) Interaction field estimation — estimating dense hand-object relative distance fields from images.
- Provides two baselines: ArcticNet (reconstruction) and InterField (interaction field).
- Serves as the reference benchmark for all "bi-manual / articulated / contact-aware" methods in Ch3 / Ch5.

## 2. Data Composition
- Source: real capture. 11 subjects manipulate 11 articulated objects under multi-view (8 Aria-like cameras), using both hands.
- Viewpoint: subject wears multi-view head-mounted + 4 external static cameras + tabletop multi-view (≥8 views).
- Scale: 2.1M video frames; about 210 sequences; 360K annotated frames (2K sampling).
- Object and action: scissors, laptop, box, kitchen, phone, espresso, scissor (double-layer), cubes, waffle iron, ketchup, microwave; actions cover open/close, push/pull, stir, press, and other bi-manual manipulations.
- Some of the 11 articulated objects have 1 joint (scissors, laptop, drawer), others have multiple joints (blender).

## 3. Annotation and Supervision
- Hand: 3D 21 joints × 2 hands, MANO β / θ (left and right hands independent), hand mesh.
- Object: mesh-based 6D pose + joint angle (articulated state, such as the scissor joint angle or laptop opening angle).
- Contact: dense contact annotation (whether each vertex on the hand surface is in contact with the object), at the per-timestep level.
- Scene: multi-view RGB, camera intrinsics / extrinsics, scene point cloud (partial sequences).
- No language, no robot annotation, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) hand pose (MPJPE, PA-MPJPE, Mesh Error, per left / right hand); (2) object pose / joint angle error; (3) Contact accuracy (vertex-level F-score); (4) Motion smoothness / physical plausibility.
- Key metrics: hand MPJPE / Mesh Error; object rotation / translation / joint angle error; contact F-score @ 1mm / 5mm.
- Standard train / val / test split and an evaluation server are provided.
- Cross-subject capability: 11 subjects enable unseen-subject evaluation.

## 5. Why It Matters
- The only real dataset that simultaneously has bi-manual + articulated + dense contact annotations.
- Promotes "consistent motion reconstruction" as a standard sub-task for bi-manual HOI.
- The dense contact annotation has inspired follow-up contact-aware HOI methods such as CP3, IPMAN, DAMON, and AlignSDF.
- It is reported as the main benchmark in multiple Ch3 / Ch5 SOTA methods including iHOI, HOISDF, AlignSDF, HMP, and ProHMR-HOI.
- The de facto "gold standard" for evaluation in the bi-manual / dexterous-manipulation direction.

## 6. Limitations and Biases
- 11 articulated objects: object diversity is still limited; the cost of mocap restricts new objects.
- 11 subjects: bi-manual style still has within-subject consistency.
- No dynamic physical quantities (force, torque, tactile).
- Annotation depends on multi-view optimization; joint hinge-point definitions rely on manual work.
- Most objects are indoor / kitchen; few industrial / outdoor objects.
- No language instruction, which limits direct usability for VLA training.

## 7. Takeaway
ARCTIC is best for demonstrating the accuracy of bi-manual + articulated + contact-aware reconstruction methods. **Not suitable** for evaluating robotic transfer, in-the-wild egocentric, language-conditioned generation, or hand embodiments other than dexterous ones. In this survey, ARCTIC plays the role of "main benchmark for bi-manual articulated HOI reconstruction" and serves as the unified benchmark for the comparison of "bi-manual / articulated methods" in Ch2 and "shape-completion-prior" methods in Ch3.
