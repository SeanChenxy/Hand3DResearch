# GRAB (ECCV 2020)

> Taheri, Ghorbani, Black, Tzionas. *GRAB: A Dataset of Whole-Body Human Grasping of Objects.* ECCV 2020. DOI: 10.1007/978-3-030-58548-8_34. Zotero Key: `DTIVEXBB`.

## Summary
GRAB is the first mocap-driven "whole-body grasping" dataset: 10 subjects, 51 everyday objects, SMPL-X full-body mesh + object mesh + contact ground truth, focused on "how the whole body participates in grasping". It is the pioneering benchmark for grasp generation, whole-body HOI, and affordance learning.

## 1. Dataset Purpose
- Solves the gap that "existing hand-object datasets only care about hands, not the whole body". GRAB fully records the motion in which the entire human body participates in grasping.
- Tasks: (1) 3D hand pose estimation; (2) 6D object pose estimation; (3) hand-object contact prediction; (4) whole-body grasp generation (GrabNet baseline verified).
- Includes both single-hand grasping and two-hand / full-body contact (e.g., body helping to hold up a large object).
- Does not emphasize the visual benchmark under a single RGB image, but emphasizes the mocap-truth-driven high-precision 3D reconstruction / generation benchmark.

## 2. Data Composition
- Source: mocap (58 Vicon markers), collected in a controlled studio.
- Viewpoint: the subject wears a mocap suit, with no external RGB-D, only SMPL-X full-body + object mesh sequences.
- Scale: 10 subjects × 51 objects × multiple grasp types = about 1,334 sequences; 10 hours of mocap data.
- Object and action: 51 everyday objects (box, cylinder, ball, bottle, tool, toy, kitchen utensil), each subject performs multiple grasp categories (lift, use, pass, off-table, etc.) per object.
- Provides a "subject-generalization" split: 2 subjects are left out for testing.

## 3. Annotation and Supervision
- Full body: SMPL-X mesh (body + face + hand), pose + shape.
- Hand: MANO β / θ, 3D 21 joints (decoupled from SMPL-X).
- Object: object mesh, 6D pose trajectory (aligned with the body).
- Contact: vertex-level contact labels between hand-body-object (obtained via mesh distance threshold); contact maps can be computed.
- Scene: mocap coordinates, no RGB, no scene.
- No language, no robot, no tactile, no depth.

## 4. Supported Evaluation
- Benchmark tasks: (1) hand pose (MPJPE / PA-MPJPE / Mesh Error); (2) object pose (rotation / translation error); (3) contact prediction (F-score @ threshold); (4) grasp generation (GrabNet verified).
- Key metrics: hand / object mesh error, contact F-score, generated grasp physical plausibility.
- Mainly used to evaluate 3D grasp generation and whole-body HOI methods; not directly used for vision (RGB / video) tasks.
- Cross-subject capability: 2 subjects can be left out for evaluation.

## 5. Why It Matters
- The first dataset to take the "whole body" as the protagonist of HOI, establishing whole-body HOI as an independent sub-task.
- The GrabNet baseline + GRAB training data have become the "standard" comparison for subsequent grasp generation papers.
- The contact annotation has inspired contact / affordance datasets such as ContactDB, ContactPose, and AffordPose.
- Together with SMPL-X, it provides ground truth for the joint modeling of the entire body-hand.
- A core reference dataset frequently cited in the "affordance" section of Ch4.

## 6. Limitations and Biases
- Studio mocap: large distribution difference from real RGB video / egocentric video; domain adaptation is required when migrating to vision.
- No RGB annotation: cannot be directly used for RGB-based vision tasks.
- 51 objects + 10 subjects: relatively small compared to datasets like ARCTIC in terms of object scale.
- Contact is inferred from mesh distance, which is physically inaccurate (hand penetration, air contact, etc.).
- No articulated object, tool use, or dynamic contact annotation.
- No language or affordance language annotation.

## 7. Takeaway
GRAB is best for demonstrating the capability of "whole-body 3D grasp generation + contact-aware reconstruction", especially the joint generation of SMPL-X full-body mesh + object mesh + contact map. **Not suitable** for evaluating RGB-based vision tasks, in-the-wild egocentric, articulated object, or language-conditioned generation. In this survey, GRAB plays the role of "whole-body HOI + grasp generation main benchmark" and serves as the unified anchor for evaluating "affordance prior" in Ch4 and "motion generative prior" in Ch5 for grasp generation.
