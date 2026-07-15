# AffordPose (ICCV 2023)

> Jian, Liu, Li, Hu, Liu (Dalian Univ. of Technology, Shandong Univ., Shenzhen Univ., Tsinghua). *AffordPose: A Large-scale Dataset of Hand-Object Interactions with Affordance-driven Hand Pose.* ICCV 2023. DOI: 10.1109/ICCV51070.2023.01352. Zotero Key: `F99HQWYJ`. Code: https://github.com/GentlesJan/AffordPose.

## Summary
AffordPose provides 26,712 affordance-driven 3D hand-object interaction annotations, covering 641 3D objects, 13 categories, and 8 part-level affordance labels (twist, pull, handle-grasp, press, lift, wrap-grasp, support, lever). It is a large-scale benchmark for part-level affordance-driven fine-grained HOI.

## 1. Dataset Purpose
- Solves the problem that "existing HOI datasets have coarse affordance annotations (use / handover) or only grasp type labels". AffordPose takes "part-level affordance + fine-grained hand pose" as the explicit evaluation dimension.
- Tasks: (1) hand-object affordance understanding (affordance classification + localization); (2) affordance-oriented hand-object interaction generation (generating reasonable hand pose given an object and an affordance).
- Anchors "part-level affordance-driven fine-grained HOI" as an independent sub-task; forms a "three-tier affordance granularity" contrast with OakInk (use / handover granularity) and ContactPose (grasp type granularity).

## 2. Data Composition
- Source: synthetic + manual annotation. 641 3D objects come from PartNet and PartNet-Mobility, resized to a normal human-hand scale.
- Viewpoint: mocap coordinates; no RGB / video.
- Scale: 26,712 hand-object interactions; 641 3D objects; 13 object categories; 8 affordance types.
- Object categories: Bag, Bottle, Dispenser, Earphone, Faucet, Handle bottle, Jar, Keyboard, Knife, Laptop, Mug, Pot, Scissors.
- Subset size: 53 / 52 / 34 / 50 / 55 / 32 / 45 / 53 / 57 / 50 / 55 / 48 / 57 (in the above order).
- Action: each affordance (such as twist, pull, handle-grasp) is paired with a set of representative hand poses, and a certain degree of diversity is allowed.

## 3. Annotation and Supervision
- Hand: MANO hand model parameters — palm pose (translation t + rotation q) + 16 joint rotation angles θ (intrinsic parameters).
- Object: 3D mesh (original model of PartNet / PartNet-Mobility, normalized to the human-hand scale) + part-level affordance label.
- Affordance: 8 part-level categories (handle-grasp, press, lift, pull, twist, wrap-grasp, support, lever), determined by 5 volunteers discussing a consensus for each part.
- Contact: based on the GraspIt! simulator + force analysis to avoid physical infeasibility + penetration.
- Scene: mocap coordinates; no RGB, no language, no robot, no tactile.
- Two-stage annotation pipeline: (1) part-level affordance annotation; (2) manually adjust the hand pose in the GraspIt simulator to satisfy the affordance.

## 4. Supported Evaluation
- Benchmark tasks: (1) hand-object affordance understanding (affordance classification + part localization Top-1); (2) affordance-oriented hand-object interaction generation (MPJPE / Mesh Error / physical plausibility of the generated hand pose vs GT).
- Key metrics: affordance classification accuracy, hand pose MPJPE / PA-MPJPE / Mesh Error, physics-simulation success rate of generated hand poses.
- Provides split evaluation by affordance and object category.
- Cross-object / cross-affordance split tests generalization ability.

## 5. Why It Matters
- The first to take "part-level affordance" as the explicit benchmark dimension, distinguishing it from the human-objective of OakInk and the human-objective of ContactPose.
- 641 objects + 26.7K interactions are the largest-scale dataset in "affordance-aware HOI" at the time.
- Provides a causal chain of "affordance label → hand pose", which can directly train affordance-conditioned grasp generation.
- Together with OakInk / ContactPose / GRAB, it forms the "affordance + contact + intent + grasp" full-dimensional HOI grasping evaluation ecosystem.
- One of the core reference datasets of the "affordance semantic prior" section in Ch4.
- The dataset and GT are both public (GitHub), with high reproducibility.

## 6. Limitations and Biases
- Still mocap synthetic (no RGB / video): cannot be directly used for RGB-based vision tasks.
- 13 object categories and 8 affordance categories: limited granularity, cannot cover industrial / outdoor objects.
- No bi-manual specialized design — dominated by single-hand.
- No articulated-object joint tracking (the joints of scissors objects are not explicitly tracked).
- No language instruction, which limits direct VLA / text-to-grasp application.
- No tactile, no force, no dynamic contact annotation.
- Affordance labels are 8 discrete categories, which cannot cover the continuous affordance spectrum (such as press at different intensities).
- Annotation depends on manual work (14 volunteers × 42 interactions on average), with style differences.

## 7. Takeaway
AffordPose is best for demonstrating the capability of "part-level affordance-driven + fine-grained hand pose generation". **Not suitable** for evaluating RGB-based vision tasks, bi-manual, articulated 4D, language-conditioned, or in-the-wild tasks. In this survey, AffordPose plays the role of "part-level affordance HOI main benchmark" and serves as the hard anchor for evaluating "affordance semantic prior" in Ch4 and "motion generative prior" in Ch5 for affordance-conditioned grasp generation. Together with ContactPose, ContactDB, and OakInk, it forms the four major HOI grasping benchmarks.
