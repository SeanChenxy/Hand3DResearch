# OakInk (CVPR 2022)

> Yang, Li, Zhan, Wu, Xu, Liu, Lu. *OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction.* CVPR 2022. DOI: 10.1109/CVPR52688.2022.02028. Zotero Key: `6TLSFNKW`.

## Summary
OakInk is a "knowledge-repository-style" 3D HOI dataset: 1,800 CAD objects, 100 carefully recorded objects, 50K affordance-aware + intent-oriented grasping interactions; provides the Oak knowledge base (object affordance) + Ink knowledge base (hand-object interaction) + Tink transfer (interaction reconstruction on virtual objects), and is a composite benchmark for affordance / grasping / handover tasks.

## 1. Dataset Purpose
- Solves the problem that "existing HOI datasets are small in scale and lack affordance and intent dimensions". OakInk explicitly encodes affordance and intent as benchmark dimensions.
- Tasks: (1) 3D hand pose estimation (based on affordance labels); (2) grasp generation (affordance-conditioned); (3) intent-based interaction generation; (4) handover generation.
- Supports both real grasping (Ink, 100 objects) and transferred grasping (Tink, the remaining 1,700 objects).
- Anchors "affordance-aware + intent-oriented HOI" as an independent sub-task.

## 2. Data Composition
- Source: real mocap grasping + Tink virtual-object transfer. 100 real objects are precisely recorded, and 1,800 objects constitute the knowledge base.
- Viewpoint: mocap studio + object scanning; no RGB-D video sequence.
- Scale: 50K distinct affordance-aware, intent-oriented hand-object interactions.
- Object and action: 1,800 objects (household, tools, tableware, toys, etc.), each object is assigned 4–10 affordance labels.
- Action coverage: use, hold, handover, pass, press, lift, etc.

## 3. Annotation and Supervision
- Hand: 3D hand pose (mocap), MANO β / θ.
- Object: 3D mesh (1,800), 6D pose, affordance-region labels.
- Interaction: affordance labels, intent labels (use / hold / handover), contact map.
- Scene: mocap coordinates + object CAD; no RGB images.
- No language, no robot, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) pose estimation (MPJPE / Mesh Error); (2) grasp generation (physical plausibility / diversity / intent accuracy); (3) handover generation (haptic comfort metrics); (4) affordance prediction.
- Key metrics: MPJPE, physics-simulation success rate of generated grasps, affordance accuracy, handover stability.
- Provides Ink (real) / Tink (transferred) split to test cross-object transfer capability.
- The core evaluation source for affordance-aware grasp generation.

## 5. Why It Matters
- The first HOI dataset that takes "affordance" as an explicit benchmark dimension.
- The 50K interactions + 1,800-object knowledge base were among the largest-scale HOI grasping datasets in 2022.
- Tink (transferred grasping) makes it possible to generate HOI on a large number of virtual objects, a "data amplifier" for 3D grasping synthesis.
- Inspires the research direction of "object affordance injection" in Ch4 "semantic prior".
- The training data source for SOTA grasp generation methods such as iHOI, UniDexGrasp, and ContactGen.

## 6. Limitations and Biases
- Only 100 real objects, with the remaining 1,700 objects being transferred; the ground-truth accuracy of the Tink part is weaker than that of Ink.
- No RGB / video: cannot be directly used for vision (RGB / video) tasks.
- No articulated object, tool use, or dynamic manipulation annotation.
- No language instruction, which limits its application to VLA / text-to-grasp tasks.
- Affordance labels are predefined discrete categories, and cannot cover the continuous affordance spectrum.
- The contact map is obtained based on the mesh distance threshold, and there is systematic bias.

## 7. Takeaway
OakInk is best for demonstrating the capability of "affordance-aware / intent-oriented grasp generation", especially cross-object transfer and handover tasks. **Not suitable** for evaluating RGB-based vision tasks, articulated, language-conditioned, or in-the-wild egocentric tasks. In this survey, OakInk plays the role of "affordance-aware HOI grasping + intent-conditioned generation main benchmark" and serves as the hard anchor for evaluating "affordance semantic prior" in Ch4 and "motion generative prior" in Ch5 for grasp synthesis.
