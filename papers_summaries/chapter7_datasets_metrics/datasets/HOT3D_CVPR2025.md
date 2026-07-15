# HOT3D (CVPR 2025)

> Banerjee, Shkodrani, Moulon, Hampali, Han, Zhang, Zhang, Fountain, Miller, Basol, Newcombe, Wang, Engel, Hodan. *HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos.* CVPR 2025. Zotero Key: `D3WKFRS5`.

## Summary
HOT3D is Meta's large-scale first-person 3D hand + object tracking benchmark: using both Project Aria and Quest 3 headsets, it records 833 minutes (3.7M+ images) of egocentric multi-view sequences from 19 subjects interacting with 33 rigid objects. Mocap-grade ground-truth hand / object 6D pose is provided, making it the core benchmark for "real-machine 3D HOI tracking" in the Meta Aria ecosystem.

## 1. Dataset Purpose
- Fills the gap in "first-person + multi-view + multi-headset + large-scale + real physical ground truth" 3D HOI tracking evaluation.
- Tasks: (1) 3D hand tracking; (2) model-based 6DoF object pose tracking; (3) 3D lifting of unknown in-hand objects (model-free grasp object 3D reconstruction).
- Anchors "hand-object tracking" under AR/VR headsets (Quest 3, Aria) as an independent sub-task; fills the gap that HO-3D v3 and DexYCB are all third-view.
- Also provides motion-capture-based "hard GT" — more accurate than the GT obtained by multi-view RGB-D optimization.

## 2. Data Composition
- Source: real capture. 19 subjects perform pick-up, observe, and put-down tasks across multiple environments (kitchen, office, living room).
- Viewpoint: dual headsets (Project Aria research prototype + Quest 3 VR headset), each headset carries multiple synchronized RGB / mono cameras + eye gaze + SLAM.
- Scale: 833+ minutes of video, 3.7M+ images.
- Object and action: 33 rigid objects (kitchen utensils, stationery, living-room items); each object can be grasped and manipulated in multiple ways.
- Contains natural egocentric motion, in-hand manipulation, and multi-object handover.
- No articulated objects (all 33 objects are rigid).

## 3. Annotation and Supervision
- Hand: 3D 21 joints (mocap optical markers, sub-mm accuracy); provided in both UmeTrack and MANO formats.
- Object: 6D pose (mocap markers attached to the object); 3D mesh (in-house scanner, with PBR materials).
- Interaction: no direct contact map or grasp type; can be inferred from mocap trajectories.
- Scene: multi-view RGB + mono images, gaze signal, scene point cloud, camera / hand / object 6D pose, IMU.
- No robot annotation, no language, no tactile.

## 4. Supported Evaluation
- Benchmark tasks: (1) 3D hand tracking (MPJPE / PA-MPJPE / Mesh Error); (2) model-based 6DoF object pose (ADD-S / AUC); (3) model-free 3D lifting of in-hand objects (chamfer / F-score).
- Key metrics: hand MPJPE, object ADD-S / AUC-ADDS, reconstruction chamfer.
- The paper shows that multi-view egocentric data significantly outperforms single-view baselines, quantifying the multi-view gain.
- Also an important basis for AR/VR-headset downstream tasks (hand interaction UI, UI grasping and placing).

## 5. Why It Matters
- The first egocentric 3D HOI tracking data released by a major AR/VR company (Meta), with large scale, hard GT (mocap), and dual headsets.
- For the first time, "3D lifting of unknown in-hand objects" is defined as an independent sub-task ("model-free in-hand 3D").
- Quest 3 has shipped millions of units, and HOT3D's GT comes from this "potential user base" of devices, which is highly significant for AR practicalization.
- Cross-headset (Quest 3 + Aria) makes cross-device generalization evaluation possible.
- The flagship dataset of the Aria ecosystem, expected to drive 1–2 years of egocentric 3D tracking work.

## 6. Limitations and Biases
- Only 33 rigid objects: no articulated objects (such as scissor opening, laptop opening), so it does not directly support the evaluation of articulated priors in Ch3.
- 19 subjects: subject diversity is moderate.
- Under the headset viewpoint, "the subject cannot see his own hand": this is a fundamental difficulty of the model-free in-hand 3D task, and the dataset itself cannot work around it.
- Annotation depends on mocap markers: markers may affect tactile perception and are not applicable to large / deformable objects.
- No language instruction, no task progress / affordance annotation, which limits VLA / robotics transfer.
- No contact-map ground truth.

## 7. Takeaway
HOT3D is best for demonstrating the accuracy of egocentric multi-view 3D hand + object tracking, especially the "user-vision"-side performance in AR/VR headset scenarios. **Not suitable** for evaluating bi-manual, articulated, dexterous manipulation, long-horizon tasks, or language-conditioned generation. In this survey, HOT3D plays the role of "egocentric 3D HOI tracking + AR/VR main benchmark" and serves as a hard anchor for evaluating "video-generative-prior cross-view to 3D improvement" in Ch5.
