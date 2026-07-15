# Assembly101 (CVPR 2022)

> Sener, Chatterjee, Shelepov, He, Singhania, Wang, Yao. *Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities.* CVPR 2022. DOI: 10.1109/CVPR52688.2022.02042. Zotero Key: `JDZ3JI67`.

## Summary
Assembly101 is a large-scale "procedural activity" multi-view video dataset: 4,321 videos, 101 take-apart toys, 12 synchronized cameras (8 static + 4 egocentric), 100K+ coarse-grained / 1M+ fine-grained action segments, 18M 3D hand poses. It is a large-scale benchmark for "procedural activity + bi-manual + 3D hand".

## 1. Dataset Purpose
- Solves the dual problem of "existing video datasets lack 'procedural activities' + '3D hand pose'". Assembly101 takes "disassembling toys" as a sample of procedural activities.
- Tasks: (1) action recognition (coarse / fine-grained); (2) action segmentation; (3) 3D hand pose estimation (partial); (4) next-action anticipation; (5) procedural understanding.
- Anchors "procedural activities" + "egocentric multi-view" as independent sub-tasks.
- Complements AssemblyHands: Assembly101's strength is the original multi-view video + action labels, while AssemblyHands's strength is the re-annotation of 3D hand pose.

## 2. Data Composition
- Source: real capture. Multiple subjects freely disassemble 101 take-apart toy vehicles in a controlled studio.
- Viewpoint: 12 synchronized cameras (8 static + 4 egocentric headsets).
- Scale: 4,321 videos, 100K+ coarse-grained action segments, 1M+ fine-grained action segments, 18M 3D hand poses.
- Object and action: 101 take-apart toy vehicles; the action covers the full process of take apart and assemble, including natural errors, corrections, and action order variations.
- No bi-manual specialized design (subjects mainly use a single hand), no articulated-object joint tracking.

## 3. Annotation and Supervision
- Video: 4,321 multi-view videos.
- Annotations: coarse / fine-grained action segment, error annotation, correction annotation, 3D hand pose (2D keypoint, AssemblyHands redo 3D).
- 3D information: 18M frames of 3D hand pose (obtained based on 2D keypoint + multi-view lifting, with medium accuracy).
- Object: no 3D object annotation.
- Interaction: action categories (coarse + fine), error / correction markers.
- No language, no tactile, no robot.

## 4. Supported Evaluation
- Benchmark tasks: (1) action recognition (Top-1); (2) action segmentation (F1 / edit distance); (3) next-action anticipation; (4) 3D hand pose estimation (MPJPE / PA-MPJPE).
- Key metrics: action Top-1, segmental F1, edit distance, MPJPE.
- Provides standard train / val / test split (by subject + toy category).
- 5 major benchmark tasks each have their own specialized evaluation protocol.

## 5. Why It Matters
- The first large-scale "procedural activity + multi-view synchronization" video dataset.
- 101 take-apart toys + 4,321 videos + 18M 3D hand poses are the largest scale in egocentric procedural activity at the time (2022).
- The "error + correction" annotation enables the dataset to support "failure recovery" research.
- Inspired subsequent datasets such as AssemblyHands (3D hand re-annotation) and TACO (tool use extension).
- A video-pretraining anchor shared by multiple chapters including "language reasoning" in Ch4, "video generative prior" in Ch5, and "robot learning" in Ch6.

## 6. Limitations and Biases
- Only 101 take-apart toys: object diversity is limited.
- The 3D hand pose is obtained from 2D keypoints + multi-view lifting, and the accuracy is lower than that of mocap-level datasets such as ARCTIC.
- No 6D object pose, no mesh, no contact map.
- No language instruction (only action labels).
- No tactile, no robot, no specialized articulated-object design.
- Controlled studio environment, with limited generalization to in-the-wild tasks.

## 7. Takeaway
Assembly101 is best for demonstrating the capability of "procedural activity + multi-view egocentric video understanding", especially action segmentation, action anticipation, and error recovery. **Not suitable** for evaluating 3D object mesh reconstruction, 6D object pose, articulated 4D, language-conditioned, or in-the-wild tasks. In this survey, Assembly101 plays the role of "procedural activity + multi-view video flagship benchmark" and serves as a video-pretraining anchor shared by multiple chapters including Ch4, Ch5, and Ch6.
