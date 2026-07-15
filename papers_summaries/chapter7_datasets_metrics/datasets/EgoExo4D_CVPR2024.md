# Ego-Exo4D (CVPR 2024)

> Grauman, Westbury, Torresani, Kitani, Malik, et al. (large consortium). *Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives.* CVPR 2024. DOI: 10.1109/CVPR52733.2024.01834. Zotero Key: `MXVR856Z`.

## Summary
Ego-Exo4D is the first large-scale "first-person + third-person + multi-camera" same-activity multi-view video dataset: 1,400+ hours, 200+ subjects, 100+ skill activities (cooking, sports, mechanical, music, field skills), focused on multi-view synchronous capture and cross-view understanding of "professional skill activities".

## 1. Dataset Purpose
- Solves the fundamental problem that "existing egocentric datasets only provide first-person view and lack synchronized exocentric views". Ego-Exo4D takes "professional skill activities" as the core of multi-view synchronous capture.
- Tasks: (1) cross-view translation (egocentric ↔ exocentric conversion); (2) cross-view 3D hand-object reconstruction; (3) skill assessment; (4) cross-view action understanding.
- Anchors "ego-exo cross-view learning" and "skill understanding" as independent sub-tasks.
- Complements Ego4D: Ego4D's strength is scale and daily activities, while Ego-Exo4D's strength is "professional + multi-view synchronization".

## 2. Data Composition
- Source: real capture. Multi-institution cooperation, subjects perform "professional skill activities".
- Viewpoint: first-person headset (egocentric) + multiple third-person exocentric cameras record simultaneously.
- Scale: 1,400+ hours, 200+ subjects, 100+ skill activities.
- Object and action: cooking, sports, instrument playing, mechanical maintenance, outdoor skills, and other professional activities; the action complexity is high and requires long-term learning.
- Synchronized ego + exo video, 3D scene reconstruction (partial), hand / object annotation (partial).

## 3. Annotation and Supervision
- Video: 1,400+ hours of multi-view synchronization.
- Annotations: fine-grained action labels, skill level labels, language description, 3D scene reconstruction (partial).
- Hand: 3D hand pose is provided in partial sequences (jointly optimized with multi-view ego+exo).
- Object: 6D object pose and mesh are provided in partial sequences.
- Interaction: skill level, step label, language description.
- No contact map, no tactile, no robot annotation.

## 4. Supported Evaluation
- Benchmark tasks: (1) cross-view translation (FID / LPIPS / FVD); (2) 3D hand-object reconstruction (MPJPE / AUC-ADDS); (3) skill assessment (accuracy); (4) step prediction.
- Key metrics: cross-view generation quality, 3D reconstruction accuracy, skill classification Top-1.
- Provides standard train / val / test split (by subject + activity).
- 5 major benchmark tasks each have their own specialized evaluation protocol.

## 5. Why It Matters
- The first large-scale ego + exo synchronized multi-view "professional skill" dataset.
- 100+ skill activities provide a rich training source for "skill learning".
- Promotes "ego-exo cross-view 3D reconstruction" as an independent sub-task.
- Cross-view annotation makes it usable in three chapters: "spatial geometry prior" in Ch3 (multi-view 3D reconstruction), "semantic prior" in Ch4 (skill understanding), and "video generative prior" in Ch5 (cross-view generation).
- The flagship of egocentric datasets in 2024, expected to drive the design of follow-up "multi-view + skill" datasets.

## 6. Limitations and Biases
- Although 1,400 hours is not small, the scale is smaller than the 3,000+ hours of Ego4D.
- 3D annotation is only provided in partial sequences and is not unified.
- The skill category is limited (100+ categories); skill diversity is affected by culture / region (dominated by North America and Europe).
- Headset-specific: the camera quality of different headsets varies.
- No contact map, no detailed language-instruction annotation.
- No articulated 4D mesh reconstruction, no robot annotation.

## 7. Takeaway
Ego-Exo4D is best for demonstrating the capability of "ego-exo cross-view 3D HOI reconstruction + skill understanding". **Not suitable** for evaluating in-studio desktop operation (focused on professional skills), single-view tasks, or language-conditioned tasks. In this survey, Ego-Exo4D plays the role of "multi-view synchronization + professional skill + cross-view 3D HOI main benchmark" and serves as a multi-view anchor shared by multiple chapters including Ch3, Ch4, and Ch5.
