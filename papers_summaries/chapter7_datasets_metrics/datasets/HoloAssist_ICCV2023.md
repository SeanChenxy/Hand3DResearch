# HoloAssist (ICCV 2023)

> Wang, Kwon, Rad, Pan, Chakraborty, Andrist, Bohus, Feniello, Tekin, Frujeri, Joshi, Pollefeys. *HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World.* ICCV 2023. DOI: 10.1109/ICCV51070.2023.01854. Zotero Key: `ALJS7VKI`.

## Summary
HoloAssist is a large-scale "bi-person collaboration" first-person + mixed-reality headset dataset: 166 hours, 350 instructor-performer pairs, 7 synchronized data streams, with action / conversational annotation, focused on "guidance-execution" collaborative egocentric HOI. It is the core dataset for interactive AI assistant training.

## 1. Dataset Purpose
- Solves the problem that "existing egocentric datasets are dominated by single subjects and lack the 'instructor-performer' collaboration mode". HoloAssist explicitly takes "two-person collaboration + real-time guidance" as the evaluation paradigm.
- Tasks: (1) mistake detection; (2) intervention type prediction; (3) hand forecasting; (4) instruction grounding; (5) conversational AI.
- Anchors "egocentric collaborative HOI" + "interactive AI assistant" as independent sub-tasks.
- Complements Ego4D / Ego-Exo4D: HoloAssist's strength is "collaboration + real-time guidance + mixed-reality headset".

## 2. Data Composition
- Source: real capture. 350 instructor-performer pairs collaboratively complete physical manipulation tasks.
- Viewpoint: the performer wears a HoloLens 2 headset, with 7 synchronized data streams (RGB, depth, eye gaze, IMU, SLAM, audio, hand tracking).
- Scale: 166 hours of video; 350 pairs (700 subjects).
- Object and action: physical manipulation tasks (cooking, assembly, cleaning, etc.), dense actions, with a large amount of mistakes + corrections.
- Contains natural conversation + intervention + mistake correction.

## 3. Annotation and Supervision
- Video: 166 hours of 7-stream synchronization.
- Annotations: action label, conversational annotation (instructor's verbal instructions), mistake detection label, intervention type.
- 3D information: HoloLens 2's built-in SLAM 6D camera pose + finger tracking (partial).
- Hand: HoloLens 2's built-in hand tracking (3D joints, with average accuracy).
- Object: no unified 6D pose annotation; partial sequences have bounding boxes.
- Interaction: instructor's spoken instructions (natural language).

## 4. Supported Evaluation
- Benchmark tasks: (1) mistake detection (per-frame binary classification); (2) intervention type prediction (multi-class); (3) hand forecasting (next-N frames hand pose); (4) instruction grounding (language to video region).
- Key metrics: mAP, Top-1, hand MPJPE (for hand forecasting), BLEU (for instruction grounding).
- Provides standard train / val / test split.
- 5 major benchmark tasks each have their own specialized evaluation protocol.

## 5. Why It Matters
- The first large-scale "instructor-performer" collaborative egocentric HOI dataset.
- 166 hours + 350 pairs = the largest-scale "collaboration + mixed-reality" dataset at the time.
- 7-stream synchronization (RGB, depth, gaze, IMU, SLAM, audio, hand) makes it usable in multiple chapters including Ch3 / Ch4 / Ch5 / Ch6.
- Promotes "interactive AI assistant" + "egocentric collaborative HOI" as an independent direction.
- An important anchor of "structured HOI supervision" in Ch6 "robot learning".

## 6. Limitations and Biases
- The distribution of 350 pairs is mainly in North America / Europe, with limited cross-cultural coverage.
- HoloLens 2 headset: the camera quality is average, and the hand-tracking accuracy is not as good as that of Vision Pro / mocap.
- Object annotation is inconsistent: some have bounding boxes, some do not; no 6D pose / mesh.
- The diversity of collaborative tasks is affected by the instructor's creativity.
- No tactile, no robot annotation.
- The conversation is mainly in English, limiting multilingual evaluation.

## 7. Takeaway
HoloAssist is best for demonstrating the capability of "egocentric collaborative HOI + real-time guidance", especially the instructor-performer bidirectional interaction modeling. **Not suitable** for evaluating 6D object pose, joint hand-object reconstruction, articulated HOI, or in-the-wild tasks. In this survey, HoloAssist plays the role of "egocentric collaborative HOI + interactive AI main benchmark" and serves as the flagship anchor of "language reasoning" in Ch4 and "structured HOI supervision" in Ch6.
