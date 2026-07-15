# Ego-Pi: VLA Fine-Tuning for Ego-Centric Human and Robot Data

## Summary
Ego-Pi studies how to co-train on ego-centric human video and robot demonstrations to extend a gripper-based VLA (π0.5) to dexterous humanoid bimanual control. Through action interleaving, human-to-robot hand action alignment, and subtask prediction, the robot learns high-level task semantics (sorting rules, skill composition, rule-based ordering) that only exist in human data, and can execute novel tasks without corresponding robot demonstrations.

## 1. Problem and Setting
- Task: cross-embodiment learning on a humanoid dexterous platform (Galaxea R1 Pro with a 20-joint Tesollo or 6-joint Inspire hand). The goal is to use ego-centric human data plus robot data to co-train a VLA so that the robot can acquire new high-level task semantics (new sorting rules, skill chains, rule-based ordering) that appear only in human demonstrations.
- Input: ego-view image, left/right wrist images, proprioceptive state `s_t`, and language instruction `ℓ_t`.
- Output: optional subtask string + continuous bimanual action chunk `a_{t:t+H}`.
- Modality: multi-view video stream (ego + two wrist cameras), with actions recorded at 100 Hz.
- Hand / object reconstruction: not reconstructed; the policy directly outputs dexterous hand actions via imitation learning.
- Why is this hard:
  - 58-D dexterous hand actions exceed the 32-D output capacity of the π0.5 action head.
  - Human and robot hands differ greatly in appearance, size, and kinematics.
  - IK / optimization for high-DOF dexterous hands is unreliable.
  - The model must transfer task semantics even when no robot data exists for the target new task.

## 2. Core Method
Full pipeline: human or robot images + language + state → π0.5 VLM → infer subtask string (auxiliary) → flow-matching action expert → bimanual action sequence with left/right actions interleaved across two 32-D tokens. For human data, the 20 hand keypoints following the MANO convention are first converted to joint angles, then mapped to Tesollo/Inspire robot joint angles (29-D per hand) via per-joint offsets δ_i and scaling factors f_i, and concatenated to 58-D.

Key innovations:
- **Action interleaving**: distribute the 29-D left/right hand action at each timestep across two 32-D action tokens, so π0.5's 32-D output head can carry 58-D dexterous actions without altering the pretrained action projection.
- **Robot-centric action representation**: map the 20-D human finger joint angles directly via `(q_i + δ_i) f_i` to robot joint angles, bypassing unreliable high-DOF IK.
- **Subtask prediction as auxiliary loss**: let the VLM output a subtask string (e.g., "open box") before predicting actions, so the model learns to "think before acting" in multi-step tasks.
- **Visual skeleton alignment**: render color-coded, occlusion-aware finger skeleton lines on both human and robot hands before feeding images to the model, narrowing the visual domain gap.

Module roles:
- Human → robot action mapping: bridges kinematic differences between embodiments.
- Token interleaving: bridges the dimensional mismatch between dexterous hand action space and pretrained head capacity.
- Subtask prediction: enables the model to plan before acting in multi-step tasks.
- Skeleton overlay: reduces the visual domain gap.

Difference from prior work:
- Unlike Egomimic, EgoScale, etc. (trained from scratch), Ego-Pi fine-tunes a pretrained π0.5 VLA, preserving foundation-model generalization.
- Unlike Masquerade / Mirage (mask human + render robot), Ego-Pi does not need a rendering pipeline; it uses skeleton overlays instead.
- Unlike Egomimic (same-task human data), Ego-Pi targets cross-embodiment semantic transfer where the human data is for a *new* task the robot has never seen.

## 3. Knowledge, Supervision, and Assumptions
- Training data and supervision:
  - Human data: 60–96 demonstrations per task, collected on a tabletop with Manus gloves + Quest controllers, head-mounted ZED mini camera (no wrist camera); wrist pose is tracked by the Quest controller.
  - Robot data: 144–185 demonstrations per task, collected on Galaxea R1 Pro with head-mounted ZED mini and dual Arducam wrist cameras, teleoperated via Manus + Quest.
  - During co-training the robot:human ratio is fixed at 50:50.
  - Because human data has no wrist camera, wrist images are dropped 40% of the time in human batches to keep inputs consistent across embodiments.
  - Supervision: subtask string (language) + continuous action (flow-matching loss).
- Domain knowledge:
  - MANO 20 hand keypoints.
  - Predefined per-joint offset δ_i and scaling factor f_i for every human-to-robot joint pair (handcrafted, listed in the supplementary).
  - Dexterous hand kinematics: Tesollo 20 joints / Inspire 6 joints.
  - 6D rotation representation (Zhou et al. 2020).
- Pretrained / foundation model:
  - π0.5: a flow-matching VLA with a 32-D action head. All pretrained weights are kept; only fine-tuning is performed.
  - Visual skeleton alignment borrows from conventions of color-coding and occlusion-aware drawing from human hand / robot hand visualization.
- Assumptions and input dependencies:
  - Fixed-camera, short-horizon, tabletop settings.
  - The target new task must be related to skills the robot already has.
  - Subtask strings must be explicitly annotated in the dataset.
- Learned vs provided:
  - Learned: high-level task semantics from human demonstrations (sorting rules, skill ordering, rule-based sequencing).
  - Provided: the human-to-robot joint mapping table, token-interleaving scheme, and subtask labels.

## 4. Experiments and Findings
- Tasks: three real-robot tasks on Galaxea R1 Pro.
  1. **Tomato sorting by color**: robot data teaches single-bowl placement, human data teaches the rule of sorting into two bowls by color → tests "new sorting rule" transfer.
  2. **Boxing (skill chaining)**: robot data teaches "open box" and "place block" as isolated skills, human data shows the correct sequencing "open box then place block" → tests "skill composition".
  3. **Packaging (rule-based ordering)**: robot data covers generic item placement, human data shows "place small box first, then bear doll" (heavy items first) → tests "rule-based ordering".
- Key metric: task success rate.
- Most important quantitative results:
  - Tomato sorting: robot-only 40% → human+robot 92% → +subtask+skeleton 92%.
  - Boxing: robot+human 7% → +subtask 93% → +skeleton+subtask 100%.
  - Packaging: robot-only 10% → robot+human 90%.
  - On the Tesollo hand, simple co-training in boxing reaches only 27%; adding subtask prediction lifts it to 67%.
  - Subtask prediction is decisive for boxing (27% → 93%) but has little effect on the simpler tomato-sorting and packaging tasks.
- What the ablations show:
  - Q1: simple co-training already transfers high-level semantics, achieving 90%+ on classification / rule tasks.
  - Q2: subtask prediction solves the "must think before acting" issue in multi-step tasks; skeleton overlay helps only marginally.
  - Q3: the 6-joint Inspire hand outperforms the 20-joint Tesollo on boxing (the Inspire is closer to the human hand size).
  - Q4: human data lacks wrist cameras; dropping wrist cameras at test time noticeably degrades performance → the robot policy does rely on wrist images.
- Cross-dataset / open-category generalization: the test set varies the action target (tomato colors, in-box placement, doll+box), validating semantic transfer at the task level. No open-category object generalization is tested.

## 5. Strengths and Limitations

### Strengths
- Provides a practical recipe (action interleaving + robot-centric mapping + visual skeleton alignment) to extend a pretrained gripper VLA to dexterous bimanual hands while preserving π0.5's pretrained generalization.
- First systematic evidence that "human-only high-level semantics" (sorting rules, skill ordering, rule-based sequencing) can transfer to a humanoid robot through co-training, without requiring robot data for the target new task.
- Subtask prediction is a critical design: on the two-step boxing task it raises success from 27% to 100%.
- Carefully analyzes the human–robot hand gap (visual, skeleton, missing wrist camera) and proposes concrete mitigations.

### Limitations
- Evaluated only on short-horizon, fixed-camera, simple pick-and-place tasks; long-horizon, mobile, and dexterous-in-hand scenarios are not explored.
- Assumes the target new task is related to skills the robot already has (subtask-level reuse); effectiveness is unclear when the human sub-tasks are far from the robot's existing capabilities.
- Depends on explicit subtask string annotation (high data cost); unsupervised subtask inference is not explored.
- Human demonstrations can teach only "high-level semantics": the paper acknowledges that low-level skills must still be learned from robot data; direct low-level skill transfer is left as an open problem.
- The IK / collision issue for high-DOF dexterous hands (e.g., Tesollo 20 joints) remains a hazard — the model sometimes produces self-colliding poses.
- Training data is limited (≤185 robot + ≤96 human demos per task); scaling behavior is not studied.
- Validated only on Galaxea R1 Pro; cross-hardware generalization is unknown.

## 6. Takeaway
Ego-Pi's core contribution is: after extending the gripper-targeted π0.5 VLA to dexterous humanoid bimanual control, ego-centric human data can still teach "new task semantics" — sorting rules, skill ordering, rule-based sequencing — so the robot can succeed on a target new task without any robot data for that task. Action interleaving, human-to-robot hand action alignment, and subtask prediction are the three engineering pieces that make this work. It reframes VLA cross-embodiment learning: with a proper action-space mapping, sufficient token capacity, and the ability to "think before acting", human demonstrations can serve as the "task-semantics teacher" while robot demonstrations continue to supply the low-level execution skills.
