# PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation

## Summary
PAM is a unified engine for hand-object interaction (HOI) video generation that unifies pose, appearance, and motion in a single system to enable true sim-to-real deployment, addressing the fragmentation of existing HOI generation research into three disjoint tracks (pose-only, single-image, and pose+first-frame-conditioned video generation).

## 1. Problem and Setting
- Existing HOI video generation methods are fragmented: pose-only synthesis, single-image generation, and video generation requiring the full pose sequence and ground-truth first frame.
- Input: HOI pose sequence (3D hand-object motion) — for true sim-to-real deployment, no first-frame required.
- Output: HOI video with controllable pose, appearance, and motion.
- Video-generative prior: diffusion-based video generation conditioned on 3D HOI pose sequences.

## 2. Core Method
- A unified pose-appearance-motion engine that brings together the three previously disjoint aspects of HOI generation.
- Diffusion-based video generation conditioned on 3D HOI pose sequences, without requiring the ground-truth first frame.
- The engine enables true sim-to-real deployment: synthetic 3D HOI sequences can be converted to realistic videos.
- How FM prior is injected: a pretrained video diffusion model provides the appearance and motion priors, conditioned on 3D HOI poses.

## 3. Knowledge, Supervision, and Assumptions
- Training data: HOI motion data (e.g., GRAB, ARCTIC); HOI video datasets.
- Supervision: 3D HOI motion, video frames, joint pose-appearance alignment.
- Foundation model: pretrained video diffusion model.
- Domain knowledge: hand-object interaction anatomy, video generation, sim-to-real transfer.
- Assumption: 3D HOI poses can effectively condition video generation.

## 4. Experiments and Findings
- Datasets: HOI video benchmarks; sim-to-real evaluation.
- Metrics: video quality, 3D pose alignment, sim-to-real transfer.
- Successfully unifies pose, appearance, and motion in a single engine.
- Enables sim-to-real deployment where 3D HOI sequences (from simulation or other sources) can be directly converted to realistic videos.

## 5. Strengths and Limitations
### Strengths
- Unifies previously disjoint HOI generation paradigms.
- Enables true sim-to-real deployment.
- Single engine for pose, appearance, motion.
- Reduces fragmentation in the field.

### Limitations
- Complex unified architecture is harder to train.
- Quality depends on the pretrained video diffusion model.
- May struggle with very novel hand-object combinations.
- Sim-to-real gap may persist for out-of-distribution scenarios.

## 6. Takeaway
PAM demonstrates that unifying pose, appearance, and motion in a single HOI video generation engine enables true sim-to-real deployment and addresses the fragmentation of the field. The work exemplifies the "video-generative prior" paradigm where pretrained video diffusion models are conditioned on 3D HOI poses for unified generation.
