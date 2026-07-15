# Gaze-guided Hand-Object Interaction Synthesis: Dataset and Method

## Summary
Introduces gaze as a novel control signal for hand-object interaction synthesis, creating a dataset with synchronized gaze, hand, and object data, and a method that generates natural HOI motions from gaze input.

## 1. Problem and Setting
- Synthesize 3D hand-object interaction motions given human gaze data as the control/conditioning signal.
- Input: gaze trajectory (eye fixation points over time); output: MANO hand pose trajectory + object motion.
- Motion generation. Gaze serves as a natural, implicit indicator of human intention and attention during manipulation.

## 2. Core Method
- First contribution: gaze-guided HOI dataset — collected with eye-tracking hardware synchronized with hand-object motion capture, providing aligned gaze, hand, and object trajectories for various manipulation tasks.
- Second contribution: a gaze-conditioned motion generation model:
  - Gaze points are encoded as a spatiotemporal sequence.
  - A transformer-based autoregressive model predicts hand MANO parameters frame by frame, conditioned on the gaze history and the object geometry.
  - A contact prediction auxiliary task improves the model's understanding of when and where the hand should contact the object.
- Gaze provides early indicators of intent (humans look at objects before reaching), enabling the model to anticipate future hand motions.

## 3. Knowledge, Supervision, and Assumptions
- Training data: custom gaze+HOI dataset; also trained on GRAB and ARCTIC with pseudo-gaze generated from head orientation.
- Supervision: ground-truth MANO parameters, object poses, gaze coordinates.
- Uses MANO for hand.
- Assumes gaze data is available (from eye tracker or can be inferred from head pose).

## 4. Experiments and Findings
- Datasets: custom gaze-HOI dataset, GRAB, ARCTIC.
- Metrics: MPJPE (hand), object pose error, contact prediction accuracy.
- Gaze conditioning produces more natural and anticipatory hand motions compared to unconditional or object-only conditioned models. Gaze is particularly informative for contact timing.

## 5. Strengths and Limitations
### Strengths
- Novel use of gaze as a natural interaction signal.
- Gaze provides anticipatory cues that improve motion prediction.
- New dataset fills a gap in multi-modal HOI data.

### Limitations
- Requires eye-tracking data (not available from standard RGB video).
- Dataset is relatively small due to hardware requirements.
- Pseudo-gaze from head orientation is a noisy approximation.
- Limited to tasks where gaze and hand movements are correlated.

## 6. Takeaway
This paper introduced gaze as a valuable signal for HOI understanding, showing that where a person looks is a strong predictor of what their hands will do next. This opens up applications in intention prediction, assistive robotics, and more natural human-robot interaction.
