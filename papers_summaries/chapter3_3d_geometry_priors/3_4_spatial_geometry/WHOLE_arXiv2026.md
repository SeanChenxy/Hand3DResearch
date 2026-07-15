# WHOLE: World-Grounded Hand-Object Lifted from Egocentric Videos

## Summary
WHOLE holistically reconstructs hand and object motion in world space from egocentric videos by learning a generative prior over hand-object motion that is guided at test time to generate trajectories conforming to video observations, with joint generative reconstruction substantially outperforming approaches that process hands and objects separately.

## 1. Problem and Setting
- World-space hand and object motion reconstruction from egocentric videos with object templates.
- Input: egocentric video + object templates; severe occlusions and frequent object entries/exits from the camera view.
- Output: hand motion and 6D object pose estimation in world space, with consistent hand-object relations.
- Task: world-space hand-object interaction reconstruction; uses generative priors (hand-object motion prior).

## 2. Core Method
- Learns a generative prior over hand-object motion to jointly reason about their interactions (instead of processing them separately and post-processing).
- At test time, the pretrained generative prior is guided to generate trajectories that conform to the video observations.
- The hand-object generative prior is the key insight: it captures the natural coupling between hand and object motion during manipulation.
- How FM prior is injected: the hand-object motion prior is learned as a generative model (likely a diffusion model) on large HOI motion datasets; the FM provides the prior distribution for joint hand-object trajectories.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model / generative prior: a hand-object motion generative model (likely a diffusion model) trained on large HOI motion datasets.
- Domain knowledge: hand-object coupling during manipulation; object templates for 6D pose initialization.
- Training data: hand-object interaction motion datasets (e.g., ARCTIC, HOI4D).
- Assumption: object templates are available; the test-time video exhibits sufficient motion for guiding the prior.

## 4. Experiments and Findings
- Datasets: ARCTIC, HOI4D, egocentric hand-object benchmarks.
- Metrics: hand motion accuracy, 6D object pose accuracy, hand-object relative pose.
- Achieves state-of-the-art performance on hand motion estimation, 6D object pose estimation, and their relative interaction reconstruction.
- Joint generative reconstruction substantially outperforms approaches that process hands and objects separately followed by post-processing.

## 5. Strengths and Limitations
### Strengths
- Generative prior over hand-object motion captures natural coupling.
- Joint inference ensures consistent hand-object relations.
- Handles out-of-sight cases via generative prior.
- Strong empirical performance on multiple tasks.

### Limitations
- Requires object templates.
- Generative model inference is slower than direct prediction.
- May struggle with novel interaction types not seen in training.
- The prior is fixed after training; cannot adapt to new object categories without retraining.

## 6. Takeaway
WHOLE demonstrates the power of joint generative modeling for hand-object interaction: by learning a single prior over hand-object motion rather than two separate priors, the model naturally captures the coupling and produces consistent reconstructions, especially in challenging cases (severe occlusions, out-of-sight objects). This "joint generative prior" paradigm is a compelling alternative to the common pipeline of independent hand and object estimation followed by post-hoc alignment.
