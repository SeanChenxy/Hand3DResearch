# PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction

## Summary
Generates physically plausible full-body hand-object interaction motions by learning a physics-based imitation policy from video demonstrations, enabling the humanoid to perform dynamic interactions with objects under gravity and contact forces.

## 1. Problem and Setting
- Generate physically plausible full-body human motions interacting with objects (e.g., lifting, pushing, carrying).
- Input: reference motion (from video or keyframes); output: physics-based full-body motion with hand-object contacts that respects Newtonian dynamics.
- Physics simulation based. A simulated humanoid learns to imitate reference HOI motions while respecting physics constraints (gravity, contact forces, friction).

## 2. Core Method
- A reinforcement learning framework with a physics simulator:
  1. Reference motion: extracted from video via off-the-shelf pose estimation (SMPL+H or similar).
  2. Imitation policy: a neural network policy controls the humanoid's joint torques to track the reference motion while maintaining balance and stable object contact.
  3. Reward design: includes terms for pose tracking, object tracking, contact consistency, energy efficiency, and stability (not falling).
  4. Domain randomization: physics parameters (mass, friction) are randomized during training for robustness.
- The key challenge is that the hand must stably grasp the object — the policy learns to apply appropriate forces through the fingers to maintain contact under dynamic conditions.

## 3. Knowledge, Supervision, and Assumptions
- Training data: video-based reference motions; physics simulation for RL training (no real robot data).
- Supervision: reference motion tracking (imitation); physics-based rewards (self-supervised in simulation).
- Uses MANO for hand representation within the SMPL+H humanoid model.
- Assumes the reference motion provides sufficient information for imitation; physics parameters are within the randomization range.

## 4. Experiments and Findings
- Datasets: GRAB, SAMP, custom captures.
- Metrics: pose tracking error, object tracking error, contact consistency, physical plausibility (penetration, floating).
- Generated motions are physically plausible (no floating, no penetration) while maintaining high similarity to reference motions. Policy generalizes to unseen object masses and friction.

## 5. Strengths and Limitations
### Strengths
- Guarantees physical plausibility by construction (physics simulation).
- Handles dynamic interactions with varying object properties.
- Domain randomization enables zero-shot sim-to-real transfer potential.

### Limitations
- Requires a physics simulator and humanoid model (not RGB-only).
- Training is computationally expensive (RL in physics simulator).
- Imitation quality depends on reference motion quality.
- Limited to motions within the humanoid's actuation capabilities.

## 6. Takeaway
PhysHOI demonstrated that physical simulation is the right level of abstraction for ensuring plausible hand-object interactions. Rather than learning physical constraints as a loss term (as in HOLD/HOISDF), embedding the problem directly in a physics engine provides hard guarantees. This physics-first approach is complementary to visual reconstruction methods.
