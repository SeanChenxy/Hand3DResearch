# BOTH2Hands: Inferring 3D Hands from Both Text Prompts and Body Dynamics

## Summary
Generates 3D two-hand motions conditioned on both text descriptions and body motion, addressing the gap where text-to-motion models produce body motions without hands.

## 1. Problem and Setting
- Generate 3D two-hand motions (MANO parameters) given a full-body motion sequence and an optional text prompt.
- Input: body motion sequence (SMPL/SMPL-X body joints) + text prompt; output: MANO hand motions for both hands over the same time horizon.
- Hand motion generation conditioned on body context. The body motion provides global context (reaching, walking, interacting); the text provides semantic intent.

## 2. Core Method
- A transformer-based model that takes body joint sequences and text embeddings as input, and outputs both-hands MANO parameters frame by frame.
- Key design choices:
  1. Body-hand interaction encoding: cross-attention between body joints and hand joints captures coordination patterns (e.g., when the arm reaches, the hand opens).
  2. Text conditioning via CLIP: text embeddings provide task-level context that body motion alone may not reveal (e.g., "pick up a cup" vs. "wave hello").
  3. Two-hand coordination: a dedicated cross-hand attention module ensures the two hands move in a coordinated manner during bimanual tasks.
- Trained on motion capture data with paired body and hand motions.

## 3. Knowledge, Supervision, and Assumptions
- Training data: motion capture datasets with full-body + hand data (AMASS, GRAB, ARCTIC).
- Supervision: ground-truth MANO parameters.
- Uses MANO for hand.
- Pretrained models: CLIP for text encoding.
- Assumes body motion provides sufficient context for hand motion prediction; text prompt describes the overall activity.

## 4. Experiments and Findings
- Datasets: AMASS (body+hand subset), GRAB, ARCTIC.
- Metrics: MPJPE (hand), motion diversity, text-motion alignment, hand-body coordination metrics.
- Significantly better hand motion quality than text-only or body-only baselines. Two-hand coordination module improves bimanual task performance.

## 5. Strengths and Limitations
### Strengths
- Leverages body context which is often more reliably estimated than hand pose.
- Text conditioning allows task-specific hand motion generation.
- Explicit two-hand coordination modeling.

### Limitations
- Requires body motion as input (not standalone hand generation).
- Body motion errors propagate to hand predictions.
- Limited to tasks where hand motions correlate with body motions.
- Text annotations for motion data are limited.

## 6. Takeaway
BOTH2Hands bridges body-only and hand-only motion generation, showing that body dynamics provide strong priors for hand motion. This hierarchical approach (body first, then hands) reflects the natural kinematic chain and is a practical strategy for full-body HOI generation.
