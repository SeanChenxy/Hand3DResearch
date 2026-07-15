# SeqHAND: RGB-Sequence-Based 3D Hand Pose and Shape Estimation

## Summary
Estimates 3D hand mesh from monocular RGB video by integrating temporal information through a recurrent network, demonstrating that video-based methods significantly outperform single-image approaches under occlusion.

## 1. Problem and Setting
- 3D hand mesh reconstruction from monocular RGB video (hand-only, no object).
- Input: RGB video sequence; output: MANO hand mesh per frame.
- Hand-only reconstruction from video. The hand may be self-occluded or interacting with objects, but only the hand is reconstructed.

## 2. Core Method
- An encoder-decoder architecture with a ConvLSTM temporal module.
- Per-frame: a CNN encoder extracts image features; these are processed by a ConvLSTM that aggregates temporal context across frames; a decoder predicts MANO pose and shape parameters.
- The temporal module implicitly learns to leverage neighboring frames to resolve ambiguities (e.g., when the hand is partially occluded in the current frame but visible in adjacent frames).
- Trained end-to-end with 3D hand mesh supervision.

## 3. Knowledge, Supervision, and Assumptions
- Training data: synthetic and real hand datasets with 3D annotations (FreiHAND, HO3D).
- Supervision: 3D hand keypoints, MANO parameters, 2D keypoints.
- Uses MANO for hand.
- Assumes hand is the primary visible entity; video is temporally coherent.

## 4. Experiments and Findings
- Datasets: FreiHAND, HO3D, DexYCB.
- Metrics: MPJPE, PA-MPJPE, AUC.
- Temporal modeling consistently improves over single-frame baselines, especially under occlusion. ConvLSTM provides a lightweight yet effective temporal aggregation mechanism.

## 5. Strengths and Limitations
### Strengths
- Simple and effective temporal modeling via ConvLSTM.
- Significant improvement over single-frame methods for occluded frames.
- End-to-end trainable.

### Limitations
- Hand-only (no object reconstruction).
- Temporal context is limited to a fixed window.
- ConvLSTM may struggle with very long-range dependencies or scene changes.
- Requires 3D-annotated video for training.

## 6. Takeaway
SeqHAND established that temporal information from video is a powerful cue for resolving hand pose ambiguities, a principle that later works extended to joint hand-object reconstruction from video. The ConvLSTM-based temporal aggregation remains a lightweight baseline for video hand reconstruction.
