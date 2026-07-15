# IGOR: Image-GOal Representations are the Atomic Control Units for Foundation Models in Embodied AI

## Summary
IGOR (Image-GOal Representations) aims to learn a unified, semantically consistent action space across human and various robots by compressing visual changes between an initial image and its goal state into latent actions, enabling knowledge transfer among large-scale robot and human activity data and generating latent action labels for internet-scale video data.

## 1. Problem and Setting
- Embodied AI requires unified representations across diverse embodiments (human, various robots) for knowledge transfer.
- Input: initial image + goal image (or future state).
- Output: a latent action representation (atomic control unit) for embodied AI.
- Video-based pretraining prior: latent action representations learned from video enable internet-scale pretraining.

## 2. Core Method
- Compresses visual changes between an initial image and its goal state into latent actions (atomic control units).
- The latent action space is unified across humans and various robots, enabling cross-embodiment knowledge transfer.
- Latent action labels can be generated for internet-scale video data, enabling large-scale pretraining.
- How FM prior is injected: latent actions bridge the visual state changes from video to actionable control representations.

## 3. Knowledge, Supervision, and Assumptions
- Training data: paired initial-goal image data from human and robot activities; internet-scale video for latent action generation.
- Supervision: latent action prediction; cross-embodiment alignment.
- Foundation models: pretrained image encoders; possibly video models.
- Domain knowledge: embodied AI, cross-embodiment transfer, action representation learning.
- Assumption: visual changes between initial and goal states can be compressed into a unified latent action.

## 4. Experiments and Findings
- Datasets: human activity datasets, robot manipulation benchmarks, internet video.
- Metrics: cross-embodiment transfer, downstream task performance.
- Unified latent action space enables knowledge transfer among human and various robot data.
- Generates latent action labels for internet-scale video data.

## 5. Strengths and Limitations
### Strengths
- Unified action space across embodiments.
- Enables internet-scale video pretraining.
- Cross-embodiment knowledge transfer.
- Simple and general framework.

### Limitations
- May lose fine-grained action details in compression.
- Cross-embodiment alignment may be challenging for very different morphologies.
- Internet-scale labeling requires significant compute.
- May not capture all action semantics.

## 6. Takeaway
IGOR demonstrates that unified image-goal latent action representations enable cross-embodiment knowledge transfer and large-scale video pretraining for embodied AI. The work exemplifies the "video-based pretraining" paradigm where image-goal differences serve as atomic control units.
