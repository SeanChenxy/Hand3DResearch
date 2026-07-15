# HOI-Swap: Swapping Objects in Videos with Hand-Object Interaction Awareness

## Summary
HOI-Swap is a novel diffusion-based video editing framework trained in a self-supervised manner for precisely swapping objects in videos, especially those interacted with by hands, addressing the failure of existing video editing methods to handle the intricacies of hand-object interactions, with a two-stage design: single-frame object swapping with HOI awareness, followed by temporal propagation.

## 1. Problem and Setting
- Swapping objects in videos, especially those interacted with by hands, while maintaining realistic HOI.
- Input: original video + reference object image.
- Output: edited video with the object swapped, with realistic HOI.
- Video-generative prior: diffusion-based video editing with HOI awareness.

## 2. Core Method
- A two-stage framework:
  1. Single-frame object swapping with HOI awareness: focuses on correctly swapping the object in a single frame while maintaining HOI plausibility.
  2. Temporal propagation: extends the single-frame edit to the entire video consistently.
- Trained in a self-supervised manner (no paired data required).
- How FM prior is injected: the diffusion-based video editing model provides the generative prior; HOI awareness is encoded in the single-frame swapping stage.

## 3. Knowledge, Supervision, and Assumptions
- Training data: self-supervised (uses the model itself for supervision).
- Supervision: self-supervised video editing loss, HOI consistency.
- Foundation model: pretrained video diffusion model.
- Domain knowledge: hand-object interaction, video editing, self-supervised learning.
- Assumption: self-supervised training is sufficient to learn the HOI-aware swapping.

## 4. Experiments and Findings
- Datasets: video object swapping benchmarks.
- Metrics: swap accuracy, HOI realism, temporal consistency.
- Produces realistic object swaps with preserved HOI.
- Two-stage design is critical for HOI preservation.

## 5. Strengths and Limitations
### Strengths
- HOI-aware object swapping.
- Self-supervised training (no paired data required).
- Temporal consistency via two-stage design.
- Practical video editing application.

### Limitations
- Self-supervised training may have quality limits.
- Requires good reference object image.
- May struggle with very complex HOI scenarios.
- Two-stage pipeline is more complex.

## 6. Takeaway
HOI-Swap demonstrates that hand-object interaction awareness is critical for high-quality object swapping in videos, with a two-stage self-supervised framework enabling realistic and temporally consistent results. The work exemplifies the "video-generative prior" paradigm applied to a practical video editing task with HOI considerations.
