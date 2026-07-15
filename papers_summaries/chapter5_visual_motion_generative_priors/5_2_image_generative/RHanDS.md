# RHanDS: Refining Malformed Hands for Generated Images with Decoupled Structure and Style Guidance

## Summary
RHanDS is a conditional diffusion-based framework that refines malformed hands in generated images by utilizing decoupled structure guidance (from a hand mesh reconstructed from the malformed hand) and style guidance (from the original malformed hand), with a two-stage training strategy and multi-style hand datasets to alleviate mutual interference between style and structure.

## 1. Problem and Setting
- Diffusion models can generate high-quality human images but suffer from instability in generating hands with correct structures.
- Input: a generated image with malformed hands.
- Output: a refined image with corrected hand structure and preserved style.
- Image-generative prior: a conditional diffusion model provides the refinement capability.

## 2. Core Method
- Two guidance types:
  - Structure guidance: hand mesh reconstructed from the malformed hand provides structure information for correcting the hand.
  - Style guidance: the original malformed hand provides style information to preserve appearance.
- A two-stage training strategy:
  - First stage: use paired hand images to ensure stylistic consistency in style transfer.
  - Second stage: build multi-style hand datasets to handle diverse hand styles.
- A latent diffusion model conditioned on both structure and style guidance generates the refined hand image.
- How FM prior is injected: the diffusion model is the foundation for refinement; the structure and style guidance come from external analysis of the input image.

## 3. Knowledge, Supervision, and Assumptions
- Training data: multi-style hand datasets (paired and unpaired) for the two-stage training.
- Supervision: image-level diffusion loss, hand structure supervision.
- Foundation model: pretrained diffusion model.
- Domain knowledge: hand mesh reconstruction, hand structure representation, style/structure disentanglement.
- Assumption: structure and style guidance can be effectively decoupled.

## 4. Experiments and Findings
- Datasets: hand image datasets (paired for training, generated images for evaluation).
- Metrics: hand structure accuracy, style preservation, image quality.
- Effectively refines malformed hands while preserving the original style.
- The two-stage training and multi-style datasets are critical for performance.

## 5. Strengths and Limitations
### Strengths
- Decoupled structure and style guidance addresses a common failure mode.
- Preserves the original style while correcting structure.
- Two-stage training reduces interference.

### Limitations
- Requires hand mesh reconstruction from the input, which itself may be error-prone.
- Limited to hand-only refinement (no object).
- May not handle all hand malformations.
- The two-stage training is more complex.

## 6. Takeaway
RHanDS demonstrates that decoupled structure and style guidance in a conditional diffusion framework can effectively refine malformed hands in generated images while preserving the original style. The work exemplifies the "image-generative prior" paradigm applied to post-hoc refinement of generative model outputs.
