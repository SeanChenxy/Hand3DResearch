# Affordance Diffusion: Synthesizing Hand-Object Interactions

## Summary
Affordance Diffusion is a two-step generative approach for synthesizing plausible images of a human hand interacting with a given object, where a LayoutNet samples an articulation-agnostic hand-object interaction layout and a ContentNet synthesizes images of a hand grasping the object, both built on a large-scale pretrained diffusion model, enabling generalization to novel objects and providing descriptive affordance information (hand articulation, approaching orientation).

## 1. Problem and Setting
- Synthesize complex interactions (an articulated hand) with a given object from a single RGB image of the object.
- Input: RGB image of a target object.
- Output: plausible RGB image of a hand interacting with the object, with descriptive affordance information (articulation, approaching orientation).
- Image-generative prior: a pretrained large-scale diffusion model serves as the latent representation foundation.

## 2. Core Method
- Two-step generative approach:
  1. LayoutNet: samples an articulation-agnostic hand-object-interaction layout (hand pose and contact region).
  2. ContentNet: synthesizes an image of a hand grasping the object given the predicted layout.
- Both networks are built on top of a large-scale pretrained diffusion model, leveraging its learned latent representation.
- The system can also predict descriptive affordance information: hand articulation, approaching orientation.
- Generalizes better to novel objects and in-the-wild scenes compared to baselines.

## 3. Knowledge, Supervision, and Assumptions
- Training data: paired hand-object images with ground-truth hand poses and object images (likely Objaverse or similar 3D asset datasets).
- Supervision: image-level loss, optional hand pose supervision.
- Foundation model: large-scale pretrained diffusion model (Stable Diffusion or similar).
- Domain knowledge: grasp affordance reasoning, hand-object interaction anatomy.
- Assumption: the pretrained diffusion model's latent space captures sufficient structure to be conditioned on hand-object layouts.

## 4. Experiments and Findings
- Datasets: novel object categories, out-of-distribution in-the-wild scenes of portable-sized objects.
- Metrics: visual quality, hand plausibility, affordance prediction accuracy.
- Generalizes better to novel objects than baselines.
- Performs surprisingly well on out-of-distribution in-the-wild scenes.
- Affordance predictions (articulation, approaching orientation) are useful for downstream robotic manipulation.

## 5. Strengths and Limitations
### Strengths
- Two-step design cleanly separates layout (where) from content (what).
- Leverages pretrained diffusion model's rich representation.
- Generalizes to novel objects.
- Provides descriptive affordance information.

### Limitations
- Two-step pipeline may accumulate errors.
- Quality depends on the underlying pretrained diffusion model.
- May not handle very complex articulated objects.
- Training data with diverse objects is needed for generalization.

## 6. Takeaway
Affordance Diffusion demonstrates that complex hand-object interactions can be synthesized by decomposing the problem into layout generation and content synthesis, both leveraging a large-scale pretrained diffusion model. The work exemplifies the "image-generative prior" paradigm: using pretrained generative models as the foundation for controllable, realistic HOI image synthesis.
