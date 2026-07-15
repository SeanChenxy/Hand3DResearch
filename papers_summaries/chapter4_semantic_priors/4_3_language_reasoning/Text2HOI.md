# Text2HOI: Text-guided 3D Motion Generation for Hand-Object Interaction

## Summary
Text2HOI is the first text-guided work for generating sequences of 3D hand-object interaction, decomposing the task into contact generation (VAE-based, conditioned on text + object) and motion generation (Transformer-based diffusion, using the contact map as a strong prior), with a hand refiner module improving temporal contact stability and physical plausibility.

## 1. Problem and Setting
- Text-guided 3D hand-object interaction sequence generation for diverse interaction types and object categories.
- Input: text prompt describing the desired interaction + 3D object mesh.
- Output: physically plausible 3D hand-object interaction motion with correct contact and semantics.
- Language reasoning prior; uses text as the control signal for HOI generation.

## 2. Core Method
- Two-stage decomposition:
  1. Contact generation: a VAE-based network takes text + object mesh and generates the probability of contacts between hand and object surfaces.
  2. Motion generation: a Transformer-based diffusion model uses the 3D contact map as a strong prior for generating physically plausible hand-object motion from text prompts.
- A hand refiner module minimizes the distance between the object surface and hand joints to improve temporal stability of object-hand contacts and suppress penetration.
- How language prior is injected: text prompts condition the contact generation VAE; the generated contact map conditions the motion diffusion model; text is also used to label augmented training data.
- The contact generation network learns local geometry structure independent of object category, enabling generalization to general objects.

## 3. Knowledge, Supervision, and Assumptions
- Training data: existing 3D hand and object motion datasets with newly annotated text labels (contributed by the paper).
- Supervision: hand-object motion, contact maps, text descriptions.
- Domain knowledge: MANO, contact physics, hand-object interaction.
- Assumption: text descriptions can be annotated for existing motion data at scale.

## 4. Experiments and Findings
- Datasets: hand-object motion benchmarks with text annotations.
- Metrics: motion quality, text alignment, physical plausibility, contact accuracy.
- Demonstrates effective text-to-HOI generation with physically plausible results.
- The two-stage decomposition (contact first, then motion) is critical for performance.

## 5. Strengths and Limitations
### Strengths
- First text-guided work for 3D HOI sequence generation.
- Contact-then-motion decomposition is principled and effective.
- Category-agnostic contact generation enables generalization.
- Hand refiner improves temporal stability.

### Limitations
- Requires text annotations on motion data (contributed but limited).
- Transformer-based diffusion inference is slower.
- May not handle very long sequences.
- Contact generation is geometric; dynamic contact not considered.

## 6. Takeaway
Text2HOI pioneers text-guided 3D HOI sequence generation, demonstrating that decomposing the task into contact generation (text-conditioned) and motion generation (contact-conditioned) is a powerful approach. The work establishes the "language reasoning prior" paradigm for HOI generation, with text serving as the semantic control signal and contact maps as the geometric bridge between language and motion.
