# Text2Grasp: Grasp Synthesis by Text Prompts of Object Grasping Parts

## Summary
Text2Grasp is a grasp synthesis method guided by text prompts of object grasping parts (rather than full task descriptions), providing more precise part-level control via a two-stage approach: a text-guided diffusion model (TextGraspDiff) generates a coarse grasp pose, then a hand-object contact optimization ensures plausibility and diversity, with extension to LLM-driven task-level and personalized grasp synthesis.

## 1. Problem and Setting
- Grasp synthesis controlled by natural language, addressing the ambiguity in prior text-driven methods that use full task descriptions.
- Input: text prompt describing the object part to grasp (e.g., "grasp the handle of the mug"); optional task-level description.
- Output: MANO hand grasp pose on the specified part.
- Language reasoning prior; uses text for part-level grasp specification.

## 2. Core Method
- A two-stage method:
  1. TextGraspDiff: a text-guided diffusion model generates a coarse grasp pose conditioned on the part-level text prompt.
  2. Hand-object contact optimization: ensures the coarse grasp is both physically plausible and diverse.
- Leveraging Large Language Models: the method facilitates grasp synthesis guided by task-level and personalized text descriptions without additional manual annotations (LLM translates high-level descriptions to part-level prompts).
- How language prior is injected: part-level text prompts provide fine-grained spatial control over grasp location; the LLM translates broader task descriptions into actionable part-level prompts.

## 3. Knowledge, Supervision, and Assumptions
- Training data: HOI datasets with part-level text annotations (or generated via LLM).
- Supervision: MANO grasp parameters, part-level text descriptions.
- Domain knowledge: MANO, grasp physics, LLM-based prompt translation.
- Assumption: part-level text descriptions can be obtained (either annotated or LLM-generated).

## 4. Experiments and Findings
- Datasets: standard grasp benchmarks (e.g., GRAB, DexGraspNet).
- Metrics: grasp quality, part-level accuracy, text alignment, diversity.
- Achieves not only accurate part-level grasp control but also comparable performance in grasp quality.
- LLM-based prompt translation enables task-level and personalized grasp synthesis.

## 5. Strengths and Limitations
### Strengths
- Part-level control is more precise than task-level control.
- Two-stage design (coarse + refinement) provides both flexibility and quality.
- LLM-based prompt translation enables high-level control.
- Contact optimization ensures physical plausibility.

### Limitations
- Requires part-level text annotations (or LLM translation).
- Two-stage pipeline is more complex.
- Diffusion inference is slower.
- May not handle very novel object parts.

## 6. Takeaway
Text2Grasp demonstrates that part-level text prompts provide more precise control over grasp synthesis than full task descriptions, with the two-stage design (text-guided diffusion + contact optimization) ensuring both controllability and physical plausibility. The LLM-based prompt translation extends the method to high-level task descriptions, making it a versatile and practical text-to-grasp pipeline.
