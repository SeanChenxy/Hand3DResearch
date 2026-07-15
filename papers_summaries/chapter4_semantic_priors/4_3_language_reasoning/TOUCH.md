# TOUCH: Text-guided Controllable Generation of Free-Form Hand-Object Interactions

## Summary
TOUCH introduces a new task — Free-Form HOI Generation — for generating controllable, diverse, and physically plausible hand-object interactions beyond fixed grasping patterns (e.g., pushing, poking, rotating), with a multi-level diffusion model that enables fine-grained semantic control conditioned on text, leveraging explicit contact modeling and the new WildO2 dataset of 4.4K diverse in-the-wild interactions across 92 intents and 610 object categories.

## 1. Problem and Setting
- Free-form hand-object interaction generation beyond grasping, including pushing, poking, rotating, and other daily interactions.
- Input: text prompt describing the interaction intent + 3D object.
- Output: controllable, diverse, physically plausible hand-object interaction motion.
- Language reasoning prior; uses text as the fine-grained control signal.

## 2. Core Method
- A three-stage framework centered on a multi-level diffusion model that facilitates fine-grained semantic control to generate versatile hand poses beyond grasping priors.
- The multi-level diffusion model uses explicit contact modeling as a conditioning signal.
- Contact consistency and physical constraints are applied as a refinement stage to ensure realism.
- How language prior is injected: text prompts provide intent descriptions that condition the multi-level diffusion model; the diffusion generates hand poses that align with the described intent.

## 3. Knowledge, Supervision, and Assumptions
- Training data: WildO2 dataset (introduced) — 4.4K unique interactions across 92 intents and 610 object categories from internet videos.
- Supervision: hand pose sequences, object trajectories, contact labels, intent labels.
- Domain knowledge: physical contact constraints, hand pose manifold, language-grounded interaction semantics.
- Assumption: free-form interactions can be controlled by fine-grained intent descriptions.

## 4. Experiments and Findings
- Datasets: WildO2 (introduced); possibly other HOI benchmarks.
- Metrics: generation quality, diversity, physical plausibility, language-instruction alignment.
- Demonstrates the ability to generate controllable, diverse, and physically plausible hand interactions beyond grasping.
- The new dataset (WildO2) is itself a significant contribution to the field.

## 5. Strengths and Limitations
### Strengths
- First framework specifically for free-form HOI generation beyond grasping.
- Multi-level diffusion enables fine-grained semantic control.
- New WildO2 dataset covers diverse intents and object categories.
- Physical constraints ensure realism.

### Limitations
- Free-form interactions are harder to evaluate than grasps.
- Diffusion inference is slower than direct prediction.
- May not cover all possible free-form interactions.
- Depends on the quality and diversity of WildO2.

## 6. Takeaway
TOUCH expands HOI generation from grasping-centric to free-form interactions, demonstrating that text-guided multi-level diffusion with explicit contact modeling can produce controllable, diverse, and physically plausible hand-object interactions for a much wider range of daily activities. The new WildO2 dataset is a valuable resource that broadens the scope of HOI generation research beyond the grasping paradigm.
