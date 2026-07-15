# OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model

## Summary
OpenHOI is the first framework for open-world HOI synthesis, capable of generating long-horizon manipulation sequences for novel objects guided by free-form language commands, integrating a 3D Multimodal Large Language Model fine-tuned for joint affordance grounding and semantic task decomposition, plus an affordance-driven diffusion model and training-free physics refinement for physically plausible interactions.

## 1. Problem and Setting
- Open-world hand-object interaction synthesis for novel objects and open-vocabulary language instructions.
- Input: novel 3D object + free-form language command (e.g., "Find a water bottle and take a sip").
- Output: long-horizon hand-object interaction sequence that accomplishes the instructed task.
- Language reasoning prior; uses 3D MLLM for semantic understanding and affordance grounding.

## 2. Core Method
- A 3D Multimodal Large Language Model (MLLM) is fine-tuned for:
  1. Joint affordance grounding: localizing interaction regions on objects (e.g., handles, buttons).
  2. Semantic task decomposition: breaking down complex instructions into executable sub-tasks.
- An affordance-driven diffusion model synthesizes the hand-object interaction motion conditioned on the affordance grounding and task decomposition.
- A training-free physics refinement stage minimizes penetration and optimizes affordance alignment.
- How language prior is injected: 3D MLLM provides both the affordance grounding and the semantic task decomposition from free-form language, enabling novel object and instruction generalization.

## 3. Knowledge, Supervision, and Assumptions
- Training data: 3D object datasets + HOI motion datasets; MLLM training data includes language-instruction pairs.
- Supervision: HOI motion sequences, affordance labels, language instructions.
- Domain knowledge: physical contact constraints, affordance reasoning, 3D scene understanding.
- Assumption: 3D MLLM can transfer affordance understanding to novel objects.

## 4. Experiments and Findings
- Datasets: open-world HOI evaluation across diverse scenarios.
- Metrics: generalization to novel objects, multi-stage task completion, language instruction alignment.
- Superior to state-of-the-art in generalizing to novel object categories, multi-stage tasks, and complex language instructions.
- The 3D MLLM's affordance grounding is critical for open-world generalization.

## 5. Strengths and Limitations
### Strengths
- First open-world HOI synthesis framework supporting novel objects and open-vocabulary instructions.
- 3D MLLM provides strong language-grounded affordance reasoning.
- Affordance-driven diffusion ensures physical plausibility.
- Training-free physics refinement adds robustness without retraining.

### Limitations
- Depends on the 3D MLLM's quality.
- Multi-stage pipeline is complex.
- May struggle with very long-horizon instructions.
- Diffusion inference is slower than feed-forward methods.

## 6. Takeaway
OpenHOI demonstrates that open-world HOI synthesis is achievable by combining 3D MLLM (for affordance grounding and task decomposition) with affordance-driven diffusion (for motion synthesis) and physics refinement. The work pushes HOI generation from closed-set evaluation to open-world generalization, opening up applications where users provide novel objects and free-form language commands.
