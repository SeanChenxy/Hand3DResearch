# AnchorCrafter: Animate Cyber-Anchors Selling Your Products via Human-Object Interacting Video Generation

## Summary
AnchorCrafter is a novel diffusion-based system designed to generate 2D videos featuring a target human and a customized object with high visual fidelity and controllable interactions, specifically for product promotion video generation, integrating human-object interactions into pose-guided human video generation.

## 1. Problem and Setting
- Anchor-style product promotion video generation is challenging due to the need to integrate human-object interactions with pose-guided human video generation.
- Input: target human pose + reference object image.
- Output: 2D video featuring the human interacting with the object in a controllable, high-fidelity manner.
- Video-generative prior: diffusion-based video generation with HOI awareness.

## 2. Core Method
- Two key innovations:
  1. HOI-appearance perception: integrates the appearance of the object with the human pose for HOI-aware video generation.
  2. (Likely) HOI-aware motion control: ensures the human-object interaction is realistic and controllable.
- The system is designed for product promotion videos, with high visual fidelity requirements.
- How FM prior is injected: the diffusion-based video generation model provides the visual fidelity; the HOI-aware conditioning provides the controllable interaction.

## 3. Knowledge, Supervision, and Assumptions
- Training data: product promotion videos, HOI video datasets.
- Supervision: video diffusion loss, HOI consistency, visual fidelity.
- Foundation model: pretrained video diffusion model.
- Domain knowledge: human-object interaction, product video generation, anchor-style presentation.
- Assumption: the HOI-awareness can be effectively integrated into the video generation process.

## 4. Experiments and Findings
- Datasets: product promotion video datasets; HOI benchmarks.
- Metrics: visual fidelity, HOI realism, controllable interaction.
- Generates high-fidelity product promotion videos with controllable HOI.
- The HOI-appearance perception is critical for visual quality.

## 5. Strengths and Limitations
### Strengths
- Specifically designed for product promotion videos.
- High visual fidelity with controllable interactions.
- Integrates HOI into pose-guided video generation.

### Limitations
- Specialized for product promotion use case.
- May not generalize to non-promotion contexts.
- Requires high-quality reference object images.
- Quality depends on the underlying video diffusion model.

## 6. Takeaway
AnchorCrafter demonstrates that integrating human-object interactions with pose-guided video generation enables high-quality product promotion videos with controllable interactions. The work exemplifies the "video-generative prior" paradigm applied to a specific commercial use case, showing how foundation models can be specialized for practical applications.
