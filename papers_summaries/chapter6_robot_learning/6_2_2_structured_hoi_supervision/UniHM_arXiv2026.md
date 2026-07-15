# UniHM: Unified Dexterous Hand Manipulation with Vision Language Model

## Summary
UniHM is the first framework for unified dexterous hand manipulation guided by free-form language commands, with a Unified Hand-Dexterous Tokenizer that maps heterogeneous dexterous-hand motion data into a unified discrete token space, plus a Vision-Language Model for generating motion tokens from language and visual inputs, enabling physically feasible dexterous manipulation from open-vocabulary instructions.

## 1. Problem and Setting
- Planning physically feasible dexterous hand manipulation is a central challenge in robotic manipulation and Embodied AI.
- Prior work relies on object-centric cues or precise hand-object interaction sequences, foregoing the rich compositional guidance of open-vocabulary instruction.
- Input: free-form language command + visual observation + dexterous hand motion data.
- Output: a unified dexterous hand motion sequence that follows the language instruction.
- Structured HOI supervision prior: dexterous hand motion data provides structured HOI supervision.

## 2. Core Method
- A Unified Hand-Dexterous Tokenizer that maps heterogeneous dexterous-hand motion data into a unified discrete token space.
- A Vision-Language Model (VLM) generates motion tokens from language and visual inputs, conditioned on the unified token space.
- The framework enables unified dexterous hand manipulation guided by free-form language.
- How FM prior is injected: the VLM (a large foundation model) provides the language understanding and visual reasoning; the unified tokenizer bridges motion data and language.

## 3. Knowledge, Supervision, and Assumptions
- Training data: dexterous hand manipulation datasets; visual-language data.
- Supervision: motion token prediction; VLA training; physical feasibility.
- Foundation model: Vision-Language Model (e.g., from large-scale pretraining).
- Domain knowledge: dexterous manipulation, VLA, hand-object interaction, open-vocabulary understanding.
- Assumption: the unified token space can represent diverse dexterous hand motions.

## 4. Experiments and Findings
- Datasets: dexterous hand manipulation benchmarks; language-instruction evaluation.
- Metrics: motion quality, language alignment, physical feasibility.
- Achieves unified dexterous hand manipulation from free-form language.
- The unified tokenizer enables effective VLM-based motion generation.

## 5. Strengths and Limitations
### Strengths
- First unified framework for free-form language-guided dexterous manipulation.
- Unified Hand-Dexterous Tokenizer enables diverse motion data integration.
- VLM provides open-vocabulary understanding.
- Physical feasibility by design.

### Limitations
- Requires dexterous hand motion data.
- Computational cost of VLM.
- May not handle very novel physical tasks.

## 6. Takeaway
UniHM demonstrates that unified dexterous hand manipulation can be achieved by combining a hand-dexterous tokenizer with a Vision-Language Model. The work exemplifies the "structured HOI supervision" paradigm where dexterous motion tokens and language conditioning enable free-form instruction following.
