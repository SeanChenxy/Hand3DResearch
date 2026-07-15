# HOIGPT: Learning Long-Sequence Hand-Object Interaction with Language Models

## Summary
HOIGPT is a token-based generative method that unifies 3D hand-object interaction (HOI) perception and generation, providing the first comprehensive solution for captioning and generating high-quality 3D HOI sequences from diverse conditional signals (text, objects, partial sequences), built on a novel physically-grounded HOI tokenizer (hand-object decomposed VQ-VAE) and a motion-aware language model trained on both text and HOI tokens.

## 1. Problem and Setting
- Unified 3D HOI perception (captioning) and generation across diverse input/output modalities.
- Input: any of (text, objects, partial HOI sequences).
- Output: any of (text descriptions, completed HOI sequences, generated HOI from text).
- Language reasoning prior; uses a large language model as the central reasoning engine for HOI.

## 2. Core Method
- A novel physically-grounded HOI tokenizer: the hand-object decomposed VQ-VAE, which discretizes HOI sequences into tokens that respect the physical structure of hand-object interaction.
- A motion-aware language model trained to process and generate both text and HOI tokens, enabling bidirectional transformation between language and HOI.
- A unified token sequence representation that allows the LLM to handle both perception (captioning) and generation tasks in a single model.
- How language prior is injected: the LLM is the central model; the HOI tokenizer enables the LLM to process HOI sequences in a language-like manner, with text serving as the natural interface.

## 3. Knowledge, Supervision, and Assumptions
- Training data: HOI motion datasets (e.g., GRAB, ARCTIC) + text caption datasets.
- Supervision: HOI sequences, text descriptions, paired HOI-text data.
- Domain knowledge: VQ-VAE tokenization, large language model training, physical structure of HOI.
- Assumption: HOI sequences can be tokenized in a way that respects their physical structure while being processable by LLMs.

## 4. Experiments and Findings
- Datasets: multiple HOI benchmarks.
- Metrics: R Precision (text generation), FID (HOI generation), and other task-specific metrics.
- Sets new state-of-the-art on both text generation (+2.01% R Precision) and HOI generation (-2.56 FID) across multiple tasks and benchmarks.
- The unified model handles both perception and generation effectively.

## 5. Strengths and Limitations
### Strengths
- First comprehensive unified solution for HOI perception and generation.
- Novel physically-grounded HOI tokenizer that respects HOI structure.
- Motion-aware LLM trained on both modalities.
- Strong empirical results across multiple tasks.

### Limitations
- Requires large-scale paired HOI-text data for training.
- LLM inference is slower than specialized models.
- Tokenization may lose fine-grained details.
- The physical grounding in the tokenizer requires careful design.

## 6. Takeaway
HOIGPT pioneers the "HOI as language" paradigm by tokenizing 3D hand-object interaction sequences in a physically grounded way and applying large language models to process them. This unification of perception and generation through a single LLM-based framework represents a powerful new direction for HOI research, with broad implications for embodied AI and multimodal learning.
