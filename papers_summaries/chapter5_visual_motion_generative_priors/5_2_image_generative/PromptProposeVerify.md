# Prompt-Propose-Verify: A Reliable Hand-Object-Interaction Data Generation Framework using Foundational Models

## Summary
Prompt-Propose-Verify is a framework for generating well-annotated hand-object interaction data for fine-tuning diffusion models, where prompts are proposed, verified, and used to curate high-quality synthetic HOI data that significantly improves hand-object interaction image generation on metrics like CLIPScore, ImageReward, Fidelity, and alignment.

## 1. Problem and Setting
- Pre-trained diffusion models fail to generate accurate hand-object interaction images due to lack of well-annotated training data.
- Input: a foundation model (e.g., an LLM) for prompt generation; an HOI generation system.
- Output: high-quality, well-annotated HOI image dataset.
- Image-generative prior: pre-trained diffusion model serves as the generation backbone, with Prompt-Propose-Verify providing quality-controlled training data.

## 2. Core Method
- Three-step framework: Prompt, Propose, Verify.
  - Prompt: generate text prompts describing hand-object interactions (often via LLM).
  - Propose: use the prompts to generate HOI images via diffusion models.
  - Verify: filter and validate the generated images to ensure quality and accuracy.
- The verified data is used to fine-tune a stable diffusion model for HOI image generation.
- How FM prior is injected: LLM generates prompts; diffusion model generates images; both are foundation models that are orchestrated by the framework.

## 3. Knowledge, Supervision, and Assumptions
- Training data: the framework itself generates the training data via the three-step process.
- Supervision: CLIPScore, ImageReward, Fidelity, alignment metrics for verification.
- Foundation models: LLM for prompt generation; stable diffusion for image generation.
- Domain knowledge: hand-object interaction, prompt engineering, image quality assessment.
- Assumption: the verification step can effectively filter out low-quality images.

## 4. Experiments and Findings
- Datasets: the framework-generated HOI dataset; standard HOI generation benchmarks.
- Metrics: CLIPScore, ImageReward, Fidelity, alignment.
- Shows considerably better performance over baselines.
- The three-step design is critical for data quality.

## 5. Strengths and Limitations
### Strengths
- Three-step framework ensures data quality.
- Uses foundation models for both prompt and image generation.
- Effective for fine-tuning stable diffusion for HOI.
- Quantitative and qualitative improvements.

### Limitations
- Depends on the foundation models' quality.
- Verification step may miss subtle errors.
- Fine-tuning on synthetic data may have residual gap to real data.
- May not scale to very diverse HOI scenarios.

## 6. Takeaway
Prompt-Propose-Verify demonstrates a practical framework for generating high-quality HOI training data by orchestrating multiple foundation models (LLM for prompts, diffusion for images, verification for quality). The work exemplifies the "image-generative prior" paradigm where data quality is achieved through careful orchestration of generative foundation models.
