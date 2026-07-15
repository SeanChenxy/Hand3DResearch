# ByteLoom: Weaving Geometry-Consistent Human-Object Interactions through Progressive Curriculum Learning

## Summary
ByteLoom is a Diffusion Transformer (DiT)-based framework that generates realistic HOI videos with geometrically consistent object illustration by using simplified human conditioning and progressive curriculum learning, addressing the lack of effective multi-view information injection and the heavy reliance on fine-grained hand mesh annotations in existing HOI video generation.

## 1. Problem and Setting
- HOI video generation suffers from: (1) lack of effective mechanisms to inject multi-view object information, leading to poor cross-view consistency; (2) heavy reliance on fine-grained hand mesh annotations.
- Input: simplified human condition (e.g., skeleton or coarse pose) + object information.
- Output: realistic HOI video with geometrically consistent object illustration.
- Video-generative prior: a DiT-based video generation model with multi-view object conditioning.

## 2. Core Method
- A DiT-based framework that generates HOI videos with geometrically consistent object illustration.
- Uses simplified human conditioning (not full mesh) to reduce the dependency on fine-grained hand mesh annotations.
- Progressive curriculum learning enables stable training of the complex multi-modal generation model.
- How FM prior is injected: the DiT video diffusion model provides the generative foundation; multi-view object information is injected for cross-view consistency.

## 3. Knowledge, Supervision, and Assumptions
- Training data: HOI video datasets (possibly Objaverse for 3D object data).
- Supervision: video diffusion loss, 3D consistency loss.
- Foundation model: DiT-based video diffusion model.
- Domain knowledge: HOI video generation, curriculum learning, multi-view consistency.
- Assumption: simplified human conditioning is sufficient for generating realistic HOI videos.

## 4. Experiments and Findings
- Datasets: HOI video benchmarks; possibly Objaverse.
- Metrics: video quality, 3D consistency, hand-object realism.
- Generates HOI videos with geometrically consistent object illustration.
- Progressive curriculum learning enables stable training of complex multi-modal generation.

## 5. Strengths and Limitations
### Strengths
- Simplified human conditioning reduces annotation requirements.
- DiT-based architecture provides high video quality.
- Multi-view object conditioning ensures cross-view consistency.
- Progressive curriculum learning enables stable training.

### Limitations
- Simplified conditioning may miss fine hand details.
- DiT-based models are computationally expensive.
- May not handle very complex hand articulations.
- Multi-view consistency is hard to guarantee perfectly.

## 6. Takeaway
ByteLoom demonstrates that simplifying the human conditioning (not requiring full hand mesh) and using multi-view object conditioning with progressive curriculum learning enables geometrically consistent HOI video generation. The work exemplifies the "video-generative prior" paradigm where DiT-based video diffusion models with multi-view object information produce high-quality HOI videos.
