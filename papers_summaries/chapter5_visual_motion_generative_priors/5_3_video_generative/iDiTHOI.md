# iDiT-HOI: Inpainting-based Hand Object Interaction Reenactment via Video Diffusion Transformer

## Summary
iDiT-HOI is a novel framework that enables in-the-wild HOI reenactment generation by inpainting the hand-object interaction into a target video, addressing occlusion between hands and objects, variations in object shapes and orientations, and the need to generalize to unseen humans and objects, all via a Video Diffusion Transformer backbone.

## 1. Problem and Setting
- Realistic Hand-Object Interaction (HOI) reenactment generation in digital human videos.
- Input: target video + reference HOI (e.g., hand pose and object trajectory).
- Output: the target video with the reference HOI reenacted in a natural, plausible way.
- Video-generative prior: a Video Diffusion Transformer (DiT) provides the inpainting and video generation capability.

## 2. Core Method
- An inpainting-based framework: the reference HOI is inpainted into the target video, replacing the original hand-object scene.
- A Video Diffusion Transformer (DiT) backbone generates the inpainted video with realistic hand-object interactions.
- Handles occlusions between hands and objects, variations in object shapes/orientations, and generalizes to unseen humans and objects.
- How FM prior is injected: the Video DiT provides the generative prior; inpainting-conditioned generation enables controllable HOI reenactment.

## 3. Knowledge, Supervision, and Assumptions
- Training data: HOI video datasets, in-the-wild human video datasets.
- Supervision: video diffusion loss, HOI consistency, inpainting supervision.
- Foundation model: pretrained Video DiT.
- Domain knowledge: hand-object interaction, video inpainting, DiT-based generation.
- Assumption: the Video DiT can effectively handle inpainting with HOI conditions.

## 4. Experiments and Findings
- Datasets: in-the-wild human video datasets; HOI benchmarks.
- Metrics: video quality, HOI realism, generalization to unseen humans and objects.
- Generates realistic HOI reenactments in in-the-wild videos.
- Generalizes to unseen humans and objects.

## 5. Strengths and Limitations
### Strengths
- Inpainting-based design enables flexible HOI reenactment.
- Video DiT backbone provides high video quality.
- Generalizes to unseen humans and objects.
- Handles complex occlusions and object variations.

### Limitations
- Requires a reference HOI for reenactment.
- Quality depends on the Video DiT backbone.
- May struggle with very large occlusions.
- Computational cost of Video DiT inference.

## 6. Takeaway
iDiT-HOI demonstrates that inpainting-based HOI reenactment in a Video DiT framework enables realistic and generalizable HOI generation in in-the-wild videos. The work exemplifies the "video-generative prior" paradigm where the Video DiT is conditioned on HOI for controllable reenactment.
