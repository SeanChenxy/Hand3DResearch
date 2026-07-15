# Follow My Hold: Hand-Object Interaction Reconstruction through Geometric Guidance

## Summary
A novel diffusion-based framework for reconstructing 3D geometry of hand-held objects from monocular RGB images by leveraging hand-object interaction as geometric guidance — conditioning a latent diffusion model on an inpainted object appearance and using inference-time guidance with optimization-in-the-loop design that directly generates high-quality object geometry during the diffusion process.

## 1. Problem and Setting
- 3D reconstruction of hand-held objects from monocular RGB images, with the geometric relationship between hand and object serving as a guiding signal.
- Input: a single RGB image of a hand holding an object.
- Output: 3D shape of the held object (neural implicit or explicit mesh).
- Task: hand-held object reconstruction with shape completion. This method uniquely conditions the generation process on the hand-object interaction geometry itself.

## 2. Core Method
- A latent diffusion model conditioned on the inpainted object appearance, guided at inference time by geometric cues derived from the hand (spatial proximity, contact regions, relative orientation).
- Optimization-in-the-loop design: supervises the diffusion model's velocity field while simultaneously optimizing the transformations of both the hand and the object being reconstructed.
- The optimization is driven by multi-modal geometric cues: normal and depth alignment, silhouette consistency, and 2D keypoint reprojection.
- Incorporates signed distance field supervision and enforces contact and non-intersection constraints to ensure physical plausibility.
- How FM prior is injected: the diffusion model provides the generative prior for object appearance and shape; hand geometry acts as a control signal constraining generation to be physically compatible with the observed hand pose.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: pre-trained latent diffusion model (likely fine-tuned on Objaverse renderings for 3D-aware generation).
- Domain knowledge: hand-object proximity and contact as geometric constraints; MANO hand model for pose estimation.
- Training data: the diffusion model is pre-trained; an object inpainting model is also used off-the-shelf.
- Assumption: object is held with visible grasping configuration; hand pose can be reliably estimated.

## 4. Experiments and Findings
- Datasets: HO3D, DexYCB, and in-the-wild images.
- Metrics: Chamfer distance, F-score, normal consistency for object shape; visual quality metrics for generated views.
- Outperforms baseline methods that use only appearance-based diffusion guidance (e.g., pure image inpainting + 3D lifting) by leveraging the hand pose as geometric guidance.
- Geometric guidance from the hand pose provides significant improvement in shape accuracy and physical plausibility of the reconstructed object relative to the hand.

## 5. Strengths and Limitations
### Strengths
- Novel conditioning strategy: uses the interaction geometry (not just visual appearance) to guide the diffusion prior.
- Physically more consistent with observed hand pose than appearance-only methods.
- Single-image input, no video required.
- Modular design (inpainting + geometric guidance + 3D lifting).

### Limitations
- Quality depends on accurate hand pose estimation.
- Diffusion-based generation can still produce shapes inconsistent with actual object identity.
- Multi-view consistency of generated images is not guaranteed.
- Limited to hand-held objects with visible grasping configurations.

## 6. Takeaway
Follow My Hold introduces an important conceptual advance: the interaction geometry itself (hand pose, contact, proximity) can serve as a conditioning signal for foundation model priors, not just a downstream constraint. This bridges the gap between purely visual priors (appearance-based diffusion) and physical interaction reasoning, suggesting a direction where FM priors are guided by task-specific geometric constraints for more physically grounded HOI reconstruction.
