# SIGHT: Synthesizing Image-Text Conditioned and Geometry-Guided 3D Hand-Object Trajectories

## Summary
Generates 3D hand-object interaction trajectories conditioned on text descriptions and object geometry, using a diffusion model that jointly synthesizes hand motion and object manipulation paths.

## 1. Problem and Setting
- Generate 3D hand-object interaction trajectories (hand poses + object motion over time) from text descriptions and object geometry.
- Input: text description + 3D object model; output: time-varying MANO hand poses + 6D object pose trajectory.
- Motion generation (not reconstruction). Text-conditioned, geometry-guided generative modeling.

## 2. Core Method
- A diffusion-based generative model operating in the space of hand-object trajectories.
- The trajectory is represented as a sequence of MANO parameters + object 6D poses over a fixed time horizon.
- Conditioning signals:
  1. Text: encoded via a pretrained CLIP text encoder, providing semantic intent (e.g., "lift the mug", "pour water").
  2. Object geometry: encoded via a PointNet-style point cloud encoder from the input 3D object mesh.
- The diffusion model denoises a random trajectory into a coherent hand-object motion that satisfies the semantic intent and is geometrically compatible with the object.
- Additional geometric guidance ensures hand-object contact consistency and non-penetration during the denoising process.

## 3. Knowledge, Supervision, and Assumptions
- Training data: hand-object interaction motion datasets (GRAB, ARCTIC) with associated text descriptions (manually annotated or generated).
- Supervision: ground-truth MANO parameters and object trajectories.
- Uses MANO for hand.
- Pretrained models: CLIP for text encoding; PointNet for object geometry encoding.
- Assumes object 3D model is known; interaction follows the textual description.

## 4. Experiments and Findings
- Datasets: GRAB, ARCTIC.
- Metrics: FID (trajectory quality), contact accuracy, diversity, text-trajectory alignment.
- Generated trajectories are diverse, semantically aligned with text, and geometrically compatible with object shape. Outperforms unconditional and text-only baselines.

## 5. Strengths and Limitations
### Strengths
- Multi-modal conditioning (text + geometry) enables fine-grained control.
- Diffusion model produces diverse, high-quality trajectories.
- Geometry guidance ensures physical compatibility with the object.

### Limitations
- Requires full 3D object model as input (not RGB).
- Training requires text annotations on motion data (scarce).
- Generated trajectories may not be physically realizable (no physics simulation).
- Limited to the interaction types present in training data.

## 6. Takeaway
SIGHT represents the growing trend of using diffusion models for controllable HOI generation, showing that text + geometry conditioning can produce semantically meaningful and geometrically valid manipulation trajectories. This line of work bridges language understanding and 3D interaction modeling.
