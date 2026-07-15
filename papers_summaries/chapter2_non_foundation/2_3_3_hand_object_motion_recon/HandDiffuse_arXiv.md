# HandDiffuse: Generative Controllers for Two-Hand Interactions via Diffusion Models

## Summary
Introduces a large-scale two-hand interaction dataset (HandDiffuse12.5M) and a diffusion-based generative model for synthesizing diverse, realistic two-hand interaction motions.

## 1. Problem and Setting
- Generate realistic two-hand interaction motions (bimanual, with or without objects).
- Input: optional condition signals (hand pose seed, interaction type); output: two-hand MANO motion sequence.
- Two-hand motion generation. Addresses the severe data scarcity in bimanual hand interaction data.

## 2. Core Method
- First contribution: HandDiffuse12.5M, a large-scale dataset of 12.5 million two-hand interaction frames, created through a combination of motion capture, procedural generation, and data augmentation of existing datasets.
- Second contribution: a diffusion-based generative controller for two-hand interactions:
  - A transformer-based diffusion model denoises two-hand MANO parameter sequences.
  - Supports multiple conditioning signals: initial hand pose, desired interaction type, object category.
  - A cross-hand attention mechanism ensures the two hands move in coordinated, physically plausible ways.
  - The model can generate both short-range (grasping) and long-range (extended manipulation) interactions.

## 3. Knowledge, Supervision, and Assumptions
- Training data: HandDiffuse12.5M (custom large-scale dataset), supplemented by InterHand2.6M, GRAB.
- Supervision: MANO parameters.
- Uses MANO for hand.
- Assumes large-scale data enables learning diverse interaction patterns.

## 4. Experiments and Findings
- Datasets: InterHand2.6M, GRAB, custom test set.
- Metrics: FID, diversity, physical plausibility (penetration, contact), user study.
- The scale of HandDiffuse12.5M significantly improves generation quality and diversity compared to models trained on existing smaller datasets. Generates realistic long-range two-hand interactions.

## 5. Strengths and Limitations
### Strengths
- Large-scale dataset addresses the fundamental data bottleneck in two-hand interaction.
- Diffusion model generates diverse, realistic bimanual motions.
- Multiple conditioning modes provide flexible control.

### Limitations
- Dataset quality may vary (procedural generation introduces artifacts).
- Generated motions may lack fine-grained contact details.
- Evaluation of two-hand interaction realism is challenging.
- Requires significant compute for training.

## 6. Takeaway
HandDiffuse demonstrated that scale matters for hand interaction generation, following the broader deep learning trend. The HandDiffuse12.5M dataset is a valuable resource for the community, and the diffusion-based approach shows that two-hand interactions can be learned from data without hand-crafted physical constraints.
