# Dynamic Full-body Motion Agent with Object Interaction via Blending Pre-trained Modular Controllers

**Authors:** Sanghyeok Nam, Byoungjun Kim, Daehyung Park, Tae-Kyun Kim  
**Date:** 2026-05-12  
**Identifier:** [arXiv:2605.11369](https://arxiv.org/abs/2605.11369)  
**Zotero item:** `AH3W8WAT` ([Zotero](zotero://select/library/items/AH3W8WAT))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

This KAIST work targets physically plausible *dynamic*, long-horizon HOI (e.g., running while holding a table) in a physics simulator. A planning stage injects a FullBodyManip HOI prior into an AMASS-pretrained MDM via interaction-consistent inpainting and recovers object trajectories by step-wise rigid alignment; an execution stage trains a lightweight composer that blends two frozen pretrained imitation agents (PHC for full-body agility, InterMimic for HOI contact) at the action level, raising success rate to 0.591 versus 0.526 for fine-tuned InterMimic while cutting training time roughly threefold.

## Background and Problem

Existing HOI datasets are dominated by static, short-horizon interactions, while pretrained agents handle either dynamic locomotion without objects or static HOI motions. Planning-execution methods such as InsActor and CLoSD reduce HOI to goal-reaching (assuming massless objects, emphasizing collision avoidance) and show plan-execution discrepancy when planned contacts are physically infeasible. Synthesis approaches like DAViD require category-specific LoRA adapters per object class and do not enforce contact consistency. The problem is generating goal-driven dynamic HOI — combining locomotion styles such as running, jumping, high kick, and dance with sustained hand-object contacts — executable by a high-DoF humanoid in physics simulation.

## Method

**Planning:** a text-conditioned MDM pretrained on AMASS is steered during sampling with HOI priors from FullBodyManip via interaction-consistent inpainting: the interaction onset (first contact timestep) is detected from a sampled FullBodyManip clip; before onset, full-body poses are inpainted; after onset, only interaction-related joints (thorax, shoulder, elbow, wrist) are inpainted to keep contacts consistent while other joints follow the dynamic MDM prior. Object trajectories are produced by timestep-wise Kabsch-SVD rigid alignment of contact anchors defined in the object's local frame.

**Execution:** two frozen experts are blended — PHC receives humanoid observations and outputs body-only actions (it does not control hands), while InterMimic receives full observations and produces full-body actions. A lightweight eigenbasis composer predicts per-DoF interpolation weights w and bounded extrapolation weights r scaling the action difference between experts, plus PCA-based exploration coefficients over a basis of recent expert action differences; hand actions come directly from InterMimic. Only the composer is trained, with InterMimic's imitation objective on the synthetic plans, in Isaac Gym.

## Contributions

1. A two-stage dynamic HOI planning-and-execution framework that generates kinematically dynamic, interaction-consistent motion plans and executes them in physics simulation. 2. Prior blending for planning: HOI geometric/contact constraints injected into a diffusion motion model, replacing per-category LoRA fine-tuning while explicitly enforcing contact consistency during sampling. 3. Composer-blended execution: spatio-temporal, per-joint action-level blending of pretrained full-body and HOI imitation experts, enabling behaviors neither expert achieves alone, with substantially reduced training cost.

## Experimental Setup

Evaluation uses the InterMimic humanoid (51 SMPL-X joints — 21 body joints excluding root plus 15 per hand — 153-DoF action space with PD control) in Isaac Gym. Four motion styles (run forward, jump forward, high kick, dance) and three objects (smallbox, largetable, clothesstand) define five interaction categories. For generation, a dedicated dynamic HOI test set is built from text prompts combining styles with interaction contexts; metrics cover HOI quality (contact percentage C%, and a newly proposed contact-consistency metric Ccons), physical plausibility following PhysDiff (foot skate, float, Jitter_pos, plus object-floor penetration), and motion quality (Top-3 R-Precision, Diversity; FID omitted for lack of ground-truth dynamic HOI motions). Baselines: MDM, HOI-Diff, DAViD (planning); PPO-from-scratch, PHC, PHC-R, InterMimic, InterMimic-R, InterMimic-FT (imitation). An episode succeeds when the style-specific goal is achieved while contact is maintained.

## Results

In generation, Ours-P achieves C% of 1.000 and Ccons of 0.906, versus DAViD (0.848, 10.9), HOI-Diff (0.285, 18.2), and MDM (0.133, 29.3); with execution, Ours-P+E keeps C% 0.999, Ccons 2.95, and reduces object-floor penetration to 0.009 from 4.196. Ours-P attains higher R-Precision (0.332) than DAViD (0.310) but lower diversity (5.56 vs 6.70). In imitation, the composer reaches the highest success rate 0.591 versus 0.526 for InterMimic-FT, 0.397 for PHC, and 0.227 for PPO from scratch, with training time of 23 versus 75 (about 3x faster) and Jitter_DoF 0.359e3. Ablations show per-DoF blending with PCA exploration (SR 0.591) beats heuristic hand/arm partitioning (0.365-0.472), hard MoE (0.383), and MLP-only blending (0.571).

## Limitations

The inpainting strategy constrains interaction-related joints, reducing motion diversity in exchange for accurate contact — a diversity-contact trade-off. Physical execution slightly degrades contact consistency (Ccons 2.95 versus 0.906 for planning only) and raises foot skate because corrective balance-recovery foot motions are physically valid. Planned references are synthetic and do not account for object mass or induced lean/inertia, so the method accepts a modest geometric deviation (higher E_HOI of 11.667 versus 8.635 for InterMimic-FT). Future work includes orchestrating more pretrained agents and physics-aware planning that models object mass and contact geometry.
