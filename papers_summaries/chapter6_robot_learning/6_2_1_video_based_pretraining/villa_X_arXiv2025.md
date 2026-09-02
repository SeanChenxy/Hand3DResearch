# villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models

**Authors:** Xiaoyu Chen, Hangxing Wei, Pushi Zhang, Chuheng Zhang, Kaixin Wang, Yanjiang Guo, Rushuai Yang, Yucen Wang, Xinquan Xiao, Li Zhao, Jianyu Chen, Jiang Bian  
**Date:** 2025-09-25  
**Identifier:** [arXiv:2507.23682](https://arxiv.org/abs/2507.23682); DOI `10.48550/arXiv.2507.23682`  
**Zotero item:** `CCFUCZJR` ([Zotero](zotero://select/library/items/CCFUCZJR))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
villa-X is a Vision-Language-Latent-Action (ViLLA) framework that improves both how latent actions are learned and how they are integrated into VLA pre-training. Its latent action model adds a proprioception-conditioned future dynamics module so latent codes carry information predictive of robot actions, and the actor module is trained to exploit the pre-trained latent actions. villa-X can plan latent actions zero-shot for unseen embodiments and open-vocabulary symbolic concepts. Across all eight SIMPLER tasks it reports the highest average success rate on both the Google robot (77.7%) and the WidowX robot (62.5%), and it also performs strongly on two real-robot setups covering gripper and dexterous-hand manipulation.

## Background and Problem
Latent action pre-training lets VLAs learn from unlabelled video, but learned latent actions can miss what matters for control, and their integration into policy pre-training is underexplored. The paper targets generalizable language-conditioned manipulation, asking how to make latent actions both higher-quality (control-relevant) and better exploited by the downstream policy.

## Method
The latent action model is improved by incorporating a proprio FDM module — a future dynamics model conditioned on proprioception — so latent tokens are forced to be informative about low-level robot actions (validated by probing with a 3-layer MLP on LIBERO). Ablations also confirm that attention masks and embodiment context improve latent action learning. The actor module is then pre-trained with these latent actions, supporting zero-shot latent action planning for unseen embodiments and open-vocabulary symbolic understanding, followed by task-specific fine-tuning for control.

## Contributions
- An improved latent action model that injects proprioception-conditioned dynamics, making latent actions predictive of executable robot actions.
- A ViLLA pre-training scheme that effectively exploits latent actions, including zero-shot latent planning across unseen embodiments and open-vocabulary concepts.
- Consistent state-of-the-art SIMPLER performance and strong real-robot results on both gripper and dexterous-hand setups.

## Experimental Setup
Evaluation covers all eight SIMPLER tasks in the visual-matching setting across two platforms — the Google robot (Pick, Move, Drawer, Place) and the WidowX robot (Carrot, Eggplant, Spoon, Cube; 24 unique configurations per WidowX task) — plus two real-world robotic setups involving gripper and dexterous hand manipulation. Ablations compare the model with and without the proprio FDM module, attention masks, and embodiment context. Complete training hyperparameters are not reproduced from the available evidence.

## Results
- villa-X reports the highest average success rate among compared methods on SIMPLER: 77.7% on the Google robot and 62.5% on the WidowX robot.
- Probing experiments show the latent actions learned with the proprio FDM module predict low-level robot actions markedly better than those learned without it.
- Ablations confirm each proposed component (attention mask, embodiment context, proprio FDM) improves performance on both robot platforms.
- Real-robot experiments on gripper and dexterous-hand setups support the framework's generality; per-task numbers are not reproduced from the available evidence.

## Limitations
The framework still requires task-specific fine-tuning before real-robot deployment, and zero-shot transfer is demonstrated at the latent-planning level rather than as direct cross-embodiment control. Simulation results dominate the quantitative evidence, with real-robot evaluation limited to two setups. Full real-robot tables and failure analyses are not reproduced from the available evidence.
