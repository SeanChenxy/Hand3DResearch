# Motus: A Unified Latent Action World Model

**Authors:** Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, Jun Zhu  
**Date:** 2025  
**Identifier:** [arXiv:2512.13030](https://arxiv.org/abs/2512.13030); DOI `10.48550/ARXIV.2512.13030`  
**Zotero item:** `2338RB22` ([Zotero](zotero://select/library/items/2338RB22))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Motus unifies understanding, world modeling, and control in a single latent action world model. A Mixture-of-Transformer (MoT) architecture integrates three experts (understanding, video generation, and action), and a UniDiffuser-style scheduler lets one model switch between world models, vision-language-action models, inverse dynamics models, video generation, and video-action joint prediction. Latent actions are learned from optical flow as pixel-level "delta actions", extracted at scale through a three-phase training pipeline over a six-layer data pyramid. On RoboTwin 2.0 simulation Motus reports a +15% improvement over X-VLA and +45% over π0.5, with +11–48% gains in real-world scenarios.

## Background and Problem
Current embodied systems are built as isolated models for perception, world modeling, and control, which fragments multimodal generative capabilities and blocks learning from large heterogeneous data. The paper targets a unified model that performs all embodied functionalities and can be pre-trained on large-scale action-free video by leveraging sharable motion information.

## Method
The MoT architecture couples three experts so one set of weights supports multiple modeling modes, switched by the UniDiffuser-style scheduler. Latent actions are derived from optical flow, giving pixel-level "delta action" supervision that transfers across scenes. Training follows a three-phase pipeline over a six-layer data pyramid, scaling action pre-training from heterogeneous video sources before fine-tuning on robot data.

## Contributions
- A unified latent action world model covering world modeling, VLA control, inverse dynamics, video generation, and video-action joint prediction in one architecture.
- Optical-flow-derived pixel-level "delta actions" enabling large-scale action pre-training from action-free video.
- A three-phase training pipeline with a six-layer data pyramid, plus state-of-the-art simulation and real-robot results demonstrating that unified modeling benefits downstream tasks.

## Experimental Setup
Evaluation covers simulation and real robots. In simulation, single-task performance is measured on 50 representative RoboTwin 2.0 manipulation tasks in randomized scenes, plus multi-task training over 50+ tasks in clean and randomized settings; models are fine-tuned for 40k steps from their pre-trained checkpoints and each task is evaluated over 100 execution trials. Baselines include π0.5 and X-VLA, along with from-scratch and Stage-1-only variants of Motus. Full real-world task lists are not reproduced from the available evidence.

## Results
- RoboTwin 2.0 simulation: Motus reports a +15% average improvement over X-VLA and +45% over π0.5 across the evaluated tasks.
- Real-world scenarios report improvements of +11–48% over baselines in the stated comparisons.
- Multi-task RoboTwin evaluation shows Motus achieving state-of-the-art success in both clean and randomized settings (e.g., 100%/98% on Stack Blocks Two versus 48%/56% for π0.5).

## Limitations
The reported simulation evaluation is concentrated on the RoboTwin 2.0 suite; broader benchmark coverage is not presented in the available evidence. Real-robot gains are summarized as aggregate ranges rather than per-task tables in the extracted evidence. The unified model inherits the compute and data requirements of training three coupled experts, which the paper does not quantify in the available evidence.
