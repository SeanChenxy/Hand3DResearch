# IGOR: Image-GOal Representations are the Atomic Control Units for Foundation Models in Embodied AI

**Authors:** Xiaoyu Chen, Junliang Guo, Tianyu He, Chuheng Zhang, Pushi Zhang, Derek Cathera Yang, Li Zhao, Jiang Bian  
**Date:** 2024-10-17  
**Identifier:** [arXiv:2411.00785](https://arxiv.org/abs/2411.00785); DOI `10.48550/arXiv.2411.00785`  
**Zotero item:** `XCLMJPCN` ([Zotero](zotero://select/library/items/XCLMJPCN))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
IGOR learns a unified, semantically consistent latent action space shared by humans and multiple robots by compressing the visual change between an initial image and its goal state into a latent action. Because these image-goal representations can be generated for Internet-scale video, they serve as atomic control units for training foundation policies and world models across robot and human data (about 2.8M video clips). IGOR can "migrate" an object's motion from one video to another — even across human and robot — by combining its latent action model with a world model, and a foundation policy aligns latent actions with language while a low-level policy grounds them in robot control.

## Background and Problem
Embodied foundation models need a shared action interface that spans human videos and heterogeneous robots, but action labels exist only for narrow robot data. The paper defines the problem as learning a unified latent action space from visual goal change, so that large-scale human activity video and robot data can jointly train world models and policies.

## Method
The latent action model compresses the visual transition between an initial image and a goal state into a latent action; the world model predicts future frames conditioned on these tokens, and the two together enable motion "migration" between videos across embodiments. Pre-training uses roughly 0.8M robot trajectories from an Open X-Embodiment subset (single-arm end-effector control, RT-1 excluded for out-of-distribution evaluation, actions and proprioception discarded) alongside large-scale human activity video with language instructions (Something-Something v2, EGTEA and similar egocentric sources), totaling about 2.8M clips. The foundation policy aligns latent actions with natural language, and a low-level policy maps them to robot control.

## Contributions
- Image-GOal representations: latent actions defined by image-to-goal visual change, forming a semantically consistent action space across humans and robots.
- A pipeline that labels Internet-scale video with latent actions and uses them to train foundation policies and world models jointly on human and robot data.
- Demonstrated cross-embodiment motion migration via the latent action model plus world model, and language alignment of latent actions for effective robot control.

## Experimental Setup
Evaluation covers real-robot manipulation tasks — "Pick Coke Can", "Move Near", and "Open/Close Drawer" — comparing a policy trained with IGOR pre-training against one trained from scratch, plus qualitative retrieval analyses showing image-goal pairs with similar latent actions across out-of-distribution tasks. Trial counts and full success tables are not reproduced from the available evidence.

## Results
- On the evaluated real-robot tasks, IGOR achieves higher or equal success rates than the model trained from scratch, indicating the learned latent actions generalize to real robot control.
- Retrieval analyses show image-goal pairs sharing similar latent actions across semantically related but out-of-distribution language tasks, evidencing a consistent action space.
- The joint latent action and world model transfers an object's motion between videos, including across human and robot embodiments, in qualitative demonstrations.

## Limitations
Latent actions are defined purely by visual change, so aspects of interaction not visible in images are outside the representation. Quantitative evaluation is limited to a small set of real-robot tasks plus qualitative analyses; large-scale benchmark comparisons are not reported in the available evidence. The pre-training corpus discards robot actions and proprioception, so the approach depends on the low-level policy to bridge latent tokens to executable control.
