# FLARE: Robot Learning with Implicit World Modeling

**Authors:** Ruijie Zheng, Jing Wang, Scott Reed, Johan Bjorck, Yu Fang, Fengyuan Hu, Joel Jang, Kaushil Kundalia, Zongyu Lin, Loic Magne, Avnish Narayan, You Liang Tan, Guanzhi Wang, Qi Wang, Jiannan Xiang, Yinzhen Xu, Seonghyeon Ye, Jan Kautz, Furong Huang, Yuke Zhu, Linxi Fan  
**Date:** 2025-05-21  
**Identifier:** [arXiv:2505.15659](https://arxiv.org/abs/2505.15659); DOI `10.48550/arXiv.2505.15659`  
**Zotero item:** `SXPU7G8N` ([Zotero](zotero://select/library/items/SXPU7G8N))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
FLARE integrates implicit world modeling into robot policy learning by aligning diffusion-transformer policy features with latent embeddings of future observations, so the policy anticipates long-term consequences while generating actions. The modification is lightweight — adding a few tokens to standard VLA architectures — yet FLARE achieves state-of-the-art on two multitask simulation benchmarks (single-arm RoboCasa and humanoid GR1 tabletop), outperforming prior policy baselines by up to 26%. It also co-trains with actionless human egocentric video, boosting generalization to novel objects with unseen geometry from as few as a single robot demonstration.

## Background and Problem
Policy learning without foresight struggles with long-horizon consequences, while explicit world models add heavy machinery. The paper targets high-frequency robotic control enriched with predictive latent world modeling, keeping the VLA architecture essentially unchanged.

## Method
A diffusion transformer policy is trained with an alignment objective against latent embeddings of future observations: the model predicts future latent representations alongside actions, giving it an implicit world model. Because the mechanism is token-level, it composes with standard VLA backbones. The same observation-embedding model supports co-training with human egocentric video without action labels.

## Contributions
- Future latent representation alignment as a minimal, general add-on for implicit world modeling in diffusion transformer policies.
- State-of-the-art multitask simulation results across single-arm and humanoid tabletop manipulation with up to 26% gains over prior baselines.
- Actionless human egocentric video co-training that improves generalization to novel-geometry objects with as few as one robot demonstration.

## Experimental Setup
Simulation evaluation covers RoboCasa (24 atomic kitchen tasks with a Panda arm: pick-and-place, door/drawer manipulation, faucet operation) and GR1 humanoid tabletop manipulation (24 tasks), against baselines including Policy Only, UWM, GR00T N1 (scratch), and Diffusion Policy. Real-world experiments collect 100 trajectories per task on a real GR1 humanoid post-trained from the pre-trained action-aware observation embedding model. Full hyperparameters are not reproduced from the available evidence.

## Results
- Multitask simulation: FLARE averages 70.1% on the 24 RoboCasa tasks and 55.0% on the 24 GR1 tabletop tasks, versus 61.9% and 44.0% for the Policy Only baseline, with gains over prior policy-learning baselines reaching 26%.
- Real-world: with 100 trajectories per task on a real GR1 humanoid, the FLARE policy achieves a 95% success rate in the reported evaluations.
- Co-training with human egocentric video significantly boosts generalization to a novel object with unseen geometry, even with a single robot demonstration.

## Limitations
The world model is implicit and latent-level, so it does not expose interpretable future predictions or support planning by simulation. The quantitative evidence centers on two simulation suites plus a limited real-robot task set. Full ablation tables and cross-embodiment transfer beyond the reported setups are not reproduced from the available evidence.
