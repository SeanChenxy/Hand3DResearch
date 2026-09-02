# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions

**Authors:** Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, Hongyang Li  
**Date:** 2025-11-03 (RSS 2025)  
**Identifier:** [arXiv:2505.06111](https://arxiv.org/abs/2505.06111); DOI `10.48550/arXiv.2505.06111`  
**Zotero item:** `RVMFCT44` ([Zotero](zotero://select/library/items/RVMFCT44))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
UniVLA is a cross-embodiment vision-language-action framework that learns task-centric latent actions from unlabeled videos instead of relying on action-annotated robot data. Language instructions condition a latent action model built in DINO feature space, suppressing task-irrelevant dynamics so the learned action space transfers across embodiments and viewpoints. Pre-trained on Internet-scale video plus heterogeneous robot and human data, the policy is deployed on different robots through lightweight latent action decoding. UniVLA outperforms OpenVLA across manipulation and navigation benchmarks while using less than 1/20 of the pre-training compute and 1/10 of downstream data, with continuous gains as heterogeneous data (including human videos) is added.

## Background and Problem
Scaling action-annotated data ties VLA models to a single embodiment and limits cross-environment transfer. The paper targets a generalist policy that works across manipulation and navigation, embodiments, and camera viewpoints, learned from mixed unlabeled video sources without requiring action labels during pre-training.

## Method
A latent action model derives task-centric action representations from video pairs, regularized by language instructions and instantiated in the DINO feature space to filter out task-irrelevant dynamics. Pre-training spans manipulation subsets of Open X-Embodiment, the GNM navigation dataset, and Ego4D human videos. For deployment, a decoder maps latent actions to each robot's native action space with minimal fine-tuning. Fine-tuning experiments use the LIBERO suites (four task suites, 10 tasks each, 50 human-teleoperated demonstrations per task) trained by behavioral cloning.

## Contributions
- Task-centric latent actions learned with language conditioning in DINO feature space, enabling embodiment- and view-agnostic pre-training.
- A cross-embodiment generalist policy pre-trained on Internet-scale video and deployable across robots via efficient latent action decoding.
- State-of-the-art results on manipulation and navigation benchmarks with far lower pre-training compute, and a scaling recipe where heterogeneous data (even human videos) continuously improves performance.

## Experimental Setup
Evaluation covers LIBERO (four suites: Spatial, Object, Goal, Long-horizon; 50 trials per task, averaged over three seeds), SIMPLER, CALVIN, real-world navigation (Room2Room), and real-robot deployments, benchmarked against OpenVLA and other VLA baselines. Pre-training data comprise an Open X-Embodiment subset, GNM, and Ego4D; full corpus statistics are not reproduced from the available evidence.

## Results
- UniVLA achieves an 18.5% success-rate increase over OpenVLA on LIBERO, a 29.6% improvement on navigation tasks, and a 36.7% improvement in real-world deployments in the reported comparisons.
- It surpasses prior state of the art while using less than 1/20 of the pre-training compute (in GPU hours) and 1/10 of the downstream data.
- Adding heterogeneous data — including human videos — to the training pipeline yields continuous performance improvements, supporting scalable policy learning.

## Limitations
Latent actions are derived from visual change, so contact forces and other non-visible state remain outside the pre-training signal. Deployment still requires an action decoder per embodiment, albeit with minimal data. Full per-task breakdowns across all benchmarks and ablations over the latent-action design are not reproduced from the available evidence.
