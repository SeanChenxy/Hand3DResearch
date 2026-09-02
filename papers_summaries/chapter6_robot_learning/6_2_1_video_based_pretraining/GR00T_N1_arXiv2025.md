# GR00T N1: An Open Foundation Model for Generalist Humanoid Robots

**Authors:** NVIDIA GR00T N1 team (Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, et al.; authors listed alphabetically; project leads Linxi "Jim" Fan and Yuke Zhu)  
**Date:** 2025-03-27  
**Identifier:** [arXiv:2503.14734](https://arxiv.org/abs/2503.14734)  
**Zotero item:** `YYA6XZNK` ([Zotero](zotero://select/library/items/YYA6XZNK))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
GR00T N1 is an open Vision-Language-Action foundation model for humanoid robots with a dual-system architecture: a vision-language module (System 2) interprets the environment from vision and language, while a diffusion transformer module (System 1) generates fluid motor actions in real time; both are jointly trained end-to-end. It is trained on a heterogeneous mixture of real robot trajectories, human videos, and synthetic data, outperforming state-of-the-art imitation-learning baselines on standard simulation benchmarks across embodiments, and is deployed on the Fourier GR-1 humanoid for language-conditioned bimanual manipulation with high data efficiency.

## Background and Problem
Generalist autonomy in the human world requires a robot foundation model trained on massive, diverse data so robots can reason about novel situations and rapidly learn new tasks. The paper targets an open, generalist humanoid model covering reasoning, robust real-world behavior, and data-efficient skill acquisition, released with checkpoints, simulation environments, and datasets for reproducibility.

## Method
The dual-system design couples a slower vision-language reasoning module with a fast diffusion transformer action generator, trained end-to-end. Training mixes real robot trajectories, human videos, and synthetically generated datasets. An implementation finding reported by the paper: using middle-layer instead of final-layer LLM embeddings yields both faster inference and higher downstream policy success rate.

## Contributions
- An open VLA foundation model for humanoids with a dual-system (reasoning + real-time action) architecture trained end-to-end.
- A heterogeneous training recipe combining real robot trajectories, human videos, and synthetic data.
- Cross-embodiment simulation results exceeding state-of-the-art imitation-learning baselines, plus real deployment on the Fourier GR-1 with high data efficiency.

## Experimental Setup
Simulation evaluation spans three benchmarks chosen to mirror real settings: RoboCasa Kitchen (24 tasks) and two further open-source multitask suites across robot embodiments, plus a newly developed tabletop suite matching the real-robot tasks. Real-world evaluation uses the Fourier GR-1 humanoid on tabletop bimanual manipulation, reporting average success over 10 trials per task (with a placement-count protocol for the Pack Machinery task under a 30-second limit). Full data-mixture statistics are not reproduced from the available evidence.

## Results
- GR00T N1 outperforms state-of-the-art imitation-learning baselines on the standard simulation benchmarks across multiple embodiments.
- On the real Fourier GR-1 humanoid, the model achieves strong performance with high data efficiency on language-conditioned bimanual tabletop manipulation.
- Middle-layer LLM embeddings improve both inference speed and downstream policy success rate relative to final-layer embeddings.

## Limitations
The model depends on large heterogeneous data collection pipelines spanning real, video, and synthetic sources, which the paper does not fully cost-characterize in the available evidence. Simulation benchmarks dominate the quantitative comparisons; real-robot evaluation is centered on the GR-1 humanoid tabletop setting. Per-benchmark numerical tables are not reproduced from the available evidence.
