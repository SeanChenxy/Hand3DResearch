# HunyuanVideo: A Systematic Framework For Large Video Generative Models

**Authors:** Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, Caesar Zhong  
**Date:** 2024-12-03  
**Identifier:** [arXiv:2412.03603](https://arxiv.org/abs/2412.03603)  
**Zotero item:** `9CZ6RWWS` ([Zotero](zotero://select/library/items/9CZ6RWWS))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
HunyuanVideo is Tencent's open-source video foundation model with over 13 billion parameters, the largest among open-source models at release. The technical report covers the full framework: data curation with structured captioning, a dual-stream-to-single-stream diffusion transformer with a causal 3D VAE and an MLLM text encoder, scaling-law-guided model sizing, and efficient training/inference infrastructure. In professional human evaluation over 1,533 prompts by 60 evaluators, it ranks first in overall preference (41.3%) ahead of Runway Gen-3 alpha (27.4%) and Luma 1.6 (24.8%), excelling in motion quality.

## Background and Problem
Leading video generation models remain closed-source, creating a performance gap between industry capabilities and the public. Unlike image generation, open video models lack the full-stack recipes needed to close this gap. HunyuanVideo aims to provide a systematic, open framework spanning data preprocessing, architecture, scaling, training infrastructure, and inference acceleration, targeting four quality axes: visual quality, motion dynamics, text-video alignment, and advanced filming techniques (including semantic scene cut).

## Method
Data: image-video joint training with five video groups and two image groups, multi-stage filtering (including motion-connectivity criteria), a billions-scale image pool for stage-1 pretraining, and a hundreds-of-millions subset for stage 2. An in-house structured captioning model produces JSON captions covering subject, dense description, background, style, shot type, lighting, and atmosphere. Architecture: a Causal 3D VAE (CausalConv3D) compresses video 4x in time and 8x8 in space with 16 latent channels; the diffusion backbone uses flow matching in a "dual-stream to single-stream" design with 20 dual-stream blocks followed by 40 single-stream blocks (model dimension 3072, FFN 12288, 24 heads, RoPE with head dims 16/56/56 for t/h/w). Text conditioning uses a decoder-only MLLM (better image-text alignment and reasoning than T5/CLIP) with a bidirectional token refiner, plus a CLIP pooled embedding added to the timestep embedding. Scaling: power-law fits on model families from 92M to 6.6B parameters yield compute-optimal sizing, reducing required compute by up to 5x; progressive curriculum training goes from low-resolution short videos to high-resolution long videos. Inference: text-guidance distillation gives about 1.9x acceleration, and training runs on the AngelPTM infrastructure with model parallelism.

## Contributions
(1) The largest open-source video generation model at the time (13B parameters) with released code and applications. (2) A systematic, reproducible framework covering data curation, structured captioning, architecture, scaling-law-based sizing, and efficient training/inference. (3) Empirical human evaluation showing parity or superiority over leading closed-source systems, particularly in motion dynamics.

## Experimental Setup
Professional human evaluation uses 1,533 representative text prompts and a panel of 60 people, comparing against Runway Gen-3 alpha, Luma 1.6, and three top Chinese commercial models, scoring text alignment, motion quality, visual quality, and overall ranking; 600 sampled videos are released for public inspection.

## Results
In the Table 3 comparison, HunyuanVideo achieves overall ranking 41.3% (rank 1), text alignment 61.8%, motion quality 66.5%, and visual quality 95.7%, versus CNTopA (37.7%, rank 2), CNTopB (37.5%, rank 3), GEN-3 alpha (27.4%, rank 4), Luma 1.6 (24.8%, rank 5), and CNTopC (24.6%, rank 6), with the largest margin in motion quality. Text-guidance distillation provides roughly 1.9x inference acceleration, and the compute-optimal scaling strategy cuts training compute by up to 5x.

## Limitations
The paper has no dedicated limitations section; it is a technical report. Implicitly, evaluation relies on human preference panels rather than standardized automated benchmarks, and comparisons are limited to the five competitor models listed, at 5-6 second generation durations in the reported table. Broader quantitative failure-mode analysis is not reported in the paper.

