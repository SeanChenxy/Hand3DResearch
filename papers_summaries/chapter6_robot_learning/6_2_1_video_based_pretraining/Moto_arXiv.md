# Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

**Authors:** Yi Chen, Yuying Ge, Weiliang Tang, Yizhuo Li, Yixiao Ge, Mingyu Ding, Ying Shan, Xihui Liu  
**Date:** ICCV 2025  
**Identifier:** DOI `10.1109/ICCV51701.2025.01837`  
**Zotero item:** `GGU9NQZX` ([Zotero](zotero://select/library/items/GGU9NQZX))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Moto argues that motion is the right "language" for video pre-training toward robot manipulation. A Latent Motion Tokenizer converts video into discrete latent Motion Token sequences in an unsupervised manner, and Moto-GPT is pre-trained by next-motion-token autoregression, acquiring motion semantics, trajectory rationality, and future anticipation from videos. A co-fine-tuning strategy then bridges latent motion token prediction with real robot control. The fine-tuned Moto-GPT shows stronger robustness and data efficiency on manipulation benchmarks than counterparts trained without motion priors, and gains further when pre-trained with human videos, supporting cross-embodiment (human-to-robot) transfer.

## Background and Problem
Action-labeled robot data is scarce and expensive, while interaction-rich video is abundant. The paper asks which representation makes generative video pre-training actually useful for manipulation, and proposes motion-related knowledge: it is tied to low-level actions, hardware-agnostic, and transferable. The task is language-conditioned visual manipulation, with pre-training performed purely on video without action annotations.

## Method
The Latent Motion Tokenizer learns to encode the visual motion between frames as discrete tokens. Moto-GPT is then pre-trained with next latent motion token prediction, after which it can produce semantically interpretable motion tokens, predict plausible motion trajectories, and score trajectory rationality through output likelihood. For control, a co-fine-tuning strategy jointly optimizes latent motion token prediction and real robot action prediction, so the motion prior is not discarded when adapting to a robot embodiment.

## Contributions
- Latent Motion Tokens as an unsupervised, hardware-agnostic bridging "language" between video pre-training and robot action learning.
- Moto-GPT pre-training via next-motion-token autoregression, yielding an interpretable motion prior that can also rank trajectory plausibility by likelihood.
- A co-fine-tuning recipe that transfers video-learned motion priors to real robot policies, including evidence of human-video-to-robot transfer.

## Experimental Setup
Evaluation covers robot manipulation benchmarks including SIMPLER and CALVIN-style task suites, comparing Moto against ablations without the motion token and against variants pre-trained on different video corpora (OXE, and additional human video). A probing experiment trains a video classifier on initial-frame ViT patch features plus latent motion tokens to predict semantic labels for 34 CALVIN tasks. Exact trial counts and per-task splits are not reproduced from the available evidence.

## Results
- The probing classifier reaches 79.7% accuracy in predicting semantic labels for 34 CALVIN tasks, indicating that latent motion tokens carry manipulation-relevant semantics.
- On the evaluated manipulation tasks, Moto-GPT consistently outperforms the variant without motion tokens, lifting the average success rate from 23.33% to a substantially higher level in the reported comparison.
- Pre-training with additional video data further improves performance, and human video pre-training boosts the fine-tuned policy, supporting cross-embodiment (human-to-robot) transfer.
- Pre-trained Moto-GPT produces plausible motion trajectory predictions and uses output likelihood to assess trajectory rationality without robot-specific training.

## Limitations
The motion-token objective captures visually observable motion, so manipulation details not visible in video must be learned during fine-tuning. Reported gains are concentrated on the evaluated benchmark suites; broader cross-embodiment deployment is supported by transfer evidence rather than exhaustive multi-robot evaluation. Full per-task tables and failure analyses are not reproduced from the available evidence.
