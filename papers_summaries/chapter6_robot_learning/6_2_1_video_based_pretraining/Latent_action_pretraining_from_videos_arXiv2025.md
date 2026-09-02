# Latent Action Pretraining from Videos

**Authors:** Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo  
**Date:** 2025-05-15 (ICLR 2025)  
**Identifier:** [arXiv:2410.11758](https://arxiv.org/abs/2410.11758); DOI `10.48550/arXiv.2410.11758`  
**Zotero item:** `8FUI3IKS` ([Zotero](zotero://select/library/items/8FUI3IKS))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
LAPA (Latent Action Pretraining) is the first unsupervised method for pre-training Vision-Language-Action (VLA) models without ground-truth robot action labels. It learns discrete latent actions between video frames with a VQ-VAE-style action quantization model, pre-trains a latent VLA to predict those latent actions from observations and task descriptions, and finally fine-tunes on a small robot dataset to map latent actions to real robot actions. The resulting model outperforms the state-of-the-art action-labeled VLA (OpenVLA) by +6.22% on real-world manipulation tasks while using over 30× greater pre-training efficiency, and pre-training on human manipulation videos alone still transfers positively.

## Background and Problem
VLA models traditionally require action labels from human teleoperation during pre-training, which caps both the diversity and scale of usable data. The paper targets language-conditioned real-world manipulation with requirements the paper emphasizes: language conditioning, generalization to unseen objects, and semantic generalization to unseen instructions, all learned from videos that carry no robot actions.

## Method
Training proceeds in three stages. First, latent action quantization: a VQ-VAE-based encoder learns a discrete latent action describing the visual change between two frames, decoded by reconstructing the later frame. Second, latent pre-training: a VLA is trained to predict the latent action from the current observation and task description. Third, action fine-tuning: on a small robot dataset, a mapping from latent actions to executable robot actions is learned. The same trained components can also be read as a neural world model — predicting future frames conditioned on the observation and the VLA-predicted latent action — enabling closed-loop evaluation purely by neural inference.

## Contributions
- The first unsupervised pre-training recipe for VLAs that removes the dependence on ground-truth robot action labels.
- A three-stage pipeline (latent action quantization → latent VLA pre-training → action fine-tuning) applicable to Internet-scale video.
- Evidence of outperforming action-label-pretrained VLAs with far cheaper pre-training, plus positive transfer from human-manipulation-only video.

## Experimental Setup
Evaluation spans simulation and real-world robot manipulation, comparing against (i) methods that train manipulation policies from large-scale videos without action labels and (ii) OpenVLA, the state-of-the-art VLA pre-trained with ground-truth actions on 970k real-robot demonstrations from the Open X-Embodiment dataset. The reported ablations include pre-training on BridgeV2 robot video versus human manipulation video only. Complete task lists, trial counts, and per-benchmark splits are not reproduced from the available evidence.

## Results
- The LAPA-pretrained VLA outperforms OpenVLA by +6.22% on the reported real-world manipulation evaluation while requiring over 30× less pre-training cost.
- Against video-only policy baselines without action labels, LAPA significantly outperforms prior techniques; it also narrows the gap to action-supervised VLAs in simulation despite lacking action labels during pre-training.
- Pre-training on human manipulation video alone yields positive transfer and outperforms models pre-trained on BridgeV2 robot video in the reported comparison.
- Qualitatively, the latent-action decoder doubles as a world model for closed-loop neural simulation.

## Limitations
Latent actions are learned from visual change, so manipulation details invisible in video (forces, in-hand state) are not captured by pre-training and must come from fine-tuning. The reported gains depend on the quality of the quantization codebook and the small-scale action fine-tuning set. Complete coverage of failure cases and the full evaluation protocol are not reproduced from the available evidence.
