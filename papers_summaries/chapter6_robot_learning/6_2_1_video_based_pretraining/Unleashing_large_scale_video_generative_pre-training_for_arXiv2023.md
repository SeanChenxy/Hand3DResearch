# Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation

**Authors:** Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, Tao Kong  
**Date:** 2023-12-21  
**Identifier:** [arXiv:2312.13139](https://arxiv.org/abs/2312.13139)  
**Zotero item:** `9G23VIKN` ([Zotero](zotero://select/library/items/9G23VIKN))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
GR-1 shows that a single GPT-style transformer can be pre-trained on large-scale unlabeled video and then fine-tuned into a strong multi-task, language-conditioned robot manipulation policy. The model consumes a language instruction, a sequence of camera observations, and a sequence of robot states, and it predicts both future images and robot actions end-to-end. On the CALVIN benchmark, GR-1 raises the reported success rate from 88.9% to 94.9% over prior baselines, and in zero-shot unseen-scene generalization it raises the rate from 53.3% to 85.4%. The result is an early, direct demonstration that video generative pre-training transfers to visual robot control.

## Background and Problem
Generative pre-training has been decisive in language and vision, but robot manipulation policies are still trained mostly on limited teleoperated robot data. The paper asks whether large-scale video generative pre-training can supply transferable world knowledge for manipulation. The task is multi-task, language-conditioned visual robot manipulation: given an instruction, observation images, and robot states, produce executable robot actions, with future-image prediction used as an auxiliary generative objective during pre-training and fine-tuning.

## Method
GR-1 is a GPT-style transformer trained end-to-end. During pre-training on a large-scale video dataset, the model learns to generate future video frames, forcing it to model scene dynamics and object motion. The same architecture is then fine-tuned on robot data, where it takes the language instruction, the observation-image sequence, and the robot-state sequence as input and autoregressively predicts robot actions together with future images. The shared design makes the video-pretrained weights directly reusable for action prediction without architectural surgery.

## Contributions
- A unified GPT-style architecture that uses video generative pre-training as the pre-training stage for visual robot manipulation.
- End-to-end joint prediction of future images and robot actions from instruction, observations, and robot states.
- Evidence on CALVIN and a real robot that video pre-training improves multi-task performance and generalization to unseen scenes, objects, and instructions.

## Experimental Setup
Evaluation uses the CALVIN long-horizon benchmark (34 tasks, a Franka Emika Panda with a parallel-jaw gripper) under the ABCD→D and ABC→D protocols, reporting success rates over chains of 1–5 sequential tasks, plus unseen-language and reduced-data (10%) settings. Baselines include MCIL, RT-1, HULC, and MT-R3M. Real-robot experiments cover multi-task manipulation with novel scenes and objects. The full real-robot task list and trial counts are not reproduced from the available evidence.

## Results
- CALVIN ABCD→D: GR-1 reports 94.9% average chain success versus 88.9% for the best baseline (HULC); GR-1 also leads at every chain length (0.949/0.896/0.844/0.789/0.731).
- CALVIN ABC→D (zero-shot unseen scenes): GR-1 reports 85.4% versus 53.3% for the best baseline.
- With 10% of the fine-tuning data, GR-1 reports 77.8% versus 66.8% for the best baseline, indicating improved data efficiency.
- Unseen-language evaluation: GR-1 reports the highest average among the compared methods. Real-robot experiments report consistent outperformance of baselines with qualitative generalization gains.

## Limitations
The paper's generative pre-training objective is future-image prediction, so transfer depends on how well video dynamics cover the target manipulation distribution; the paper does not characterize failure regimes in detail in the available evidence. Real-robot evaluation is limited to the reported setups, and the pre-training corpus is large-scale video rather than robot data, leaving the action gap to be bridged by fine-tuning. Quantitative real-robot comparisons beyond the reported summaries are not reproduced from the available evidence.
