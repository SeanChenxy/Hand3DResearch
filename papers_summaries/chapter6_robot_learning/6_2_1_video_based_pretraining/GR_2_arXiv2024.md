# GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation

**Authors:** Chi-Lam Cheang, Guangzeng Chen, Ya Jing, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Hongtao Wu, Jiafeng Xu, Yichu Yang, Hanbo Zhang, Minzhao Zhu (ByteDance Research; alphabetical order)  
**Date:** 2024-10-08  
**Identifier:** [arXiv:2410.06158](https://arxiv.org/abs/2410.06158)  
**Zotero item:** `2IJFQ6CJ` ([Zotero](zotero://select/library/items/2IJFQ6CJ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
GR-2 is a generalist video-language-action (VLA) robot agent pre-trained on 38 million Internet video clips (over 50 billion tokens) to absorb world dynamics, then fine-tuned on robot trajectories for both video generation and action prediction. In real-robot multi-task evaluation it reports a 97.7% average success rate across more than 100 tasks, and it generalizes to novel backgrounds, environments, objects, and tasks. The model also scales positively with size. Together with GR-1, it establishes web-scale video pre-training as a practical foundation for versatile manipulation.

## Background and Problem
A generalist manipulation agent must acquire many skills and adapt to new tasks and disturbances, but robot-collected data alone is too small to support such breadth. The paper defines the task as language-conditioned multi-task visual manipulation from camera observations, with pre-training on Internet video to inject broad, web-scale knowledge of how the world and hands behave before any robot-specific training.

## Method
GR-2 first pre-trains on a large Internet video corpus to model generic scene and hand-object dynamics. It is then fine-tuned on robot trajectories with two coupled objectives: video prediction of the manipulated scene and autoregressive action prediction, so that the model grounds language and observations in executable control. The default model has 230M parameters, of which 95M are trainable, and the paper reports controlled model-scaling experiments.

## Contributions
- Web-scale video pre-training (38M clips, 50B+ tokens) as the foundation stage of a manipulation generalist.
- A fine-tuning scheme that jointly learns video generation and action prediction on robot data.
- Demonstration of state-of-the-art multi-task success, out-of-distribution generalization, and favorable model scaling.

## Experimental Setup
Large-scale real-robot experiments cover two settings: multi-task learning over more than 100 tasks spanning 8 skill types (picking, placing, uncapping, capping, opening, closing, pressing, pouring), evaluated under Simple and progressively harder out-of-distribution settings (novel backgrounds, environments, objects, and tasks), and end-to-end bin picking in an industrial-style cluttered setting with a single text prompt. A CALVIN benchmark comparison against state-of-the-art methods, including GR-1, is also reported. Baseline definitions beyond the named settings are not fully reproduced from the available evidence.

## Results
- Multi-task real-robot evaluation: an average success rate of 97.7% across 105 tasks in the Simple setting.
- GR-2 reports improved success rates over GR-1 across the evaluated generalization settings (novel backgrounds, environments, objects, and tasks).
- End-to-end bin picking and the CALVIN comparison are reported as supporting evidence for generality; the full per-setting success tables are not reproduced from the available evidence.
- Model-scaling experiments report consistent gains with size, supporting continued growth.

## Limitations
The reported evaluation is real-robot-centric and proprietary-data-heavy, so independent reproduction depends on the released resources rather than the paper alone. Out-of-distribution gains, while consistent, are summarized at a level that does not expose per-task failure cases in the available evidence. The pre-training corpus is Internet video, so skills absent from that distribution still must come from fine-tuning; the paper does not quantify this boundary.
