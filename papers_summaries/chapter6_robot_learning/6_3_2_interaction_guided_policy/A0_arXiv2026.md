# A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation

**Authors:** Rongtao Xu, Jian Zhang, Minghao Guo, Youpeng Wen, Haoting Yang, Min Lin, Jianzheng Huang, Zhe Li, Kaidong Zhang, Liqiong Wang, Yuxuan Kuang, Meng Cao, Feng Zheng, and Xiaodan Liang  
**Date:** 2026-01-20  
**Identifier:** [arXiv:2504.12636](https://arxiv.org/abs/2504.12636); DOI `10.48550/arXiv.2504.12636`  
**Zotero item:** `4AC4JCSQ` ([Zotero](zotero://select/library/items/4AC4JCSQ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Complex manipulation requires knowing where to contact an object and how to move after contact, while existing modular and end-to-end systems can lack robust spatial affordance reasoning. A0 decomposes manipulation into high-level spatial-affordance understanding and low-level action execution with a hierarchical affordance-aware diffusion model. Its embodiment-agnostic representation predicts a contact point and a post-contact trajectory, allowing the predicted affordance to be executed on multiple robot platforms. The paper reports pretraining on one million contact-point examples and superior performance across Franka, Kinova, Realman, and Dobot systems.

## Background and Problem
The target problem is general robotic manipulation in which the robot must infer both the location and the manner of interaction with an object. The input is a visual scene and a manipulation task, and the output is an executable robot action produced from a spatial affordance prediction. The paper focuses on complex interactions such as wiping and stacking, where a single undifferentiated action prediction may not expose the relevant contact geometry or subsequent motion. A0 aims to generalize the affordance representation across robot platforms.

## Method
A0 uses a hierarchical diffusion model. The high-level module predicts an embodiment-agnostic affordance consisting of a contact point and the trajectory after contact; the low-level action-execution module converts that prediction into robot control. The model is pretrained on one million contact-point examples and then fine-tuned with annotated trajectories. Position Offset Attention extracts motion-aware features, while a Spatial Information Aggregation Layer maps the predicted spatial information to execution coordinates. These components are included because they connect spatial affordance reasoning to the final action rather than merely classifying an object region.

## Contributions
- A hierarchical separation of spatial-affordance prediction from low-level action execution.
- An embodiment-agnostic affordance representation based on contact points and post-contact trajectories.
- Pretraining and cross-platform evaluation of a general manipulation model on four robot systems.

## Experimental Setup
The paper evaluates manipulation on multiple robotic systems: Franka, Kinova, Realman, and Dobot. The training protocol combines one million contact-point examples for pretraining with annotated trajectories for fine-tuning. The verified evidence reports complex manipulation tasks and cross-platform evaluation, but does not provide complete dataset names, train/test splits, baseline names, or metric definitions; those details are not inferred.

## Results
- A0 is reported to outperform comparison methods on the evaluated complex manipulation tasks across the four listed robot platforms.
- The authors report efficiency, flexibility, and real-world applicability of the hierarchical model, but the available extracted evidence does not contain representative numerical scores.
- The one-million-contact-point pretraining stage is the principal scale detail reported for the affordance prior; no additional ablation values are reproduced here.

## Limitations
The authors do not provide a complete limitation list in the verified evidence. The reported scope is tied to contact-point and post-contact-trajectory affordances and to the evaluated robot platforms; performance outside those interaction types or embodiments is not established by the record. No broader limitation is inferred.
