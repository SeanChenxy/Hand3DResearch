# DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation

**Authors:** Zhao Mandi, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, Shuran Song  
**Date:** 2025-05-30  
**Identifier:** [arXiv:2505.24853](https://arxiv.org/abs/2505.24853); DOI `10.48550/arXiv.2505.24853`  
**Zotero item:** `P9PAL4AJ` ([Zotero](zotero://select/library/items/P9PAL4AJ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
DexMachina studies functional retargeting: learning dexterous bimanual policies that make an object track the states shown in a human hand-object demonstration, instead of merely imitating human-like hand motion. For long-horizon tasks with articulated objects, it proposes a curriculum-based RL algorithm whose core idea is virtual object controllers with decaying strength that first drive the object along the demonstrated trajectory while the policy learns under motion and contact guidance, then hand control over to the policy. The paper also releases a simulation benchmark with six dexterous hands and five articulated objects, on which DexMachina clearly outperforms baselines and enables functional comparisons between hand hardware designs.

## Background and Problem
Learning-based dexterous manipulation has mostly succeeded on short-horizon tasks and is bottlenecked by manual reward engineering or costly embodiment-specific data collection. Human demonstrations are a natural guidance source, but kinematic retargeting produces human-like motions without guaranteeing feasibility, and the embodiment gap prevents direct imitation at scale. The formulated task takes one densely tracked human demonstration of an articulated object (part poses plus joint angles) and a pair of dexterous robot hands, and learns a policy minimizing accumulated tracking error of the object states—challenging because of the high-dimensional action space, intricate contact sequences, and spatiotemporally discontinuous bimanual coordination (e.g., repositioning one hand mid-air while the other holds the object).

## Method
DexMachina is a PPO-based RL method in the Genesis simulator. Demonstration pre-processing runs collision-aware kinematic retargeting to obtain reference joints and hand keypoints, plus distance-based approximations of where and when each hand link should contact each object part. The task reward is a product of position, rotation, and articulation tracking terms; auxiliary rewards encourage keypoint matching, joint-level behavior cloning, and contact matching. A hybrid action space adds policy residuals to retargeted wrist base actions and uses normalized absolute finger actions. The key component is the auto-curriculum with virtual object controllers: six virtual 1-DoF joints for base pose and one for articulation apply PD forces pulling the object toward the demonstrated states, with gains initialized high and exponentially decayed once normalized reward statistics pass thresholds, so the policy progressively takes over manipulation.

## Contributions
- The functional retargeting problem formulation and DexMachina, a curriculum RL algorithm over virtual object controllers with motion and contact guidance.
- A simulation benchmark with 6 curated dexterous hand assets and 5 articulated objects for evaluating both retargeting algorithms and hand designs.
- State-of-the-art retargeting performance across hands and tasks, plus hardware-design findings derived from the benchmark.

## Experimental Setup
Training uses ARCTIC hand-object clips (5 articulated objects, 7 demonstrations including short- and long-horizon sequences), Genesis physics, PPO, and up to 12,000 parallel environments, with 5 random seeds per hand-task pair. Evaluation uses a per-part ADD-AUC metric over 20 episodes per checkpoint. Baselines: direct replay of kinematic retargeting, a re-implemented ObjDex (task reward with hybrid actions), task-plus-auxiliary-rewards without curriculum, and a re-implemented ManipTrans curriculum. Four hands (Inspire, Allegro, XHand, Schunk) are used in main results; Ability and DexRobot Hand are added for the embodiment study.

## Results
DexMachina consistently improves success on all four hands and seven tasks, with the largest gains on long-horizon clips (e.g., Notebook-300, Mixer-300), where auxiliary rewards alone give inconsistent gains and the curriculum is decisive. Kinematics-only replay cannot lift objects beyond slight movement. The ObjDex re-implementation exceeds its original reported numbers (over 90% success on Ketchup-100 versus 41.2% originally), yet still trails DexMachina on long-horizon tasks; the ManipTrans-style physics-parameter curriculum is unstable and drops in performance as training progresses. Ablations show restrictive hybrid actions beat absolute or weakly-constrained residual actions. Embodiment analysis finds larger fully-actuated hands (e.g., Allegro) learn faster and better, degrees of freedom matter more than size similarity (Schunk outperforms similarly sized Inspire and Ability), and less-actuated hands develop less human-like strategies.

## Limitations
The authors note that policies consume state-based privileged simulator information not readily available on real robots; the formulation assumes high-quality, expensive-to-collect tracked demonstrations (ARCTIC uses mocap with dense manual annotation); hand simulation models are estimated from open-source assets and may misrepresent real dynamics; and no real-world evaluation on the examined hands has been performed, with sim-to-real left to future distillation work.
