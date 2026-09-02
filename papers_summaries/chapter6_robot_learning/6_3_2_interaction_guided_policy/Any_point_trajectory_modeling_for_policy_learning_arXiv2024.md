# Any-point Trajectory Modeling for Policy Learning (ATM)

**Authors:** Chuan Wen, Xingyu Lin, John So, Kai Chen, Qi Dou, Yang Gao, Pieter Abbeel  
**Date:** 2024-07-12 (arXiv preprint)  
**Identifier:** [arXiv:2401.00025](https://arxiv.org/abs/2401.00025)  
**Zotero item:** `BX6TZEY8` ([Zotero](zotero://select/library/items/BX6TZEY8))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Action-labeled robot demonstrations are expensive, while videos are abundant but lack action labels. ATM pre-trains a track transformer on action-free videos to predict the future trajectories of arbitrary points in a frame, conditioned on observation and instruction; predicted trajectories then serve as closed-loop sub-goals for a policy trained with few action-labeled demonstrations. Across over 130 language-conditioned tasks in simulation and on a real robot, ATM averages 63% success versus 37% for the strongest video pre-training baseline, and it transfers skills from human videos and other morphologies.

## Background and Problem
Learning from demonstration improves with more data, but demonstration collection is a bottleneck; videos encode behavior, physics, and semantics yet lack action labels. Prior video pre-training for control mostly relies on pixel-level future-frame prediction, which reconstructs details extraneous to control. ATM's input is an image observation, any set of 2D query points, and a language instruction; its output is their future coordinates over a horizon in the camera frame, which a policy converts into actions. The concrete task is language-conditioned visuomotor manipulation with minimal action-labeled data.

## Method
Stage 1 pre-trains a track transformer on action-free videos: an off-the-shelf vision tracker produces self-supervised track annotations, filtering static points by position variance. Tracks, image patches (50% randomly masked), and BERT-encoded instructions are embedded into shared tokens, and the transformer predicts masked future track positions with an auxiliary masked image-reconstruction loss. Stage 2 trains a track-guided transformer policy on a small action-labeled dataset with MSE loss, consuming the observation plus predicted tracks through early and late fusion. Because tracks already specify fine-grained sub-goals, the policy needs no language instruction and effectively becomes an inverse dynamics model; the frozen track transformer is not fine-tuned.

## Contributions
- Trajectory modeling as video pre-training: predicting any-point future tracks as controllable sub-goals instead of pixel-level video prediction.
- A track-guided policy that reduces imitation to sub-goal following, learning visuomotor control from few action-labeled demonstrations.
- Evidence that the learned trajectory model transfers across embodiments, including human videos and a different robot morphology.

## Experimental Setup
Simulation uses the LIBERO benchmark (Spatial, Object, Goal, Long, and LIBERO-90 suites, over 130 tasks), training each task with 10 action-labeled and 50 action-free robot videos; baselines are behavior cloning (BC), R3M-finetune, VPT, and UniPi, plus an ATM Diffusion Policy variant. Real-world experiments use a UR5 arm with GELLO-collected demonstrations (50 action-labeled trajectories, 250 action-free videos) and five dining-table tasks. Cross-embodiment studies add 100 human videos per task and 160 Franka videos for a UR5 pick-and-place task.

## Results
On LIBERO, ATM averages 63% success against 37% for the best prior baseline, with the largest gains on LIBERO-Goal and LIBERO-Long; ATM Diffusion Policy consistently improves the base diffusion policy across suites. With 4% of demonstrations, ATM matches BC trained on 20% on three suites. Human-to-robot transfer on fold-cloth/put-tomato/sweep-toys yields 63%/63%/60% success with human videos versus 0%/0%/13% with teleoperation data alone. Ablations show a track horizon of 16 steps is optimal, image masking generally helps, and late track fusion is the most critical policy component (early fusion alone drops LIBERO-Goal to 5.33%).

## Limitations
The authors state that ATM still requires action-labeled demonstrations for mapping trajectories to actions, limiting policy generalization, and that its video data contain only small domain gaps; learning from in-the-wild videos with multi-modal distributions, diverse camera motions, and sub-optimal motions is left as future work.
