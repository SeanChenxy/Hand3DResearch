# H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation

**Authors:** Hongzhe Bi, Lingxuan Wu, Tianwei Lin, Hengkai Tan, Zhizhong Su, Hang Su, Jun Zhu  
**Date:** 2025-08-01  
**Identifier:** [arXiv:2507.23523](https://arxiv.org/abs/2507.23523)  
**Zotero item:** `TMNZCWNT` ([Zotero](zotero://select/library/items/TMNZCWNT))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

H-RDT (Human to Robotics Diffusion Transformer) leverages large-scale egocentric human manipulation data to improve bimanual robotic manipulation. Robot demonstration data is scarce, and cross-embodiment robot pretraining is hampered by heterogeneous morphologies, whereas egocentric human videos with 3D hand poses offer abundant, unified-embodiment behavioral priors. H-RDT is a 2B-parameter diffusion transformer trained with flow matching in two stages: pre-training on the full EgoDex dataset with a 48-dimensional hand action representation, then cross-embodiment fine-tuning with modular action encoders and decoders. It outperforms training from scratch and the baselines RDT and Pi0 across simulation and real-world settings, improving over from-scratch training by 13.9% in simulation and 40.5% in the real world.

## Background and Problem

Robotic imitation learning depends on expensive teleoperated demonstrations, and VLA pretraining on multi-robot datasets faces conflicting morphologies and inconsistent data quality. The paper hypothesizes that egocentric human manipulation videos with 3D hand poses capture natural manipulation strategies, object affordances, and task decomposition patterns transferable to robots. The task is conditional action-sequence generation: given multi-view RGB observations, proprioceptive state, and a language instruction, the policy predicts future bimanual actions. Three challenges are identified: prior human-data methods operate at modest scale, human-robot embodiment differences impede transfer, and adapting one pretrained model to multiple embodiments efficiently remains open.

## Method

H-RDT represents human actions as compact 48-dimensional vectors combining bilateral wrist poses (18 dimensions, identical to robot end-effector poses) and fingertip positions (30 dimensions), a superset of most end-effector-controlled robots' action spaces that mitigates embodiment mismatch. Stage 1 pre-trains on the complete EgoDex dataset (338K+ trajectories, 194 tasks). Stage 2 transfers the DinoV2 and SigLIP vision encoders, the T5-XXL language encoder, and the LLaMA-3-style transformer backbone to a target robot, while the state adapter, action adapter, and action decoder are reinitialized for the target action space. Action generation uses flow matching, learning a vector field from noise to the action distribution integrated at inference by an ODE solver; image and language features enter via cross-attention.

## Contributions

- A framework systematically using large-scale egocentric human manipulation data as pre-training, orders of magnitude beyond prior human-data works.
- A diffusion transformer with modular action encoders and decoders enabling cross-embodiment transfer without relearning visual-semantic representations.
- A two-stage pre-training and fine-tuning paradigm based on flow matching for stable, efficient policy learning.
- Validation showing consistent gains over RDT, Pi0, and from-scratch training across simulation, real robots, and few-shot settings.

## Experimental Setup

Simulation uses RoboTwin 2.0 with Easy (clean) and Hard (domain-randomized) modes on Aloha-Agilex-1.0 and dual-arm Franka-Panda, in single-task (13 tasks, 50 demonstrations each) and multi-task (roughly 2250 demonstrations) configurations. Real-world experiments cover Aloha-Agilex-2.0 on towel folding and cup-to-coaster placement (25 trials per task), a dual-arm ARX5 few-shot suite of 113 pick-and-place tasks with only 1-5 demonstrations per task, and a dual UR5 + UMI bimanual takeout-bag task with four subtasks. Baselines are RDT, Pi0, and H-RDT trained from scratch without human data.

## Results

In RoboTwin 2.0 multi-task Hard-mode evaluation, H-RDT reaches 87.2% average success versus 28.8% for RDT, 48.4% for Pi0, and 67.2% for from-scratch training; cross-embodiment results are 87.2% on Aloha-Agilex-1.0 and 62.9% on Franka-Panda. Single-task averages are 68.7% (Easy) and 25.6% (Hard). On real Aloha-Agilex-2.0, H-RDT achieves 52% on towel folding and 64% on cup-to-coaster placement versus 0% and 20% without human pre-training. In the 113-task few-shot setting it scores 41.6% versus 16.0% (RDT), 31.2% (Pi0), and 17.6% (from scratch), and on UR5+UMI it averages 58.0% versus 29.0-31.0%. Gains over from-scratch training total 13.9% in simulation and 40.5% in the real world.

## Limitations

Single-task performance under Hard-mode domain randomization (25.6%) remains far below the Easy-mode level, indicating a robustness gap. The appendix notes that concatenating three 240x320 views into one input for training speed may cause some degradation relative to higher-resolution settings, and cross-embodiment deployment still requires reinitializing and retraining the modular action components per robot.
