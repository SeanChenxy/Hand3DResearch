# CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos

**Authors:** Chubin Zhang, Jianan Wang, Zifeng Gao, Yue Su, Tianru Dai, Cai Zhou, Jiwen Lu, Yansong Tang  
**Date:** 2026  
**Identifier:** [arXiv:2601.04061](https://arxiv.org/abs/2601.04061)  
**Zotero item:** `24YW5KBX` ([Zotero](zotero://select/library/items/24YW5KBX))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
CLAP addresses the visual entanglement problem of latent action models trained on human videos: latent codes end up capturing scene noise instead of manipulation skills. It contrastively aligns the video-derived visual latent space with a proprioceptive latent space built from robot trajectories, mapping video transitions onto a quantized, physically executable action codebook. On this representation it builds two VLAs — CLAP-NTP, an autoregressive model for instruction following and object generalization, and CLAP-RF, a Rectified Flow policy for high-frequency precise manipulation — plus a Knowledge Matching regularizer against catastrophic forgetting. On a real Astribot S1, CLAP-RF reports the highest mean success rate (61.0%) among compared generalist baselines.

## Background and Problem
Generalist VLAs are bottlenecked by scarce robot data relative to abundant human video. Prior latent action models pre-trained on video often learn entangled visual factors rather than transferable manipulation skills. CLAP defines the task as transferring manipulation skills from human videos (including egocentric sources such as Ego4D) to robotic execution, with the latent action space constrained to be physically executable by a robot.

## Method
CLAP learns its action codebook by contrastive alignment: video-frame transitions and the corresponding robot proprioceptive transitions are pulled together in a shared latent space, then quantized, so each latent action corresponds to an executable robot motion. Decoding a latent action yields a 3D trajectory projected onto the image plane, which the paper uses to visualize semantic alignment across robot domains (Astribot, AgiBot) and human video (Ego4D) — clustered tokens correspond to actions such as moving right, placing, and grasping. On top of the codebook, CLAP-NTP is an autoregressive VLA, and CLAP-RF is a Rectified Flow policy for precise high-frequency control. Knowledge Matching regularization preserves pretrained skills during fine-tuning.

## Contributions
- Contrastive video–proprioception alignment that turns latent action pre-training into a physically executable codebook.
- A dual-formulation VLA family (autoregressive CLAP-NTP and flow-based CLAP-RF) over the shared representation.
- Knowledge Matching regularization mitigating catastrophic forgetting, and real-robot evidence of skill transfer from human videos.

## Experimental Setup
Real-world experiments use the Astribot S1 dual-arm robot (14-DoF arms plus grippers, locked chassis/torso) with a head camera and two wrist cameras, in a live demonstration-to-execution pipeline. Five tasks evaluate distinct capabilities, including pick-and-place with seen and strictly out-of-distribution objects (20 episodes per setting), long-horizon multi-stage packing, and fine-motor deformable tasks such as cloth folding and gift packing. Robustness is evaluated under background change, lighting variation, and novel objects against π0, π0.5, and UniVLA. Pre-training spans Astribot, AgiBot, and Ego4D data; full pre-training corpus statistics are not reproduced from the available evidence.

## Results
- CLAP-RF achieves the highest mean success rate across all tasks (61.0%), outperforming π0 (54.0%), π0.5 (60.0%), and UniVLA (35.0%) in the reported comparison.
- On deformable fine-motor tasks (cloth folding, gift packing), CLAP-RF outperforms the strong generalist baselines.
- Robustness evaluation under perturbations: CLAP-RF reports a mean 66.7% versus 56.7% for π0.5, 46.7% for π0, and 16.7% for UniVLA across the perturbed settings.
- With human egocentric video added to fine-tuning, CLAP-NTP outperforms the compared baselines in the reported generalization comparison.

## Limitations
The evaluation is concentrated on one robot platform and five task families, so cross-embodiment breadth is demonstrated qualitatively (latent-space alignment visualizations) rather than through extensive multi-robot benchmarks. The contrastive codebook depends on the availability of proprioceptive robot data for alignment, which partially conditions the claimed scalability from human video. Complete ablations over codebook size and pre-training mixture are not reproduced from the available evidence.
