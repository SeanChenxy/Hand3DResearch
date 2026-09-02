# Wh0: Generative World Models as Scalable Sources of Egocentric Human Hand Manipulation Data

**Authors:** Yangtao Chen, Zixuan Chen, Peiyang Wang, Yong-Lu Li, Jing Huo, Jieqi Shi, Yang Gao  
**Date:** 2026-06-20  
**Identifier:** [arXiv:2606.22136](https://arxiv.org/abs/2606.22136)  
**Zotero item:** `6FU7C8ZL` ([Zotero](zotero://select/library/items/6FU7C8ZL))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Wh0 treats generative video world models as scalable, controllable factories of egocentric human-hand manipulation data rather than as dynamics simulators: conditioned on language, objects, and scenes, it synthesizes WM-H, a 50k-episode dataset of egocentric hand-object interaction videos with 3D hand action labels, converts them into robot-trainable supervision through hand motion reconstruction and robot-hand visual editing, and co-trains a pretrained dexterous VLA policy with limited real robot data. On 18 real-world dexterous manipulation tasks with a Unitree G1 humanoid, this raises zero-shot success on unseen tasks from 8.3% (teleop-only post-training) to 38.9%, a 4.7x gain.

## Background and Problem

Scaling dexterous manipulation requires data that generalize across objects, scenes, and tasks, but existing sources trade off scale against deployment alignment: teleoperation data match robot embodiment and workspaces yet are expensive and platform-specific, simulation is scalable but suffers the sim-to-real gap, and real egocentric video scales well but is misaligned in both scene (everyday environments versus robot workspaces) and embodiment (human hands versus dexterous robot hands). The paper reframes generative world models as compute-driven data engines in which scenes, objects, task types, and embodiment appearance become design variables that scale with GPU hours rather than human labor, targeting the gap between human-video-pretrained VLA models and their dexterous robot deployment.

## Method

- Instruction generation uses a dual-agent LLM system in which one agent discovers object nouns and attribute adjectives while the other preferentially samples under-represented words and assembles templated commands (for example, "pick the red mug"), with a database tracking word frequencies to balance breadth and per-word coverage.
- Scene- and embodiment-aligned video synthesis proceeds in three stages: workspace images are captured with the deployment camera (same viewpoint and resolution as the policy input, with a human hand as a scale anchor) and objects are inserted via Qwen-Image-Edit; Wan-I2V-A14B animates the edited image under the instruction plus a Qwen3-VL-generated dynamics description, accelerated to four inference steps with LightX2V LoRA adapters; and sparsely sampled frames are edited to replace the human hand with a realistic robot dexterous hand while preserving pose, position, and object motion (WM-H EA, for embodiment alignment).
- Action labels are extracted from the human-hand videos with HaWoR, which regresses MANO parameters and wrist poses kept in camera space, with MegaSAM camera tracking used where needed; generation costs about 5.44 GPU-hours per 1,000 81-frame videos.
- The policy is a VITRA-style VLA: a PaliGemma2-3B backbone with a field-of-view token and a cognition token conditions a diffusion DiT-B action decoder that denoises future hand motions in a unified 102-dimensional MANO action space, with robot joints retargeted to MANO and normalized using statistics precomputed from large-scale human videos.

## Contributions

- A reframing of generative world models as scalable, controllable sources of egocentric human-hand manipulation data for VLA post-training, co-trained with limited real robot demonstrations rather than used as environment simulators or trajectory generators.
- The WM-H dataset: 50k egocentric manipulation episodes with language instructions and 3D hand motion annotations (noun h-index 201, adjective h-index 117), produced by a fully automated instruction-generation, scene-aligned synthesis, embodiment-aligned editing, and motion-extraction pipeline.
- A human-robot alignment recipe that bridges human manipulation priors and robot execution through a unified MANO action space, deployment-camera scene alignment, and robot-hand appearance editing that keeps action semantics stable under embodiment changes.
- Real-robot evidence, including ablations, that world-model-generated data with scene and embodiment alignment outperforms substituting real egocentric video, and that the gains come from unlocking pretrained human manipulation priors rather than learning skills from scratch.

## Experimental Setup

- Evaluation uses a Unitree G1 humanoid with Inspire dexterous hands and an egocentric camera (teleoperation via Apple Vision Pro) across 18 real-world tasks in four scenes spanning grasping, placement, and object-specific interactions, each run over 20 trials with randomized object poses; all policies are evaluated zero-shot without task-specific demonstrations, over seen and unseen objects and one seen plus three unseen backgrounds.
- Baselines vary pretraining and adaptation data: pi-0.5 (robot-data pretraining, teleop fine-tuning), VITRA (human-video pretraining, teleop fine-tuning), and VITRA Real Version, which replaces WM-H with real egocentric HOI4D videos during co-training.
- Co-training mixes 50k WM-H samples with 400 teleoperated demonstrations at per-batch ratios of 28% teleop, 68% WM-H, and 4% WM-H EA, with the vision encoder frozen, learning rate 1e-5, batch size 256 on 4 H200 GPUs, and inference on an RTX 4090 with 10-step DDIM sampling.
- A 72-participant user study with AI practitioners assessed WM-H video quality on 5-point Likert scales, plus hand-object grounding measured as hand-object distance on held-out generated instructions and objects.

## Results

- Wh0 reaches 38.9% zero-shot success across the 18 tasks versus 8.3% for VITRA post-trained only on the 400 teleop demonstrations (a 4.7x gain), 21.4% for co-training with real egocentric HOI4D videos instead of WM-H, 8.3% for VITRA teleop fine-tuning alone, and 7.78% for pi-0.5.
- Ablations attribute the gains to alignment and scale: removing scene alignment drops success to 20.0% and removing embodiment alignment to 34.7%, while scaling WM-H from 5k to 25k to 50k samples raises success from 27.8% to 32.5% to 38.9% and improves robot-hand grounding (hand-object distance 9.6 cm at 50k versus 10.5 cm at 5k).
- Prior-unlocking analysis shows WM-H provides little benefit without human-video pretraining (teleop plus WM-H with only PaliGemma pretraining yields 0.6% success), while combining human-video pretraining, teleop data, and WM-H achieves the best grounding and the 38.9% success, indicating WM-H activates pretrained manipulation priors.
- In the user study, 37.7% of AI-generated videos were judged to be real recordings, and synthetic videos scored 4.18 on instruction alignment and 3.95 on hand-object interaction (5-point scale), while robot-hand editing preserved pose consistency (4.30) and contact preservation (4.25).

## Limitations

- The authors note that generation quality limits supervision: the generator can produce physically implausible interactions, unexpected objects, or temporally inconsistent long-horizon videos, and hand occlusions degrade reconstructed finger poses, introducing noisy action labels.
- Human-robot morphology mismatch remains: the robot hand's larger size can unintentionally disturb objects during execution, and WM-H provides little benefit without a strong human-video-pretrained backbone.
- The current experiments are restricted to single-arm pick-and-place-style manipulation, with bimanual coordination, tool use, and longer-horizon tasks left as future work.
