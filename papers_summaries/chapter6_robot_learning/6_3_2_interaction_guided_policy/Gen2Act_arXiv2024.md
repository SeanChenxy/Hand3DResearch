# Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulation

**Authors:** Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, Sean Kirmani  
**Date:** 2024-09-24 (arXiv preprint, under review)  
**Identifier:** [arXiv:2409.16283](https://arxiv.org/abs/2409.16283)  
**Zotero item:** `TB54WNUX` ([Zotero](zotero://select/library/items/TB54WNUX))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Scaling robot data collection is too expensive to cover everyday manipulation diversity, so Gen2Act instead obtains motion knowledge from web data: it casts language-conditioned manipulation as zero-shot human video generation followed by execution with a single closed-loop policy conditioned on the generated video. The pre-trained video model is used without any fine-tuning, and the translation policy is trained with an order of magnitude less robot interaction data than the video model's training data. On real robots, Gen2Act averages about 30% higher absolute success rate than the most competitive baseline on unseen object types and novel motions, and can be chained for long-horizon activities such as making coffee.

## Background and Problem
The paper asks how manipulation policies can generalize to novel tasks with unseen object types and new motions without test-time adaptation. Prior directions — pre-trained visual encoders, goal-image prediction, hand-object mask plans, point-track prediction from web videos — either require large in-domain robot data, rely on intermediate models for ground truth, or convey only what to do rather than how. The task input is a scene image and a text goal; the output is robot action sequences. The chosen intermediate is a generated human video, since video models trained on web data can render a human performing a novel task zero-shot, whereas they cannot generate robot videos zero-shot.

## Method
Given a scene image and language goal, an off-the-shelf video model (VideoPoet, adapted only to condition on square images plus language) produces a human video of the task. Training pairs are created fully automatically: for each robot demonstration, the video model generates a corresponding human video conditioned on the trajectory's first frame and the task instruction. The translation model is a closed-loop policy conditioned on the last k robot observations and the generated video: ViT features from both videos are resampled through gated cross-attention (PerceiverResampler architecture) into 64 tokens each. An auxiliary point-track prediction loss (tracks computed offline by an off-the-shelf tracker) makes the policy latents informative about scene motion; the track transformer is not used at test time. Actions are discretized into 256 bins per dimension and trained with a cross-entropy behavior-cloning loss. For long-horizon activities, an off-the-shelf LLM (Gemini) splits the goal into sub-tasks, and the last frame of each rollout seeds the next video generation.

## Contributions
- Casting language-conditioned manipulation as zero-shot human video generation plus video-conditioned execution, with no video-model fine-tuning.
- Automatically constructed pairings of robot demonstrations with generated human videos, plus a point-track auxiliary loss distilling motion into policy latents.
- Generalization to unseen object types, novel motions, and long-horizon chaining on real robots with limited robot data.

## Experimental Setup
Experiments run in kitchen, office, and lab scenes on a mobile manipulator with compliant two-finger grippers under end-effector control at 3 Hz. Evaluation follows four generalization levels: Mild (MG), Standard (G), Object-Type (OTG), and Motion-Type (MTG), with success defined by task completion per rollout. Baselines are RT1 (same robot data), RT1-GC (goal-image conditioning on the same videos), Vid2Robot (real paired human-robot videos), and a Gen2Act variant without the track loss. Co-training adds about 400 diverse teleoperated trajectories.

## Results
Gen2Act achieves 83/67/58/30% success on MG/G/OTG/MTG (average 60%), versus Vid2Robot at 83/38/25/0 (37%), RT1-GC at 26% average, RT1 at 22%, and the no-track ablation at 49% (MTG falls to 5%). Chained long-horizon results over 5 trials give, per stage, stowing an apple at 80/60/60%, making coffee at 40/20/20%, cleaning a table at 60/40/40%, and heating soup at 40/20/20%. Co-training with about 400 teleop trajectories raises the average from 60% to 64% (MTG 30% to 35%). At OTG/MTG, implausible generated videos directly cause policy failures.

## Limitations
The authors state that system capability is bounded by current video generation models — for example, their inability to render realistic hands limits very dexterous tasks — and they suggest recovering denser motion information (e.g., object meshes) and learning recovery policies for chaining failures as future work.
