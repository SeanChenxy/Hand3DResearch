# OakInk2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion

**Authors:** Xinyu Zhan, Lixin Yang, Yifei Zhao, Kangrui Mao, Hanlin Xu, Zenan Lin, Kailin Li, Cewu Lu  
**Date:** 2024-03-28  
**Identifier:** [arXiv:2403.19417](https://arxiv.org/abs/2403.19417); DOI `10.1109/CVPR52733.2024.00050`  
**Zotero item:** `QVCEJ5AW` ([Zotero](zotero://select/library/items/QVCEJ5AW))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

OakInk2 is a large-scale bimanual hand-object interaction dataset of 627 sequences containing 4.01 million image frames, captured with multi-view RGB cameras and an optical motion capture system and annotated with SMPL-X body plus MANO hand fits. Its defining feature is a three-level abstraction that connects object affordances to primitive manipulation tasks and then to long-horizon complex tasks, each represented as a Primitive Dependency Graph. The authors further derive two benchmarks: a hand mesh recovery (HMR) benchmark for monocular and multi-view bimanual reconstruction, and a Task-oriented Motion Fusion (TaMF) task with a dedicated motion-diffusion baseline, together with a Complete Task by Connection (CTC) framework that composes primitive motions into complex-task executions.

## Background and Motivation

Prior hand-object datasets largely capture short, repetitive, single-hand or loosely bimanual interactions, which limits the study of how humans organize manipulation into goal-directed, long-horizon activities. The OakInk2 authors argue that real-world task completion requires understanding at multiple levels of abstraction: the functional affordances of objects, the primitive actions those affordances enable, and the composition of primitives into complex tasks. Capturing this structure in data, they contend, is a prerequisite for both hand motion understanding (reconstruction) and generation (task-oriented synthesis of bimanual manipulation), and no existing dataset jointly provided long-horizon bimanual sequences, dense 3D annotations, and such task-level semantic structure.

## Dataset Construction

Data were recorded in four manipulation scenarios with 75 objects and 9 invited subjects, using 12 OptiTrack Prime 13W motion capture cameras and 4 synchronized commodity RGB cameras at 848 x 480 resolution and 30 fps, one egocentric and three allocentric views. Three levels of abstraction organize the data: 39 object affordances map to 60 types of Primitive Tasks, and 38 long-horizon manipulation goals instantiate 150 Complex Tasks, each encoded as a Primitive Dependency Graph that specifies how primitive actions depend on one another. The corpus comprises 627 sequences in total, 363 Primitive Task sequences and 264 Complex Task sequences, amounting to 4.01M frames. Annotation proceeds through a two-stage body fitting pipeline aligned with MoSH++-based motion capture, from which SMPL-X body parameters are fitted and MANO hand parameters are derived; expert commentary on the manipulations was additionally collected and refined with GPT-4 to support the task-level semantics.

## Evaluation Protocol

Two evaluation tracks are built on the data. The HMR benchmark evaluates hand mesh recovery from monocular and multi-view inputs on a train/test split of the sequences, with baselines including METRO, RLE with HandTailor, a keypoint-based fitting method, and POEM, measured by MPJPE, MPVPE, and AUC-style metrics. The TaMF (Task-oriented Motion Fusion) task asks a model to fuse object-centric motion cues into bimanual hand motion; the authors provide MF-MDM, a motion-fusion conditioned motion diffusion model, evaluated with contact ratio (CR), Single Interaction Volume (SIV), FID, and a human perceptual study scored against ground-truth motions. The CTC framework is demonstrated as a system-level protocol: GPT-4 generates primitive-task programs from language descriptions, oracle trajectories are re-targeted to the subject's body, and TaMF synthesizes the final bimanual motion.

## Findings and Analysis

On the HMR benchmark, bimanual and hand-hand occlusion-heavy frames remain challenging for existing monocular methods, while multi-view input improves reconstruction, quantifying the difficulty gap introduced by two-hand interaction. On TaMF, MF-MDM achieves a contact ratio of 0.90, an SIV of 4.17 cubic centimeters, and an FID of 1.369, and its generated motions score 4.66 +/- 0.48 in the perceptual study versus 3.64 +/- 0.85 for ground-truth sequences under the study's scoring scale, indicating that fused motions are judged comparable to real captures. The CTC experiments show that decomposing a complex goal into a Primitive Dependency Graph and executing primitives sequentially allows the pipeline to complete long-horizon tasks, supporting the paper's claim that the abstraction levels transfer between understanding and generation.

## Contributions

A large-scale bimanual hands-object manipulation dataset (627 sequences, 4.01M frames, four views, dense SMPL-X/MANO annotations); a three-level semantic abstraction linking affordances, primitive tasks, and complex tasks via Primitive Dependency Graphs; a two-track benchmark suite covering bimanual hand mesh recovery and task-oriented motion fusion with strong diffusion-based baselines; and the CTC framework demonstrating language-driven composition of primitive motions into complex task completion.

## Limitations

The paper does not include a dedicated limitations section. Points that can be gathered from the text: articulated object parts are tracked via few markers, which constrains the fidelity of part-level object pose; the perceptual scoring of generated motions is based on a single study protocol; and the authors position the current Primitive Task taxonomy and CTC program generation as a first step, listing extension of the task abstraction and generation quality as future work.
