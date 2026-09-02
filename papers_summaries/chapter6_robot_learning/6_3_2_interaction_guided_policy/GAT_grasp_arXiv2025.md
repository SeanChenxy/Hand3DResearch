# GAT-Grasp: Gesture-Driven Affordance Transfer for Task-Aware Robotic Grasping

**Authors:** Ruixiang Wang, Huayi Zhou, Xinyue Yao, Guiliang Liu, and Kui Jia  
**Date:** 2025-03-08  
**Identifier:** [arXiv:2503.06227](https://arxiv.org/abs/2503.06227); DOI `10.48550/arXiv.2503.06227`  
**Zotero item:** `K5CBJACJ` ([Zotero](zotero://select/library/items/K5CBJACJ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Task-aware grasping requires selecting both the appropriate object and a grasp pose for the intended use, but visual affordances can be ambiguous and conventional object priors do not cover unseen objects well. GAT-Grasp treats a human hand gesture as an intent signal and transfers gesture-associated affordances from large-scale human-object interaction videos to a robot grasp. Its retrieval-based pipeline produces task-specific grasp position and orientation without a pre-given object prior, enabling open-set and cluttered-scene evaluation. The reported real-world results include a 51.67% success rate in cluttered scenes, while the record also reports robust execution in diverse unseen scenarios.

## Background and Problem
A robot may need to grasp an object differently depending on the task, so a geometrically valid grasp is not necessarily functionally appropriate. Existing methods can be limited by ambiguous affordance reasoning and by object priors that do not transfer to novel categories. GAT-Grasp takes a visual scene together with a human gesture that conveys task intent and outputs a task-specific grasp pose, including its position and orientation, for robotic execution. The method uses human-object interaction videos as a source of gesture and affordance information.

## Method
GAT-Grasp uses the implicit relationship between a human gesture and the affordance required by a task. It retrieves relevant human-object interaction examples, transfers their affordance information to the target scene, and maps the resulting hand-level grasp to a robot grasp. The verified full-text extraction identifies a DIFT-based affordance-transfer component and a hand-to-gripper rotation mapping. The design avoids requiring a pre-specified object prior at inference time, so the gesture can guide grasping for novel objects and cluttered scenes.

## Contributions
- A gesture-driven formulation that uses human hand motion as a task-intent cue for grasp selection.
- A retrieval-based affordance-transfer pipeline that draws grasping knowledge from large-scale human-object interaction videos.
- A robot grasp mapping and evaluation of open-set, cluttered-scene execution.

## Experimental Setup
The paper evaluates task-aware grasp execution in real-world scenes, including novel objects and cluttered environments. The verified evidence identifies the cluttered-scene evaluation and the DIFT-based transfer ablation, but does not provide complete dataset names, the full baseline list, or all metric definitions. The reported success rate is a task-execution success measure; the complete evaluation protocol is not reproduced because it is not available in the current extracted evidence.

## Results
- The reported success rate in cluttered scenes is **51.67%**.
- Real-world evaluations are reported to remain robust across diverse and unseen scenarios.
- The paper attributes the transfer capability to using gesture-affordance correlation rather than relying on pre-given object priors; complete ablation numbers are not reported in the available evidence.

## Limitations
The authors do not state a complete limitation list in the verified evidence. The method requires a meaningful human gesture as an intent signal and depends on human-object interaction data for retrieval-based affordance transfer; these are direct scope requirements rather than claims about untested failure cases. No additional limitation is inferred.
