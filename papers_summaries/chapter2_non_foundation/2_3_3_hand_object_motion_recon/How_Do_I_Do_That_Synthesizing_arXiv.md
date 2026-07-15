# How Do I Do That? Synthesizing 3D Hand Motion and Contacts for Everyday Interactions

## Summary
Generates 3D hand motion and detailed contact maps for everyday object interactions from natural language descriptions of tasks, using a retrieval-augmented diffusion framework.

## 1. Problem and Setting
- Generate 3D hand motion sequences and detailed hand-object contact maps for daily activities, given a textual task description.
- Input: natural language task description (e.g., "open a drawer", "pour tea"); output: time-varying MANO hand poses + contact maps on the object surface.
- Text-to-motion generation. Focuses on everyday manipulation tasks with known object categories.

## 2. Core Method
- Retrieval-augmented generation pipeline:
  1. A retrieval module finds similar interactions from a motion database given the text query.
  2. A diffusion model generates the hand motion sequence, conditioned on the retrieved examples, the text embedding, and the object point cloud.
  3. A contact predictor produces per-frame contact maps (which hand vertices contact which object surface points).
- The retrieval augmentation helps the model produce realistic motions for rare tasks by finding structurally similar examples.
- Contact maps are generated as an auxiliary output and can be used to refine or evaluate the interaction quality.

## 3. Knowledge, Supervision, and Assumptions
- Training data: large-scale HOI motion datasets (GRAB, ARCTIC, plus custom captured daily activities).
- Supervision: MANO parameters, object contact labels.
- Uses MANO for hand.
- Assumes task description is specific enough to identify the interaction type; object 3D model available.

## 4. Experiments and Findings
- Datasets: GRAB, custom daily activity captures.
- Metrics: FID, diversity, contact accuracy, text-motion alignment.
- Retrieval augmentation improves motion quality for rare tasks. Generated contacts align well with human annotations.

## 5. Strengths and Limitations
### Strengths
- Retrieval augmentation addresses data scarcity for rare tasks.
- Generates both motion and contact maps, providing richer output.
- Text interface is intuitive and flexible.

### Limitations
- Requires a motion database for retrieval.
- Object 3D model must be provided.
- Generated contact maps may be physically imprecise.
- Limited to tasks in the training/retrieval distribution.

## 6. Takeaway
This paper demonstrated that retrieval augmentation is a practical way to improve HOI motion generation, especially for long-tail tasks. The simultaneous generation of motion and contact maps is a useful capability for downstream applications like robot learning from demonstration.
