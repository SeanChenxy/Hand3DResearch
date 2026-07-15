# ClickDiff: Click to Induce Semantic Contact Map for Controllable Grasp Generation with Diffusion Models

## Summary
Enables user-controllable grasp generation by allowing users to "click" on desired object regions to specify semantic contact points, which a diffusion model then expands into a full contact map and optimizes into a MANO grasp.

## 1. Problem and Setting
- Task: controllable grasp generation where a user specifies desired contact regions on an object via 2D/3D clicks, and the system generates a plausible hand grasp respecting those user constraints.
- Input: 3D object mesh + user-specified contact points (clicks) on the object surface + optional semantic hand-part labels for each click; Output: MANO hand grasp.
- Key challenge: pure generative methods produce diverse but uncontrollable grasps; explicit user control (e.g., "grasp the handle" or "pinch the rim") requires a mechanism to convert sparse user input into a complete, physically plausible contact configuration.

## 2. Core Method
- Diffusion-based contact completion: a 3D diffusion model (on object surface points) is trained to generate full object-surface contact maps (contact probability + hand-part per point). User clicks act as conditioning — they are encoded as sparse "anchor" tokens that the diffusion model must satisfy while generating the rest of the contact map.
- Semantic click encoding: each user click can optionally specify which hand part should contact that point (e.g., "thumb," "index finger"), encoded as a one-hot part label.
- Grasp fitting: same as ContactGen-style optimization — the completed contact map guides MANO parameter fitting via energy minimization.
- Key innovation: enabling intuitive user-in-the-loop control over grasp generation via sparse clicks, bridging unconditional generative modeling and user-guided synthesis.

## 3. Knowledge, Supervision, and Assumptions
- Training data: GRAB, ObMan, plus synthetic data.
- Supervision: per-object-point contact maps and hand-part labels derived from MANO fits.
- Domain knowledge: MANO model with part segmentation.
- Assumption: object geometry is known and static; single-hand grasp only.

## 4. Experiments and Findings
- Datasets: GRAB, ObMan; user study with non-expert participants asked to click desired grasp points on novel object meshes.
- Metrics: contact completion accuracy (given partial clicks, does the model fill in a valid full contact map?), user preference, contact IoU, penetration.
- Main findings: ClickDiff generates grasps that consistently satisfy user-specified contact points while maintaining overall physical plausibility; users prefer ClickDiff grasps over unconditional methods for task-specific grasping scenarios.

## 5. Strengths and Limitations
### Strengths
- Intuitive user interface for grasp control — clicking is much easier than specifying MANO parameters.
- Diffusion model naturally handles sparse-to-dense contact completion.

### Limitations
- Click placement requires some understanding of object functionality; naive clicks may lead to implausible completed contact maps.
- Still inherits the speed limitations of iterative grasp optimization.
- Limited to static grasping; no temporal or dynamic control.

## 6. Takeaway
ClickDiff brings user controllability to diffusion-based grasp generation, showing that sparse semantic clicks can effectively steer a generative contact model toward task- or user-specific outcomes. This work bridges the gap between fully automatic grasp generation and the practical need for user-guided, intent-driven synthesis in interactive applications.
