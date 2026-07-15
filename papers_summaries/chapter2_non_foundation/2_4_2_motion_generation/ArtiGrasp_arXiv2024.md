# ArtiGrasp: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation

## Summary
Extends physics-based hand-object interaction synthesis to bimanual grasping and articulation of articulated objects (e.g., opening drawers, turning knobs), using reinforcement learning with a novel curriculum that progressively increases task difficulty.

## 1. Problem and Setting
- Task: given an articulated object (e.g., a cabinet with a drawer, a laptop, scissors), synthesize a bimanual hand motion sequence that grasps the object and performs an articulation (opening, closing, rotating a part).
- Input: articulated object model (URDF with joint definitions) + target articulation parameters (e.g., "open drawer by 30 cm"); output: two-hand MANO motion trajectory performing the task.
- Key challenge: bimanual manipulation of articulated objects is significantly harder than single-hand static grasping — it requires coordinating two hands, respecting object joint constraints, and generating temporally extended motions that involve both grasping and dynamic articulation.

## 2. Core Method
- Physics-based RL framework (extends D-Grasp paradigm) in a differentiable simulator. Two separate policy networks (one per hand) with shared features for coordination.
- Curriculum learning: training progresses through stages — (1) static bimanual grasp generation, (2) single-hand articulation with the other hand stabilizing, (3) full bimanual grasping and articulation, and (4) complex multi-step articulation sequences.
- Reward design: composite reward including finger-to-object contact reward, object articulation progress reward, coordination reward (hands should not collide or work against each other), and natural motion reward (smoothness, joint limits).
- Articulated object URDF support (simulator natively supports joint constraints and kinematic chains).
- Key innovation: first method to handle full bimanual grasping + articulation of articulated objects with physics-based training, using curriculum learning to overcome the difficulty of the joint search space.

## 3. Knowledge, Supervision, and Assumptions
- Training data: none (pure simulation-based RL); objects sourced from PartNet-Mobility or similar articulated object datasets.
- Supervision: reward signal from simulation (articulation progress, contact stability, coordination).
- Domain knowledge: articulated object models (URDF), bimanual coordination priors (e.g., hands typically operate on opposite sides of a drawer).
- Assumption: object articulation parameters and model are known; rigid object parts connected by known joints.

## 4. Experiments and Findings
- Environments: PartNet-Mobility objects (drawers, doors, laptops, scissors, etc.) in Isaac Gym.
- Metrics: articulation success rate (does the object part reach the target configuration?), grasp stability (object not dropped), motion naturalness (user study), completion time.
- Main findings: ArtiGrasp successfully synthesizes bimanual articulation motions across a wide range of object categories; curriculum learning is critical — training without it fails to converge on all but the simplest tasks; the learned policies exhibit emergent coordination behaviors (one hand stabilizes while the other articulates).

## 5. Strengths and Limitations
### Strengths
- First comprehensive bimanual grasping + articulation synthesis with physics guarantees.
- Curriculum learning effectively breaks down a very hard problem into manageable stages.

### Limitations
- Extremely computationally intensive (bimanual RL on articulated objects).
- Requires full URDF models of articulated objects, which may not be available for arbitrary real objects.
- Only handles pre-specified articulation tasks, not open-ended manipulation instructions.

## 6. Takeaway
ArtiGrasp demonstrates that bimanual dexterous manipulation of articulated objects — a task long considered extremely challenging for generative methods — is feasible with physics-based RL when combined with carefully designed curriculum learning. This work establishes a foundation for physics-grounded synthesis of complex, multi-stage hand-object interactions.
