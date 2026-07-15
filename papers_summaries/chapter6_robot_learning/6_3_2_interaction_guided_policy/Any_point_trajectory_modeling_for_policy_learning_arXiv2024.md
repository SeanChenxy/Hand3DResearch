# Any-Point Trajectory Modeling for Policy Learning (ATM)

## Summary
ATM (Any-Point Trajectory Modeling) learns policies by modeling trajectories of arbitrary points in the scene, enabling learning from video demonstrations and providing a flexible policy representation for robotic manipulation, addressing limitations of policies that operate on specific representations (e.g., end-effector, image features).

## 1. Problem and Setting
- Robot policies typically operate on specific representations (e.g., end-effector pose, image features), limiting their flexibility.
- Input: video demonstration of the task.
- Output: a policy that predicts trajectories of any point in the scene.
- Interaction-guided policy prior: any-point trajectory modeling enables flexible policy learning from video.

## 2. Core Method
- ATM models trajectories of arbitrary points in the scene, not just specific end-effector or feature points.
- Learns from video demonstrations: point trajectories can be extracted from video via tracking.
- The policy can be applied to any point in the scene, providing flexibility.
- How FM prior is injected: pretrained video or tracking models (likely point tracking models) provide the FM prior for trajectory extraction.

## 3. Knowledge, Supervision, and Assumptions
- Training data: video demonstrations; possibly robot trajectory data.
- Supervision: trajectory prediction; robot action supervision.
- Foundation models: pretrained video or point tracking models.
- Domain knowledge: point tracking, trajectory modeling, robot policy learning.
- Assumption: any-point trajectories provide a useful policy representation.

## 4. Experiments and Findings
- Datasets: video demonstration datasets; robot manipulation benchmarks.
- Metrics: policy success rate, generalization.
- ATM provides a flexible policy representation.
- Trajectory modeling from video is effective.

## 5. Strengths and Limitations
### Strengths
- Flexible any-point trajectory representation.
- Learns from video demonstrations.
- Generalizes across tasks.

### Limitations
- Requires accurate point tracking.
- May not handle all manipulation tasks.
- Computational cost of trajectory modeling.

## 6. Takeaway
ATM demonstrates that any-point trajectory modeling provides a flexible and effective policy representation for robotic manipulation, learning from video demonstrations. The work exemplifies the "interaction-guided policy" paradigm with a flexible trajectory-based approach.
