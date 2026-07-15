# You Only Teach Once: Learn One-Shot Bimanual Robotic Manipulation from Video Demonstrations (YOTO)

## Summary
YOTO (You Only Teach Once) learns bimanual robotic manipulation from video demonstrations in a one-shot manner, addressing the inefficiency of multiple demonstrations by teaching the robot only once via a video demonstration, with the policy generalizing to variations of the demonstrated task.

## 1. Problem and Setting
- Bimanual robotic manipulation typically requires multiple demonstrations for training.
- One-shot learning from a single video demonstration would be more efficient.
- Input: a single video demonstration of the bimanual task.
- Output: a bimanual manipulation policy that generalizes from the single demonstration.
- Dexterous motion retargeting prior: a single video demonstration provides the FM prior for one-shot bimanual learning.

## 2. Core Method
- YOTO learns bimanual manipulation from a single video demonstration in a one-shot manner.
- The policy generalizes to variations of the demonstrated task (e.g., different object instances, slight position changes).
- How FM prior is injected: the single video demonstration serves as the FM prior; the policy learns to extract and generalize from it.

## 3. Knowledge, Supervision, and Assumptions
- Training data: a single video demonstration per task.
- Supervision: imitation learning; task-level alignment.
- Foundation models: pretrained video understanding or world models.
- Domain knowledge: one-shot imitation learning, bimanual manipulation, video understanding.
- Assumption: a single video demonstration contains enough information for generalization.

## 4. Experiments and Findings
- Datasets: bimanual manipulation tasks with video demonstrations.
- Metrics: one-shot task success rate, generalization.
- Successfully learns bimanual manipulation from a single video.
- The one-shot paradigm is efficient.

## 5. Strengths and Limitations
### Strengths
- One-shot learning efficiency.
- Bimanual manipulation focus.
- Generalization from a single demonstration.

### Limitations
- May not handle very novel task variations.
- Quality of the single demonstration is critical.
- May not generalize to very different bimanual tasks.

## 6. Takeaway
YOTO demonstrates that bimanual manipulation can be learned from a single video demonstration, with the policy generalizing to task variations. The work exemplifies the "dexterous motion retargeting" paradigm with the one-shot learning approach.
