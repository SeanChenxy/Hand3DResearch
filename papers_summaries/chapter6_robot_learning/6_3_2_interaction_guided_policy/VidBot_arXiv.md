# VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation

## Summary
VidBot is a framework for learning generalizable 3D actions from in-the-wild 2D human videos, enabling zero-shot robotic manipulation by bridging the embodiment gap via learned 3D actions, leveraging vast amounts of human video data on the internet to minimize physical robot learning.

## 1. Problem and Setting
- Bridging the embodiment gap between humans and robots while minimizing physical robot learning is a key challenge.
- Human videos offer a promising source of manipulation data at scale.
- Input: in-the-wild 2D human videos.
- Output: a robot manipulation policy with generalizable 3D actions.
- Interaction-guided policy prior: learned 3D actions from human videos provide the FM prior for generalizable manipulation.

## 2. Core Method
- Learns generalizable 3D actions from in-the-wild 2D human videos.
- Bridges the embodiment gap via the learned 3D action representation.
- Enables zero-shot robotic manipulation without task-specific robot training.
- How FM prior is injected: human video 3D action understanding provides the FM prior for generalizable manipulation.

## 3. Knowledge, Supervision, and Assumptions
- Training data: in-the-wild 2D human videos; possibly some robot data.
- Supervision: 3D action learning; action transfer to robot.
- Foundation models: pretrained 3D action or video understanding models.
- Domain knowledge: 3D action learning, embodiment gap, human-to-robot transfer.
- Assumption: 3D actions learned from human video generalize to robots.

## 4. Experiments and Findings
- Datasets: in-the-wild 2D human video datasets; robot manipulation benchmarks.
- Metrics: zero-shot task success rate, generalization.
- VidBot enables zero-shot manipulation from human video.
- The 3D action learning is the key contribution.

## 5. Strengths and Limitations
### Strengths
- Leverages in-the-wild human video.
- Generalizable 3D action representation.
- Zero-shot manipulation.

### Limitations
- Requires 3D action supervision.
- Embodiment gap may still limit transfer.
- May not handle all manipulation tasks.

## 6. Takeaway
VidBot demonstrates that generalizable 3D actions learned from in-the-wild 2D human videos enable zero-shot robotic manipulation. The work exemplifies the "interaction-guided policy" paradigm with 3D action learning from human video.
