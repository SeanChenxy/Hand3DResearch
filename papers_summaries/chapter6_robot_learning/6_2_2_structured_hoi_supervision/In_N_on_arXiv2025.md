# In-N-On: Scaling Egocentric Manipulation with In-the-Wild and On-Task Data

## Summary
In-N-On scales egocentric manipulation learning by combining in-the-wild and on-task data, leveraging both diverse egocentric human videos and robot-specific data for effective robot manipulation policy training, addressing the data scalability challenge in egocentric manipulation learning.

## 1. Problem and Setting
- Egocentric manipulation learning requires large-scale data, but robot data is limited.
- Both in-the-wild human egocentric videos and on-task robot data provide complementary information.
- Input: in-the-wild egocentric human videos + on-task robot data.
- Output: a scaled egocentric manipulation policy.
- Structured HOI supervision prior: in-the-wild human videos provide structured HOI supervision for egocentric manipulation.

## 2. Core Method
- Combines in-the-wild and on-task data for egocentric manipulation learning.
- The in-the-wild data provides diversity and structured HOI supervision.
- The on-task data provides task-specific grounding.
- The model learns from both sources jointly.
- How FM prior is injected: in-the-wild video pretraining provides the FM prior; structured HOI supervision from human video transfers to robot.

## 3. Knowledge, Supervision, and Assumptions
- Training data: in-the-wild egocentric human videos; on-task robot data.
- Supervision: structured HOI supervision; on-task robot action supervision.
- Foundation models: pretrained video models for in-the-wild data understanding.
- Domain knowledge: egocentric vision, hand-object interaction, robot learning.
- Assumption: in-the-wild and on-task data are complementary for egocentric manipulation.

## 4. Experiments and Findings
- Datasets: in-the-wild egocentric video datasets; on-task robot data.
- Metrics: egocentric manipulation task success rate, generalization.
- Effectively scales egocentric manipulation learning.
- The combination of in-the-wild and on-task data is critical.

## 5. Strengths and Limitations
### Strengths
- Leverages both in-the-wild and on-task data.
- Structured HOI supervision from human video.
- Effective for egocentric manipulation.

### Limitations
- May require careful data balancing.
- May not handle very novel tasks.
- Computational cost of combining data.

## 6. Takeaway
In-N-On demonstrates that combining in-the-wild and on-task data scales egocentric manipulation learning, with structured HOI supervision from human video transferring to robot control. The work exemplifies the "structured HOI supervision" paradigm with the "In-the-wild + On-task" data combination.
