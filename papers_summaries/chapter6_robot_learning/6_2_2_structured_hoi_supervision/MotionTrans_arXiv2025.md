# MotionTrans: Human VR Data Enable Motion-Level Learning for Robotic Manipulation Policies

## Summary
MotionTrans leverages human VR data to enable motion-level learning for robotic manipulation policies, addressing the challenge of acquiring motion knowledge for robot manipulation by using the rich motion information in human VR demonstrations, which offer precise and diverse motion data.

## 1. Problem and Setting
- Acquiring motion knowledge for robot manipulation is challenging, especially for fine-grained manipulation.
- Human data with rich diversity of manipulation behaviors is a valuable resource, but motion-level learning is hard.
- Input: human VR data (with precise hand-object interaction motion); robot data.
- Output: a robot manipulation policy with motion-level learning.
- Structured HOI supervision prior: human VR data provides structured HOI motion supervision.

## 2. Core Method
- MotionTrans uses human VR data to enable motion-level learning for robot manipulation policies.
- The VR data provides precise, structured HOI motion supervision.
- The motion knowledge transfers to robot manipulation policy training.
- How FM prior is injected: human VR motion data serves as the FM prior for motion-level knowledge.

## 3. Knowledge, Supervision, and Assumptions
- Training data: human VR manipulation data; robot trajectory data.
- Supervision: human motion supervision; robot action supervision; motion-level alignment.
- Foundation models: pretrained motion or video models.
- Domain knowledge: motion-level learning, VR data, robot manipulation, hand-object interaction.
- Assumption: human VR motion data transfers to robot manipulation.

## 4. Experiments and Findings
- Datasets: human VR data; robot manipulation benchmarks.
- Metrics: motion-level policy quality, task success rate.
- Enables motion-level learning for robot manipulation.
- VR data provides precise motion supervision.

## 5. Strengths and Limitations
### Strengths
- Leverages precise VR motion data.
- Motion-level learning.
- Effective for fine-grained manipulation.

### Limitations
- Requires VR data (specialized).
- Embodiment gap may limit transfer.
- May not generalize to all manipulation.

## 6. Takeaway
MotionTrans demonstrates that human VR data enables motion-level learning for robotic manipulation, with structured HOI motion supervision transferring to robot control. The work exemplifies the "structured HOI supervision" paradigm applied to motion-level learning.
