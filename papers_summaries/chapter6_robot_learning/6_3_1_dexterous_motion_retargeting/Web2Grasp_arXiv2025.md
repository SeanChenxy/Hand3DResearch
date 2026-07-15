# Web2Grasp: Learning Functional Grasps from Web Images of Hand-Object Interactions

## Summary
Web2Grasp proposes extracting human grasp information from web images of hand-object interactions to enable functional grasp synthesis for dexterous multi-finger robot hands, avoiding the need for power grasping focus or costly teleoperated robot demonstrations, by learning functional grasps from the rich HOI information in web images.

## 1. Problem and Setting
- Functional grasp synthesis for dexterous multi-finger robot hands is important but challenging.
- Most prior work focuses on power grasping or relies on costly teleoperated robot demonstrations.
- Input: web images of hand-object interactions.
- Output: a functional grasp synthesis model for dexterous robot hands.
- Dexterous motion retargeting prior: human grasp information from web images serves as the FM prior for functional grasping.

## 2. Core Method
- Extracts human grasp information from web images of hand-object interactions.
- Learns functional grasps from this rich HOI information.
- Enables dexterous multi-finger robot hand grasping without costly teleoperation.
- How FM prior is injected: web image HOI serves as the FM prior for functional grasping.

## 3. Knowledge, Supervision, and Assumptions
- Training data: web images of hand-object interactions; possibly robot grasp data.
- Supervision: human grasp supervision; functional grasp supervision.
- Foundation models: pretrained image understanding or HOI models.
- Domain knowledge: functional grasping, HOI understanding, web-scale data.
- Assumption: human grasp information from web images transfers to functional robotic grasping.

## 4. Experiments and Findings
- Datasets: web HOI image datasets; dexterous robot grasp benchmarks.
- Metrics: functional grasp success rate, generalization.
- Successfully learns functional grasps from web images.
- Avoids the need for teleoperated robot demonstrations.

## 5. Strengths and Limitations
### Strengths
- Leverages web-scale HOI images.
- Avoids costly teleoperation.
- Functional grasping focus.

### Limitations
- Web image quality varies.
- May not capture all functional grasps.
- Embodiment gap may limit transfer.

## 6. Takeaway
Web2Grasp demonstrates that functional grasps for dexterous robot hands can be learned from web images of hand-object interactions, avoiding costly teleoperation. The work exemplifies the "dexterous motion retargeting" paradigm with web-scale data as the source.
