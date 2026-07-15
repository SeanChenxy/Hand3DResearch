# RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning

## Summary
RoboWheel is a data engine that converts human hand-object interaction (HOI) videos into training-ready supervision for cross-morphology robotic learning, performing high-precision HOI reconstruction from monocular RGB or RGB-D inputs and enforcing physical plausibility via a reinforcement learning (RL) optimizer that refines hand-object relative poses under contact and penetration constraints.

## 1. Problem and Setting
- Data scarcity fundamentally limits robot learning, especially across morphologies.
- Human HOI videos are abundant but need to be converted to robot training data.
- Input: human HOI videos (monocular RGB or RGB-D).
- Output: a robot learning data engine that produces cross-morphology supervision.
- Dexterous motion retargeting prior: HOI reconstruction serves as the basis for retargeting.

## 2. Core Method
- High-precision HOI reconstruction from monocular RGB or RGB-D inputs.
- Physical plausibility enforcement via an RL optimizer that refines hand-object relative poses under contact and penetration constraints.
- Cross-morphology robot learning from the generated supervision.
- How FM prior is injected: HOI reconstruction models (potentially with FM priors) provide the initial reconstruction; the RL optimizer refines it.

## 3. Knowledge, Supervision, and Assumptions
- Training data: human HOI videos.
- Supervision: HOI reconstruction; physical plausibility; robot action supervision.
- Foundation models: HOI reconstruction models (likely with FM priors).
- Domain knowledge: HOI reconstruction, physics-based optimization, cross-morphology learning.
- Assumption: high-precision HOI reconstruction enables cross-morphology transfer.

## 4. Experiments and Findings
- Datasets: human HOI video datasets; cross-morphology robot benchmarks.
- Metrics: cross-morphology task success, HOI reconstruction accuracy.
- The data engine produces high-quality training supervision.
- Cross-morphology learning is enabled.

## 5. Strengths and Limitations
### Strengths
- High-precision HOI reconstruction.
- Physical plausibility enforcement.
- Cross-morphology learning enabled.

### Limitations
- Requires high-quality HOI reconstruction.
- RL optimization is computationally expensive.
- May not handle very novel morphologies.

## 6. Takeaway
RoboWheel demonstrates that a data engine combining high-precision HOI reconstruction with physical plausibility enforcement can produce training-ready supervision for cross-morphology robot learning. The work exemplifies the "dexterous motion retargeting" paradigm with a data engine approach.
