# HERMES: Human-to-Robot Embodied Learning from Multi-Source Motion Data for Mobile Dexterous Manipulation

## Summary
HERMES leverages human motion data from multi-source inputs to impart robots with versatile manipulation skills, addressing the challenge of translating multi-source human hand motions into feasible robot behaviors for robots with multi-fingered dexterous hands characterized by complex, high-dimensional action spaces, especially for mobile dexterous manipulation.

## 1. Problem and Setting
- Translating multi-source human hand motions into feasible robot behaviors is challenging for dexterous robots.
- Mobile dexterous manipulation adds the complication of locomotion.
- Input: multi-source human motion data; robot data.
- Output: a mobile dexterous manipulation policy.
- Dexterous motion retargeting prior: multi-source human motion data provides the FM prior for dexterous manipulation.

## 2. Core Method
- Human-to-Robot Embodied Learning from Multi-Source Motion Data.
- Translates human hand motions from various sources into feasible robot behaviors.
- Handles the high-dimensional action space of multi-fingered dexterous hands.
- Addresses mobile dexterous manipulation (manipulation + locomotion).
- How FM prior is injected: multi-source human motion data serves as the FM prior for dexterous manipulation knowledge.

## 3. Knowledge, Supervision, and Assumptions
- Training data: multi-source human motion data; robot data.
- Supervision: human motion supervision; robot action supervision; embodiment gap handling.
- Foundation models: pretrained motion or video models.
- Domain knowledge: dexterous manipulation, mobile manipulation, embodiment gap.
- Assumption: multi-source human motion can be translated to feasible robot behaviors.

## 4. Experiments and Findings
- Datasets: multi-source human motion datasets; mobile dexterous robot benchmarks.
- Metrics: mobile dexterous task success, generalization across sources.
- Successfully transfers multi-source human motion to mobile dexterous robots.
- The multi-source approach enables diverse skill acquisition.

## 5. Strengths and Limitations
### Strengths
- Multi-source motion data integration.
- Mobile dexterous manipulation.
- Effective embodiment gap handling.

### Limitations
- Requires diverse human motion data.
- Complex multi-source integration.
- May not generalize to all robot configurations.
- Computational cost.

## 6. Takeaway
HERMES demonstrates that multi-source human motion data can be effectively translated to mobile dexterous manipulation, addressing the embodiment gap for high-dimensional dexterous action spaces. The work exemplifies the "dexterous motion retargeting" paradigm with multi-source motion data.
