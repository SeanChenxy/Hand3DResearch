# Masquerade: Learning from In-the-Wild Human Videos Using Data-Editing

## Summary
Masquerade edits in-the-wild egocentric human videos to bridge the visual embodiment gap between humans and robots, then learns a robot policy from the edited videos, addressing the data scarcity problem in robot manipulation by leveraging abundant human video data through data editing.

## 1. Problem and Setting
- Robot manipulation datasets are orders of magnitude smaller than language and vision datasets.
- In-the-wild egocentric human videos are abundant but have a visual embodiment gap with robots.
- Input: in-the-wild egocentric human videos.
- Output: a robot manipulation policy learned from the edited (human-to-robot) videos.
- Dexterous motion retargeting prior: data editing retargets human videos to robot embodiment.

## 2. Core Method
- Masquerade edits in-the-wild egocentric human videos to bridge the visual embodiment gap between humans and robots.
- The edited videos appear as if they were robot demonstrations.
- A robot policy is learned from the edited videos.
- How FM prior is injected: data editing (potentially using generative models) is the key retargeting step that enables human video to serve as robot training data.

## 3. Knowledge, Supervision, and Assumptions
- Training data: in-the-wild egocentric human videos; possibly robot data.
- Supervision: robot action supervision; data editing supervision.
- Foundation models: data editing models (likely generative); pretrained video models.
- Domain knowledge: data editing, embodiment gap, human-to-robot transfer.
- Assumption: data editing can effectively bridge the visual embodiment gap.

## 4. Experiments and Findings
- Datasets: in-the-wild egocentric video datasets; robot manipulation benchmarks.
- Metrics: task success rate, visual embodiment gap.
- Successfully learns robot policies from edited human videos.
- The data editing is the key innovation.

## 5. Strengths and Limitations
### Strengths
- Leverages abundant in-the-wild human videos.
- Data editing approach is flexible.
- Bridges visual embodiment gap.

### Limitations
- Data editing quality is critical.
- May require careful prompt engineering.
- Sim-to-real gap may persist.
- May not handle all task types.

## 6. Takeaway
Masquerade demonstrates that in-the-wild human videos can be transformed into robot training data via data editing, addressing the data scarcity problem in robot manipulation. The work exemplifies the "dexterous motion retargeting" paradigm with a data editing approach.
