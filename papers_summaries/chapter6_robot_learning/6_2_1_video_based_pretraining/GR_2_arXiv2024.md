# GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation

## Summary
GR-2 is a generative video-language-action model that leverages web-scale knowledge for robot manipulation, pre-trained on a large corpus of internet videos (especially human HOI videos) to learn generalizable manipulation knowledge, then fine-tuned on a small amount of robot data to produce a powerful VLA model for robot manipulation.

## 1. Problem and Setting
- Robot manipulation policies require large amounts of robot data, but such data is scarce and expensive to collect.
- Input: internet video (especially human HOI) + small amount of robot trajectory data.
- Output: a video-language-action model for robot manipulation that generates video predictions and actions.
- Video-based pretraining prior: internet videos (especially human HOI) provide web-scale knowledge for manipulation.

## 2. Core Method
- A video-language-action model that generates video predictions and actions.
- Pretrained on a large corpus of internet videos, including extensive human HOI demonstrations, to learn generalizable manipulation knowledge.
- Fine-tuned on a small amount of robot trajectory data to specialize for the target robot.
- The model generates future video frames and robot actions conditioned on the input observation and language instruction.
- How FM prior is injected: web-scale video pretraining provides the foundational manipulation knowledge; the model learns to generate actions as a natural extension of video prediction.

## 3. Knowledge, Supervision, and Assumptions
- Training data: large-scale internet video corpus (including human HOI); small robot trajectory dataset for fine-tuning.
- Supervision: video prediction loss (pretraining), action prediction loss (fine-tuning).
- Foundation model: web-scale video-language pretraining.
- Domain knowledge: human-object interaction, video generation, VLA modeling.
- Assumption: internet video manipulation knowledge transfers to robot manipulation.

## 4. Experiments and Findings
- Datasets: internet video corpus; robot manipulation benchmarks.
- Metrics: video generation quality, robot task success rate, generalization to novel objects.
- The web-scale pretraining significantly improves manipulation performance.
- Generalizes to novel objects and tasks.

## 5. Strengths and Limitations
### Strengths
- Leverages web-scale data for robot learning.
- Generative formulation enables both video and action prediction.
- Strong generalization to novel tasks.

### Limitations
- Requires large-scale video data for pretraining.
- May have sim-to-real gap.
- Quality of internet data affects pretraining.
- Computational cost of large-scale pretraining.

## 6. Takeaway
GR-2 demonstrates that generative video-language-action pretraining on web-scale internet videos enables effective robot manipulation with limited robot data. The work exemplifies the "video-based pretraining" paradigm where internet video serves as a scalable source of manipulation knowledge.
