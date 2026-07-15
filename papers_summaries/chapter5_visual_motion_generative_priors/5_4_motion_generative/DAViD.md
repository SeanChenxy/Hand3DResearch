# DAViD: Modeling Dynamic Affordance of 3D Objects Using Pre-trained Video Diffusion Models

## Summary
DAViD presents a novel framework for learning Dynamic Affordance across various target object categories, addressing the scarcity of 4D HOI datasets by learning the 3D dynamic affordance from synthetically generated 4D HOI samples, with a pipeline that first generates 2D HOI videos from a pre-trained video diffusion model and then lifts them to 3D dynamic affordance.

## 1. Problem and Setting
- Modeling how humans interact with objects dynamically (movement of humans and objects over time) is crucial for AI assistants and human behavior mimicry.
- Input: 3D object category (target for which to learn affordance).
- Output: 3D dynamic affordance representation showing how humans interact with the object over time.
- Motion-generative prior: pre-trained video diffusion models for generating 2D HOI videos; 3D dynamic affordance is then learned from these.

## 2. Core Method
- A pipeline that first generates 2D HOI videos for various object categories using a pre-trained video diffusion model.
- The generated 2D videos are then lifted to 3D dynamic affordance via 3D reconstruction or similar techniques.
- To address the scarcity of 4D HOI datasets, the method learns the 3D dynamic affordance from synthetically generated 4D HOI samples.
- How FM prior is injected: a pre-trained video diffusion model provides the 2D HOI video generation; the 3D dynamic affordance is derived from these videos.

## 3. Knowledge, Supervision, and Assumptions
- Training data: synthetic 4D HOI samples (generated via the pipeline); 3D object datasets.
- Supervision: 3D dynamic affordance supervision from the synthetic data.
- Foundation model: pre-trained video diffusion model.
- Domain knowledge: affordance reasoning, 3D reconstruction, video-to-3D lifting.
- Assumption: 2D HOI videos contain enough information to derive 3D dynamic affordance.

## 4. Experiments and Findings
- Datasets: various target object categories for affordance learning.
- Metrics: affordance accuracy, generalization across object categories.
- Learns 3D dynamic affordance from synthetic 4D HOI samples.
- Generalizes across various object categories.

## 5. Strengths and Limitations
### Strengths
- Addresses 4D HOI dataset scarcity via synthetic generation.
- Leverages pre-trained video diffusion models.
- Generalizes across object categories.
- Provides dynamic (temporal) affordance, not just static.

### Limitations
- Quality depends on the pre-trained video diffusion model.
- The 2D-to-3D lifting may introduce errors.
- Synthetic data may have a sim-to-real gap.
- May not capture very complex affordances.

## 6. Takeaway
DAViD demonstrates that 3D dynamic affordance can be learned from synthetic 4D HOI samples generated via pre-trained video diffusion models, addressing the data scarcity problem and enabling cross-category generalization. The work exemplifies the "motion-generative prior" paradigm where video generation models provide data for 3D affordance learning.
