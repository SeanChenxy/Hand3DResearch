# Motus: A Unified Latent Action World Model

## Summary
Motus is a unified latent action world model that leverages existing general pretrained models and rich, sharable motion information, introducing a Mixture-of-Transformer (MoT) architecture to unify multimodal generative capabilities for embodied agents, addressing the fragmentation of understanding, world modeling, and control into separate models.

## 1. Problem and Setting
- General embodied agents must function as unified systems, but current methods build on isolated models for understanding, world modeling, and control.
- Input: multimodal observations (video, proprioception, language) + action space.
- Output: unified latent action world model that integrates understanding, world modeling, and control.
- Video-based pretraining prior: existing pretrained models (for understanding) + motion information for world modeling.

## 2. Core Method
- A Mixture-of-Transformer (MoT) architecture that unifies multimodal generative capabilities.
- Leverages existing general pretrained models (e.g., for visual understanding) and rich, sharable motion information.
- The unified model integrates understanding, world modeling, and control in a single system.
- Enables learning from large-scale, heterogeneous data.
- How FM prior is injected: existing general pretrained models provide the understanding prior; the world model component is built on rich, sharable motion information.

## 3. Knowledge, Supervision, and Assumptions
- Training data: large-scale, heterogeneous data (video, robot trajectories, language).
- Supervision: multi-task losses for understanding, world modeling, and control.
- Foundation models: existing pretrained models (likely video and language models).
- Domain knowledge: embodied agents, world models, multimodal learning, mixture of experts.
- Assumption: existing pretrained models and motion information can be unified into a single system.

## 4. Experiments and Findings
- Datasets: large-scale embodied AI benchmarks.
- Metrics: task success rate, multimodal generation quality, control performance.
- Demonstrates unified multimodal generative capabilities.
- The MoT architecture effectively integrates understanding, world modeling, and control.

## 5. Strengths and Limitations
### Strengths
- Unified architecture for understanding, world modeling, and control.
- Leverages existing pretrained models.
- Mixture-of-Transformer enables effective integration.
- Multimodal generative capabilities.

### Limitations
- Complex architecture.
- May not be optimal for all specific tasks.
- Computational cost of unified model.
- May have integration challenges between components.

## 6. Takeaway
Motus demonstrates that a unified latent action world model with a Mixture-of-Transformer architecture can integrate understanding, world modeling, and control, addressing the fragmentation in current embodied AI. The work exemplifies the "video-based pretraining" paradigm where existing pretrained models are leveraged for embodied agent learning.
