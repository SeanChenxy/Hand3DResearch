# CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos

## Summary
CLAP (Contrastive Latent Action Pretraining) is a framework that aligns the visual latent space from human videos with a proprioceptive latent space from robot trajectories, addressing the visual entanglement in existing Latent Action Models that capture noise rather than manipulation skills, enabling effective learning of Vision-Language-Action models for generalist robots from human video data.

## 1. Problem and Setting
- Generalist Vision-Language-Action (VLA) models are hindered by the scarcity of robotic data compared to the abundance of human video demonstrations.
- Existing Latent Action Models attempt to leverage video data but suffer from visual entanglement, capturing noise rather than manipulation skills.
- Input: human video demonstrations (without robot data) + robot trajectories.
- Output: a VLA policy that can be deployed on robots.
- Video-based pretraining prior: human video provides manipulation knowledge; CLAP bridges it to robot proprioceptive space.

## 2. Core Method
- Contrastive latent action pretraining: aligns the visual latent space from videos with a proprioceptive latent space from robot trajectories via contrastive learning.
- The contrastive objective ensures the latent actions capture meaningful manipulation skills rather than visual noise.
- Enables effective use of large-scale human video data for training VLA models.
- How video prior is injected: human video data is encoded into a visual latent space; contrastive alignment with robot proprioceptive space enables transfer.

## 3. Knowledge, Supervision, and Assumptions
- Training data: human video demonstrations; robot trajectory data.
- Supervision: contrastive alignment between visual and proprioceptive latents; VLA training.
- Foundation models: pretrained video encoders, pretrained VLA backbones.
- Domain knowledge: hand-object interaction, contrastive learning, robot learning.
- Assumption: human video manipulation skills can transfer to robots via the aligned latent space.

## 4. Experiments and Findings
- Datasets: human video datasets; robot manipulation benchmarks.
- Metrics: robot task success rate, manipulation skill transfer.
- Effectively learns VLA models from human video data.
- The contrastive alignment addresses the visual entanglement problem.

## 5. Strengths and Limitations
### Strengths
- Addresses the scarcity of robot data.
- Contrastive alignment ensures meaningful skill transfer.
- Leverages abundant human video data.
- Reduces visual entanglement compared to prior latent action models.

### Limitations
- Requires paired or aligned data for contrastive learning.
- Sim-to-real gap may persist.
- May not capture all manipulation skills.
- Depends on the quality of human video data.

## 6. Takeaway
CLAP demonstrates that contrastive alignment between visual and proprioceptive latent spaces enables effective VLA learning from abundant human video data, addressing both the data scarcity and visual entanglement problems. The work exemplifies the "video-based pretraining" paradigm where human video serves as the primary training source.
