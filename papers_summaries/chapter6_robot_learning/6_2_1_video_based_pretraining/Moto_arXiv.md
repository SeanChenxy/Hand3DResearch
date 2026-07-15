# Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos

## Summary
Moto introduces a latent motion token as the bridging language for learning robot manipulation from videos, addressing the challenge of transferring motion knowledge from video to robot policies by using a shared discrete token representation that bridges the video and action spaces.

## 1. Problem and Setting
- Learning robot manipulation from videos requires bridging the gap between visual motion in video and robot action spaces.
- Input: video demonstrations of manipulation; robot trajectory data (optional).
- Output: a robot manipulation policy that uses latent motion tokens.
- Video-based pretraining prior: latent motion tokens learned from video provide the FM prior for robot manipulation.

## 2. Core Method
- A latent motion token representation bridges the video and action spaces.
- The latent motion tokens are learned from video, capturing manipulation-relevant motion.
- The robot policy uses these tokens as an intermediate representation, with a separate action head generating robot actions.
- The discrete token space enables unified video-action learning.
- How FM prior is injected: video data provides motion knowledge that is encoded into the latent motion tokens; the tokens serve as the FM prior for the policy.

## 3. Knowledge, Supervision, and Assumptions
- Training data: video demonstrations; possibly robot trajectory data.
- Supervision: latent motion token prediction; robot action supervision.
- Foundation models: pretrained video or motion models.
- Domain knowledge: latent representation learning, video-to-action transfer, manipulation motion.
- Assumption: a shared discrete token space can bridge video and robot action.

## 4. Experiments and Findings
- Datasets: video demonstration datasets; robot manipulation benchmarks.
- Metrics: task success rate, transfer effectiveness.
- The latent motion token bridges video and action effectively.
- The unified token space enables scalable learning.

## 5. Strengths and Limitations
### Strengths
- Unified token space for video and action.
- Bridges the video-action gap.
- Scalable learning from video.

### Limitations
- Requires video demonstration data.
- Token space design is critical.
- May not handle all manipulation tasks.

## 6. Takeaway
Moto demonstrates that latent motion tokens provide an effective bridging language for learning robot manipulation from videos. The work exemplifies the "video-based pretraining" paradigm with a discrete token-based approach to video-action transfer.
