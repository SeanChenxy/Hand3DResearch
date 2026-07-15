# SAM 2: Segment Anything in Images and Videos

# Summary Template

## Summary
SAM 2 introduces a unified foundation model for promptable visual segmentation in both images and videos, using a transformer architecture with streaming memory that enables real-time video processing and interactive refinement through user prompts.

## 1. Problem and Setting
- **Task**: Promptable Visual Segmentation (PVS) - a generalized segmentation task that works for both images and videos, where users can provide prompts (points, boxes, or masks) on any frame to define and refine object segments over time
- **Inputs**: Video sequences or single images with user prompts (positive/negative clicks, bounding boxes, or masks) on any frame
- **Outputs**: Spatio-temporal masks ("masklets") that track the object of interest across video frames, or single-frame masks for images
- **Difficulty**: Video segmentation presents unique challenges including appearance changes due to motion/deformation/occlusion, lower video quality compared to images, camera motion blur, lower resolution, and the computational challenge of efficiently processing many frames while maintaining temporal consistency

## 2. Core Method
SAM 2 uses a streaming transformer architecture that processes video frames sequentially:

**Input Processing**: Frames are processed one at a time in streaming fashion. User prompts (points, boxes, masks) can be provided on any frame.

**Representation**: The model uses a transformer-based encoder-decoder architecture equipped with a streaming memory mechanism. The memory stores information about the target object and previous user interactions.

**Inference Method**: 
- For each frame, the encoder extracts features
- A memory attention module allows the model to attend to previous memories of the target object
- The decoder produces segmentation masks for the current frame
- When applied to images (single-frame videos), the memory is empty and the model behaves similarly to the original SAM

**Final Output**: Segmentation masks that form a complete "masklet" (spatio-temporal mask sequence) across the video

**Key Innovation**: The streaming memory architecture is the critical innovation - it's a natural generalization of SAM to videos that enables:
1. Efficient single-frame-at-a-time processing for real-time performance
2. Temporal consistency through memory of previous frames and interactions
3. Interactive correction capabilities where users can refine mistakes by providing additional prompts on any frame
4. The model can "remember" the object context from previously observed frames to handle occlusions and re-appearances

## 3. Knowledge, Supervision, and Assumptions
- **Training Data**: SA-V dataset with 35.5M masks across 50.9K videos - 53× more masks than any existing video segmentation dataset
- **Data Engine**: A model-in-the-loop annotation system where SAM 2 assists annotators interactively, making the process 8.4× faster at comparable quality
- **Supervision**: The model learns from manual annotations corrected by annotators, covering not just whole objects but also parts and subparts with valid boundaries
- **Pretrained Models**: Builds upon concepts from the original SAM but extends it with novel streaming memory architecture for video
- **Assumptions**: The model assumes that objects have valid boundaries and that user prompts can be provided interactively when needed for correction
- **Learned vs Provided**: The model learns to segment and track objects across diverse video distributions, while prompts are provided by users interactively

## 4. Experiments and Findings
- **Video Segmentation Datasets**: Evaluated on established VOS benchmarks including DAVIS, YouTube-VOS, and others across 17 zero-shot video segmentation benchmarks
- **Image Segmentation Datasets**: 37 zero-shot single-image segmentation benchmarks
- **Key Metrics**: Segmentation accuracy (J&F scores), number of user interactions required, inference speed
- **Important Quantitative Results**:
  - 3× fewer user interactions than prior approaches for comparable or better accuracy
  - 6× faster than SAM for image segmentation while being more accurate
  - Strong performance across multiple VOS benchmarks under different evaluation settings
  - Fairness evaluation showed minimal performance discrepancy based on perceived gender and little variance across age groups

## 5. Strengths and Limitations
### Strengths
- Truly unifies image and video segmentation in a single model
- Real-time streaming architecture enables practical video processing
- Interactive refinement allows users to easily correct mistakes
- Handles challenging cases like occlusions, small objects, and object parts
- Massive and diverse training dataset (SA-V) with broad geographical coverage
- Strong zero-shot generalization across 17 video and 37 image segmentation benchmarks
- Significant efficiency improvements (6× faster than SAM, 3× fewer user interactions)
- Open-source release with permissive licenses

### Limitations
- Still requires user interaction for challenging cases (though much less than prior methods)
- Video quality issues (motion blur, lower resolution) can affect performance
- The paper doesn't deeply address computational requirements for very long videos
- Performance on extremely rare or novel object categories may still be limited
- The streaming memory architecture may have limitations for very long-term temporal dependencies

## 6. Takeaway
SAM 2 represents a significant milestone in visual segmentation by successfully unifying image and video segmentation in a single foundation model. The key innovation is the streaming memory architecture that enables real-time, frame-by-frame processing while maintaining temporal consistency and allowing interactive refinement. Combined with the massive SA-V dataset (35.5M masks), SAM 2 achieves strong performance across diverse distributions with dramatically improved efficiency (6× faster, 3× fewer interactions). The work demonstrates that a unified model with appropriate data can effectively handle the additional complexity of video segmentation while maintaining strong image segmentation capabilities.
