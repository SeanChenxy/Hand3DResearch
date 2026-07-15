# Reconstructing Hand-Held Objects in 3D from Images and Videos (Cross-reference)

## Summary
This entry is a cross-reference to the detailed summary in Chapter 3 (3D Geometry Priors, section 3.2 Shape Completion). The work presents a scalable paradigm for hand-held object reconstruction from monocular RGB video that combines a single-frame joint reconstruction model (MCC-Hand-Object) with retrieval-augmented reconstruction (RAR) using a text-to-3D generative model prompted by GPT-4(V).

## 1. Problem and Setting
- Reconstruction of hand-held object geometry (manipulanda) from monocular Internet RGB videos.
- Input: monocular RGB video of hand manipulating an object; 3D hand estimates per frame.
- Output: 3D object geometry aligned with hands and images across the video.
- Visual grounding prior: the retrieval component uses vision foundation models to match observed object appearance to 3D database models, providing a visual-grounded prior for the object shape.

## 2. Core Method
- MCC-Hand-Object (MCC-HO): a single-frame joint reconstruction model.
- Retrieval-Augmented Reconstruction (RAR): uses GPT-4(V) to prompt a text-to-3D generative model to retrieve a 3D object model; rigidly aligned with input images and 3D MCC-HO observations in a temporally consistent manner.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models: GPT-4(V) for retrieval prompting; text-to-3D generative model for shape synthesis; FM-based 3D hand pose estimators.
- Domain knowledge: hand-object spatial constraints; manipulanda set is small relative to all possible objects.

## 4. Experiments and Findings
- Datasets: lab HOI datasets and Internet image/video datasets.
- State-of-the-art performance on both lab and Internet data.

## 5. Strengths and Limitations
### Strengths
- Scalable paradigm leveraging FMs and 3D object datasets.
- Unified object geometry via RAR provides strong temporal consistency.
- Works on in-the-wild Internet videos.

### Limitations
- Depends on text-to-3D models covering manipulanda.
- Retrieval quality depends on LLM/VLM.
- Per-frame MCC-HO may struggle with severe occlusion.

## 6. Takeaway
This work capitalizes on the small size of the manipulanda category by using foundation models to retrieve a 3D shape, then aligning it with per-frame hand estimates. In the context of visual grounding (chapter 4), the FM-based retrieval provides a visual-grounded shape prior. See chapter 3 section 3.2 for the full technical details.
