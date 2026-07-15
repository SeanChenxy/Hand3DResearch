# Hand-held Object Reconstruction from RGB Video with Dynamic Interaction (Cross-reference)

## Summary
This entry is a cross-reference to the detailed summary in Chapter 3 (3D Geometry Priors, section 3.2 Shape Completion). The work presents a scalable paradigm for hand-held object reconstruction from monocular RGB video that combines a single-frame joint reconstruction model (MCC-Hand-Object) with retrieval-augmented reconstruction (RAR) using a text-to-3D generative model prompted by GPT-4(V), achieving state-of-the-art performance on lab and Internet video datasets.

## 1. Problem and Setting
- Reconstruction of hand-held object geometry (manipulanda) from monocular Internet RGB videos.
- Input: monocular RGB video of hand manipulating an object; 3D hand estimates per frame.
- Output: 3D object geometry aligned with hands and images across the video.
- Visual grounding prior: the retrieval component uses vision foundation models to match observed object appearance to 3D database models, providing a visual-grounded prior for the object shape.

## 2. Core Method
- MCC-Hand-Object (MCC-HO): a single-frame joint reconstruction model taking a single RGB image and inferred 3D hand as input, producing hand and object geometry.
- Retrieval-Augmented Reconstruction (RAR): uses GPT-4(V) to prompt a text-to-3D generative model to retrieve a 3D object model that matches the object in the image(s); the retrieved model is then rigidly aligned with the input images and 3D MCC-HO observations in a temporally consistent manner.
- How visual grounding prior is injected: vision foundation models ground the object appearance in 3D shape database entries, providing a complete shape prior for observed manipulanda.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models: GPT-4(V) for retrieval prompting; text-to-3D generative model for shape synthesis; FM-based 3D hand pose estimators.
- Domain knowledge: hand-object spatial constraints; the manipulanda set is small relative to all possible objects.
- Training data: pretrains the 3D hand estimators; RAR is zero-shot at inference.
- Assumption: text-to-3D models can generate recognizable manipulanda; the retrieved shape matches the in-video object.

## 4. Experiments and Findings
- Datasets: lab HOI datasets and Internet image/video datasets.
- Metrics: 3D object shape accuracy (Chamfer, F-score), temporal consistency, hand-object alignment.
- Achieves state-of-the-art performance on both lab and Internet image/video datasets.
- The combination of MCC-HO (per-frame joint reconstruction) and RAR (retrieval-based shape) outperforms either alone.

## 5. Strengths and Limitations
### Strengths
- Scalable paradigm leveraging large language/vision models and 3D object datasets.
- Unified object geometry across all frames via RAR provides strong temporal consistency.
- Works on in-the-wild Internet videos where prior methods fail.
- Modular: the per-frame model and the retrieval module can be independently improved.

### Limitations
- Depends on text-to-3D models covering the manipulanda category.
- Retrieval quality depends on the LLM/VLM correctly identifying the object.
- Per-frame MCC-HO may still struggle with severe occlusion.
- Temporal alignment of retrieved shape to video may introduce jitter.

## 6. Takeaway
This work presents a scalable paradigm for hand-held object reconstruction that capitalizes on the small size of the manipulanda category: by using foundation models (GPT-4V, text-to-3D) to retrieve a 3D shape, then aligning it with per-frame 3D hand estimates, the system achieves temporally consistent reconstruction on Internet videos. In the context of semantic/visual-grounding priors (chapter 4), this work exemplifies the visual grounding paradigm where object appearance is grounded to 3D shape via foundation model retrieval. See chapter 3 section 3.2 for the full technical details.
