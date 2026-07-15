# Hand-held Object Reconstruction from RGB Video with Dynamic Interaction (MCC-HO + RAR)

## Summary
A scalable paradigm for hand-held object reconstruction from Internet videos that combines a single-frame joint reconstruction model (MCC-Hand-Object) with retrieval-augmented reconstruction (RAR) using a text-to-3D generative model prompted by GPT-4(V) to retrieve 3D object models that match the in-video objects, achieving state-of-the-art performance on lab and Internet image/video datasets.

## 1. Problem and Setting
- Reconstruction of hand-held object geometry (manipulanda) from monocular Internet RGB videos over time.
- Input: monocular RGB video of hand manipulating an object; 3D hand estimates per frame.
- Output: 3D object geometry aligned with hands and images across the video.
- Task: hand-held object reconstruction with shape retrieval. Uses two strong anchors: estimated 3D hands disambiguate location/scale, and the manipulanda set is small relative to all possible objects.

## 2. Core Method
- Two-stage approach:
  1. MCC-Hand-Object (MCC-HO): a single-frame joint reconstruction model that takes a single RGB image and inferred 3D hand as input, and produces hand and object geometry.
  2. Retrieval-Augmented Reconstruction (RAR): prompts a text-to-3D generative model using GPT-4(V) to retrieve a 3D object model that matches the object in the image(s); this retrieved model is then rigidly aligned with both the input images and 3D MCC-HO observations in a temporally consistent manner.
- How FM priors are injected: GPT-4(V) provides language-grounded retrieval; text-to-3D generative models (e.g., from large 3D object datasets) provide the 3D shape prior; hand pose estimators (likely FM-based) provide 3D hand initialization.

## 3. Knowledge, Supervision, and Assumptions
- Foundation models: GPT-4(V) for retrieval prompting; text-to-3D generative model for shape synthesis; FM-based 3D hand pose estimators.
- Domain knowledge: hand-object spatial constraints; assumption that the manipulanda set is small enough to be covered by text-to-3D models.
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
This work presents a scalable paradigm for hand-held object reconstruction that capitalizes on the small size of the manipulanda category: by using foundation models (GPT-4V, text-to-3D) to retrieve a 3D shape, then aligning it with per-frame 3D hand estimates, the system achieves temporally consistent reconstruction on Internet videos — a setting where per-video optimization methods typically fail. This work exemplifies how combining complementary FM capabilities (language, 3D generation, 3D hand estimation) can solve a complex HOI problem.
