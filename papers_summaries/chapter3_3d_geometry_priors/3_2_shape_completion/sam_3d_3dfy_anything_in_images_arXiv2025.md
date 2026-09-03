# SAM 3D: 3Dfy Anything in Images

**Authors:** SAM 3D Team, Xingyu Chen*, Fu-Jen Chu*, et al. (Meta Superintelligence Labs; senior authors include Jitendra Malik)  
**Date:** 2025-11 (arXiv v1); 2026-06-02 (arXiv v2)  
**Identifier:** [arXiv:2511.16624](https://arxiv.org/abs/2511.16624)  
**Zotero item:** `JCH9QT44` ([Zotero](zotero://select/library/items/JCH9QT44))  
**Evidence status:** Zotero metadata and the full paper PDF (main sections through the experiments) were verified.

## Summary

SAM 3D is a generative model for visually grounded 3D object reconstruction that predicts an object's geometry, texture, and layout (rotation, translation, scale) from a single image of a natural scene. Unlike feed-forward reconstruction systems that require clean object-centric captures, SAM 3D is trained to operate directly on raw photographs with heavy occlusion, background clutter, and truncation, combining recognition of what is visible with hallucination of what is not. The system has a two-stage architecture: a geometry model that outputs a coarse object shape and pose, followed by a texture-and-refinement model that sharpens the geometry and synthesizes appearance as either a mesh or 3D Gaussian splats. Trained through a carefully staged data pipeline that scales from synthetic 3D assets to render-pasted composites to millions of in-the-wild images annotated via a model-in-the-loop data engine, SAM 3D achieves a human preference win rate of at least 5:1 over prior single-image 3D reconstruction and generation approaches on real-world objects, and releases a 1,000-item artist-created benchmark (SA-3DAO) together with model weights and a demo.

## Background and Problem

Single-image 3D reconstruction of objects "in the wild" is fundamentally ill-posed: the camera observes only the visible surface, so the model must both recognize the object (filling in occluded or truncated regions) and invent plausible geometry and appearance for the unseen parts. Existing approaches split into multi-view feed-forward reconstruction systems, which excel on clean, object-centric images but degrade sharply on cluttered natural photos, and generative 3D approaches trained on synthetic assets, which transfer poorly to real-world image distributions. There was a need for a generative model that is both grounded in the image evidence (segmentation-like object localization, layout estimation relative to the camera) and robust to the occlusion statistics of natural images, with evaluation that goes beyond synthetic renders to artist-quality assets created from real photographs.

## Method

The pipeline takes an input image with a box or mask specifying the object and predicts geometry, texture, and layout in two stages.

Geometry model: a 1.2B-parameter flow-matching transformer built with a Mixture-of-Transformers design. A DINOv2-style encoder processes pairs of the cropped object image and the full image together with their masks, so the model jointly sees object appearance and scene context; an optional conditioning input provides the scene point map from a metric 3D foundation model to improve layout estimation. The output is a coarse voxelized object shape (64³ occupancy grid) together with its similarity-transform layout (rotation, translation, scale) relative to the camera.

Texture and refinement model: a 600M-parameter flow transformer operating on a sparse latent representation of the coarse voxels. It refines the geometry to high resolution and synthesizes texture; shared VAE-based decoders emit either a textured mesh or a set of 3D Gaussian splats, letting the same latent serve both output formats.

Data and training proceed in three stages. Pretraining uses Iso-3DO, about 2.7M meshes curated from Objaverse-XL plus licensed data, rendered from 24 viewpoints (roughly 2.5T training tokens) to teach shape and layout from object-centric images. Mid-training uses RP-3DO, a render-and-paste pipeline generating about 61M composite samples from 2.8M unique meshes pasted into real images, which teaches mask following, occlusion robustness, and scene-relative layout. Post-training uses a model-in-the-loop (MITL) data engine: annotators cannot author meshes, but can select and align among N=8 model-generated candidates, which scales to about 1M in-the-wild images annotated with about 3.14M untextured and roughly 100K textured meshes; the hardest cases are routed to professional artists (Art-3DO). The final model is tuned with supervised fine-tuning and then DPO-based preference alignment, and distilled from 25 to 4 function evaluations to reach sub-second inference.

## Contributions

1. A two-stage generative architecture (1.2B geometry model plus 600M texture-and-refinement model) that predicts object geometry, texture, and camera-relative layout from a single in-the-wild image.
2. A staged data strategy — Iso-3DO pretraining, RP-3DO render-paste mid-training, and an MITL annotation engine producing ~1M real images plus Art-3DO artist data — that bridges synthetic 3D assets and natural-image distributions.
3. A preference-aligned, distilled model achieving sub-second inference with human-preference win rates of at least 5:1 over prior single-image 3D methods on real-world objects.
4. SA-3DAO, a benchmark of 1,000 artist-created 3D ground-truth assets (churches, ski lifts, animals, household items) built from natural images, along with released code, weights, benchmark, and an online demo.

## Experimental Setup

The model is evaluated on SA-3DAO (1K artist meshes from natural images) and on in-the-wild preference studies, against feed-forward reconstruction baselines (e.g., recent multi-view pose-and-shape systems) and image-to-3D generative baselines, using both geometric metrics (Chamfer-type accuracy/completeness against artist ground truth) and human preference rankings on real photographs. Inference uses 4 function evaluations after distillation. Human raters compare outputs across categories of varying occlusion and clutter. Released artifacts include the code, model weights, the SA-3DAO benchmark, and a hosted demo (github.com/facebookresearch/sam-3d-objects, ai.meta.com/sam3d).

## Results

- Human preference: on real-world object images, SAM 3D wins at least 5:1 against prior single-image 3D reconstruction and generation systems, with the largest margins on occluded and cluttered images that break object-centric capture assumptions.
- On SA-3DAO, the model improves over the strongest baselines on both geometric accuracy and completeness, indicating that its generative hallucination of unobserved parts does not come at the cost of grounding in the visible evidence.
- Qualitatively, the two-stage design preserves the object identity and pose visible in the image while completing occluded regions plausibly, and the mesh and Gaussian-splat outputs remain consistent because they decode from the shared latent.
- Distillation from 25 to 4 function evaluations keeps quality nearly unchanged while bringing inference below one second per object.

## Limitations

As a generative model, SAM 3D outputs are plausible hypotheses rather than measured geometry: for highly ambiguous views, the hallucinated back-side geometry and texture can be confidently wrong, and there is no guarantee of metric accuracy or physical validity (contact, support, or mass properties). The layout prediction depends on the quality of the provided point-map conditioning when used, and layout errors translate into misplaced assets for downstream scene composition. Training relies on Objaverse-derived synthetic data whose category coverage is uneven, and the MITL annotation engine, while scalable, still bounds performance by the model's own candidate pool. Very thin structures, transparent and reflective materials, and heavily truncated objects remain challenging, and the reported human-preference advantages are concentrated on object-level tasks rather than full-scene reconstruction.
