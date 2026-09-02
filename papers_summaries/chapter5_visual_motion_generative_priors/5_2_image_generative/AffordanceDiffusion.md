# Affordance Diffusion: Synthesizing Hand-Object Interactions

**Authors:** Yufei Ye, Xueting Li, Abhinav Gupta, Shalini De Mello, Stan Birchfield, Jiaming Song, Shubham Tulsiani, Sifei Liu  
**Date:** 2023-06 (CVPR)  
**Identifier:** DOI `10.1109/CVPR52729.2023.02153`  
**Zotero item:** `S646QDWT` ([Zotero](zotero://select/library/items/S646QDWT))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; the summary was written without full-text extraction, and unavailable details are marked as not reported.  
## Summary
Affordance Diffusion addresses the difficulty of synthesizing plausible hand-object interactions for objects that may not appear in the training data. Given a single RGB image of an object, it separates generation into sampling an interaction layout and rendering image content with a pretrained diffusion prior. The layout stage represents hand articulation and contact without committing to final image appearance, while the content stage generates the interacting hand and object. The paper reports improved generalization to novel objects and out-of-distribution in-the-wild scenes, although the available evidence does not provide representative numerical results.

## Background and Problem
Hand-object image synthesis must coordinate articulated hand geometry, contact, occlusion, and object appearance. Text or image generation alone does not explicitly specify where the hand should approach an object or how the interaction should be articulated. The task takes a single RGB image of a target object as input and produces a plausible RGB image showing a hand interacting with that object, together with descriptive affordance information such as articulation and approaching orientation.

## Method
The framework uses two generative stages. LayoutNet samples an articulation-agnostic hand-object interaction layout that describes the interaction arrangement and contact region. ContentNet then conditions image synthesis on this layout to generate the hand-object image. Both stages are built on a large pretrained diffusion model, using its learned image representation while adding interaction-specific control.

## Contributions
- A two-stage decomposition that separates interaction-layout sampling from image-content synthesis.
- An affordance representation that exposes hand articulation and approaching orientation in addition to the rendered image.
- A diffusion-prior formulation intended to generalize hand-object synthesis beyond objects seen during training.

## Experimental Setup
The reported supervision uses paired hand-object imagery and hand-pose information. Exact dataset names, split sizes, training configuration, baselines, and metric definitions are not reported in the paper evidence available for this rewrite. The evaluation considers novel object categories and out-of-distribution in-the-wild scenes.

## Results
The paper reports better generalization to novel objects than the compared baselines and strong behavior on out-of-distribution in-the-wild scenes. The available evidence does not report representative numerical values, confidence intervals, or an ablation table. It therefore does not support a more specific quantitative comparison.

## Limitations
The two-stage design can propagate layout errors into image synthesis, and its output quality is tied to the pretrained diffusion backbone. The reported setting is centered on single-image object-conditioned hand interaction; performance for substantially more complex articulated objects or interaction configurations is not reported in the paper.
