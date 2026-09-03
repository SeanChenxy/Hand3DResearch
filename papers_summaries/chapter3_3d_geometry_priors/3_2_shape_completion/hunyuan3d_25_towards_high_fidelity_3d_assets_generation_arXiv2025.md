# Hunyuan3D 2.5: Towards High-Fidelity 3D Assets Generation with Ultimate Details

**Authors:** Tencent Hunyuan3D Team (Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, et al.)  
**Date:** 2025-06-19 (arXiv v1)  
**Identifier:** [arXiv:2506.16504](https://arxiv.org/abs/2506.16504)  
**Zotero item:** `BZE8H456` ([Zotero](zotero://select/library/items/BZE8H456))  
**Evidence status:** Zotero metadata and the full paper PDF (main sections and result tables) were verified.

## Summary

Hunyuan3D 2.5 is a suite of 3D diffusion models for generating high-fidelity, detailed textured 3D assets from a single input image. It follows the two-stage pipeline of its predecessor Hunyuan3D 2.0 — shape generation first, then texture synthesis on the generated mesh — while advancing both stages substantially. For shape generation, the report introduces a new shape foundation model called LATTICE, trained on scaled high-quality 3D data with increased model size and compute; its largest model reaches 10B parameters and produces sharp yet smooth shapes with fine-grained details (down to correctly counted fingers and thin sub-part structures), markedly closing the gap between generated and handcrafted shapes. For texture generation, the paint model is upgraded to physical-based rendering (PBR): a multiview architecture extended from Hunyuan3D 2.0 Paint produces albedo, metallic-roughness, and normal material maps simultaneously, with a dual-channel attention mechanism and a dual-phase resolution enhancement strategy to keep material channels spatially aligned and texture consistent with geometry. Extensive quantitative and qualitative evaluations, including user studies, show that Hunyuan3D 2.5 outperforms previous open-source and closed-source commercial models in both shape generation and end-to-end texture generation.

## Background and Problem

3D generation has become a core driver across gaming, embodied AI, and film, with native 3D diffusion models built on 3DShape2VecSets-style representations (CLAY, Hunyuan3D 2.0, TripoSG, Direct3D) compressing shapes via triplanes, and Trellis further leveraging structured 3D latents. However, existing models still struggle to generate complex objects with fine-grained details — e.g., exact finger counts, thin structures, and sharp edges combined with smooth surfaces — leaving a gap relative to handcrafted assets. On the texture side, multiview-diffusion-based methods suffer from inconsistency across views, causing artifacts and seams during fusion and baking; traditional RGB textures no longer meet photorealistic 3D-asset demands, which require PBR materials, while open-source PBR material generation solutions remain unavailable. The report targets both problems: ultimate shape detail at scale, and consistent, physically plausible PBR texturing.

## Method

The pipeline processes the input image (background removal, resizing), generates an untextured 3D mesh with the shape model, post-processes the mesh (normal/UV extraction), and finally synthesizes texture with the paint model conditioned on the previous outputs.

Detailed shape generation (LATTICE): a large-scale diffusion model trained on an extensive, high-quality 3D dataset with increased model size and computational resources; scaling exhibits stable improvement. The model accepts a single image or four multiview images and is distilled with shape guidance to reduce inference time. It balances extreme detail (fine sub-structures), sharp edges, and smooth clean surfaces simultaneously.

Realistic texture generation (Hunyuan3D-Paint-PBR): built on the multiview PBR architecture of Hunyuan3D 2.1, it takes a normal map and CCM rendering by 3D mesh as geometry conditions plus a reference image, and generates high-quality PBR material maps. Three learnable embeddings (albedo, MR — the metallic-roughness combination — and normal) are injected into respective attention branches. Because material channels have significant domain gaps yet require pixel-level spatial correspondence, a dual-channel attention mechanism shares the attention mask computed from the basecolor branch (which is most semantically similar to the reference image) to guide reference attention in the other two branches, with an illumination-invariance consistency loss enforcing disentanglement of material properties from illumination.

Geometric alignment: since high-resolution images preserve high-frequency geometric details that mitigate VAE compression losses, but full-resolution multiview training is memory-prohibitive, a dual-phase resolution enhancement strategy is used — phase one trains conventionally at 6 views of 512×512 to establish multiview consistency, phase two applies a zoom-in training strategy (random zooming into reference and generated views) that learns fine-grained texture detail without full-resolution training. Inference leverages multiview images at up to 768×768 with the UniPC sampler.

## Contributions

1. LATTICE, a new 10B-parameter shape foundation model that generates highly detailed, sharp-edged yet smooth shapes, significantly closing the gap between generated and handcrafted 3D shapes.
2. An open-source high-fidelity PBR material generation framework producing multiview albedo, metallic-roughness, and normal maps, with a dual-channel attention mechanism for cross-channel spatial alignment.
3. A dual-phase resolution enhancement strategy (zoom-in training) that improves texture-geometry coordination and end-to-end visual quality under tractable memory budgets.
4. State-of-the-art quantitative and qualitative results — including user studies — over previous open-source and closed-source commercial models in both shape and end-to-end texture generation.

## Experimental Setup

Shape generation is compared against open-source baselines (Michelangelo, Craftsman 1.5, Trellis, Hunyuan3D 2.0) and two closed-source commercial models, using ULIP and Uni3D to compute similarity between the generated mesh and input images (ULIP-T/I, Uni3D-T/I with image prompts synthesized by a vision-language model). Texture generation is evaluated end-to-end against text-conditioned methods (Text2Tex, SyncMVD) and image-conditioned methods (Paint-it, Paint3D, TexGen) using FID, CLIP-FID, CMMD, CLIP-I, and LPIPS on generated versus ground-truth textures. A user study asks participants to rank each method on diverse in-the-wild testset images against three commercial models.

## Results

- Shape metrics: Hunyuan3D 2.5 achieves the best image-shape and text-shape similarities — ULIP-T 0.07853, ULIP-I 0.1306, Uni3D-T 0.2542, Uni3D-I 0.3151 — narrowly above Commercial Model 1 (0.0741/0.1308/0.2464/0.3106) and Hunyuan3D 2.0 (0.0721/0.1303/0.2519/0.3151); the authors note the metrics understate the visible quality gap shown in qualitative comparisons.
- Texture metrics (end-to-end): best across the board — CLIP-FID 23.97, FID 165.8, CMMD 2.064, CLIP-I 0.9281, LPIPS 0.1231, versus Paint3D (26.86/176.9/2.400/0.8871/0.1261) and TexGen (28.23/178.6/2.447/0.8818/0.1331).
- User study: in image-to-3D tasks the method achieves a 72% win rate, about 9 times higher than Commercial Model 1.
- Qualitatively, competing PBR models fail to estimate correct metallic/roughness values and struggle to decouple environment illumination embedded in the reference albedo, whereas Hunyuan3D 2.5 produces materially plausible renders with consistent front/back views.

## Limitations

The report is a system release rather than a research paper with exhaustive ablations: evaluation relies partly on embedding-similarity metrics that the authors themselves note cannot fully reflect model capability, so quality claims rest significantly on qualitative comparisons and user studies. Shape generation inherits the single-image-to-3D ambiguity — outputs are plausible hypotheses rather than measured geometry, with no guarantee of metric accuracy or contact/physical validity. High-fidelity texture synthesis depends on the input reference image; severe occlusion, lighting baked into the reference albedo, and transparent or reflective materials remain challenging, and the dual-phase training strategy, while memory-efficient, adds pipeline complexity. Model scale (10B parameters) also implies substantial compute for inference and deployment.
