# InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models

# Paper Summary

## Summary
InstantMesh is a feed-forward framework that generates high-quality 3D meshes from single images within 10 seconds by combining a multi-view diffusion model with a sparse-view Large Reconstruction Model (LRM) that directly predicts mesh geometry using differentiable iso-surface extraction.

## 1. Problem and Setting
- **Task**: Single-image to 3D mesh generation - creating complete 3D mesh models from a single input image
- **Input**: Single RGB image of an object
- **Output**: Complete 3D mesh representation with geometry
- **Difficulty**: Limited scale and poor annotations of 3D datasets; challenges with 3D consistency; computational inefficiency of existing triplane-based LRMs which require memory-intensive volume rendering

## 2. Core Method
**Pipeline**: Input image → Multi-view diffusion model (Zero123++) → 6 white-background multi-view images → Sparse-view LRM with differentiable iso-surface extraction → Final 3D mesh

**Key innovations**:
- **Multi-view generation**: Uses fine-tuned Zero123++ to generate 6 consistent multi-view images with white backgrounds (3×2 grid format, 960×640 resolution) at azimuths starting at 30° and increasing by 60°, with interleaving elevations of 20° and −10°
- **Sparse-view LRM**: Architecture based on LRM [14] that directly predicts 3D geometry from sparse multi-view inputs
- **Differentiable iso-surface extraction**: Integrates differentiable surface optimization techniques [39, 40] to enable direct geometric supervision (depths and normals) on the mesh representation without volume rendering
- **White-background fine-tuning**: Fine-tuned Zero123++ on LVIS subset of Objaverse to generate consistent white-background images, eliminating artifacts from inconsistent gray backgrounds

**Essential differences from existing methods**:
- Direct mesh prediction vs. triplane+MLP decoding requiring volume rendering
- Enables use of full-resolution images and geometric supervisions (depths, normals) without patch cropping
- Feed-forward inference (~10 seconds) vs. per-scene optimization in SDS-based methods

## 3. Knowledge, Supervision, and Assumptions
- **Training data**: LVIS subset of Objaverse [8] - renders query and 6 target images with white backgrounds at sampled poses
- **Pretrained models used**: Zero123++ (fine-tuned), stable diffusion base architecture
- **Geometric supervision**: Uses depths and normals as additional supervision signals during training
- **Learned vs provided**: The model learns the mapping from sparse multi-view images to complete 3D mesh geometry; camera poses follow predefined distribution pattern

## 4. Experiments and Findings
- **Datasets**: LVIS subset of Objaverse for training (mentioned in data preparation section)
- **Key metrics**: Not specified in provided text
- **Quantitative results**: Not mentioned in provided text (paper states "significantly outperforms other latest image-to-3D baselines, both qualitatively and quantitatively" without specific numbers)
- **Ablation studies**: Not mentioned in provided text
- **Real improvements**: Claims "state-of-the-art generation quality" and generation within 10 seconds

## 5. Strengths and Limitations

### Strengths
- Fast inference (within 10 seconds) for practical applications
- Feed-forward architecture without per-scene optimization
- Direct mesh output (not intermediate representations requiring post-processing)
- Training scalability on large datasets due to efficient architecture
- Can integrate arbitrary multi-view generation models (MVDream, ImageDream, SyncDreamer, SPAD, SV3D)

### Limitations
- **Implicit assumptions**: Objects are captured on white backgrounds; requires consistent multi-view generation
- **Generalization boundaries**: Training on Objaverse LVIS subset; performance on out-of-distribution objects not discussed
- **Computational costs**: Not mentioned in provided text
- **Multi-view generation dependency**: Quality of final mesh depends on multi-view diffusion model's consistency

## 6. Takeaway
InstantMesh demonstrates that combining multi-view diffusion models with sparse-view reconstruction architectures using differentiable mesh optimization enables fast, high-quality single-image-to-3D mesh generation while maintaining training scalability—addressing key limitations of both SDS-based optimization methods (speed) and triplane-based LRMs (memory efficiency).
