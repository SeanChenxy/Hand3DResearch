# CogVideoX: Text-to-Video Diffusion Model

## Summary
CogVideoX is a text-to-video diffusion transformer model that generates 10-second, 768×1360 pixel videos at 16 fps through a 3D causal VAE for efficient compression and an expert transformer with adaptive LayerNorm for improved text-video alignment.

## 1. Problem and Setting
Text-to-video generation models struggle with producing long-duration, temporally consistent videos with dynamic motion and coherent narratives. Key challenges include: (1) efficiently handling high-dimensional video data, (2) maintaining alignment between text prompts and generated video content, and (3) ensuring temporal coherence across extended sequences. The paper targets generation of 10-second videos at 16 fps with 768×1360 resolution.

## 2. Core Method
**3D Causal VAE**: A video compression module using 3D convolutions for spatial-temporal compression (8×8×4 ratio) with temporally causal convolutions to prevent future information leakage. Context parallel distributes computation across devices.

**Expert Transformer**: Uses expert adaptive LayerNorm (Vision Expert AdaLN and Text Expert AdaLN) to independently process vision and text modalities within the same sequence, addressing differing feature spaces and numerical scales.

**3D-RoPE**: Extension of Rotary Position Embedding to 3D coordinates (x, y, t), with 1D-RoPE independently applied to each dimension (3/8, 3/8, 2/8 channel allocation).

**Progressive Training**: Multi-resolution frame packing and resolution progressive training with Explicit Uniform Sampling for stable loss curves.

## 3. Knowledge, Supervision, and Assumptions
**Data Pipeline**: Video captioning pipeline generates accurate textual descriptions for training data without original labels. Preprocessing strategies for both text and video data.

**Pretrained Components**: T5 (Raffel et al., 2020) for text encoding. 3D VAE trained at 256×256 resolution, 17 frames, then fine-tuned on 161-frame videos using context parallel.

**Loss Functions**: Weighted combination of L1 reconstruction loss, LPIPS perceptual loss, KL loss, and 3D discriminator GAN loss after initial training steps.

## 4. Experiments and Findings
**Models**: Two sizes trained—5B and 2B parameters. Both text-to-video and image-to-video versions released.

**Evaluation**: Automated metric evaluation and human assessment compared against openly-accessible text-to-video models. CogVideoX-5B achieves state-of-the-art performance; CogVideoX-2B competitive across dimensions.

**Ablation**: 3D VAE variants tested (Table 1) showing reduced flickering (86.3 L1 difference) and improved PSNR (28.7) with 8×8×4 compression and 16 latent channels. Higher compression (16×16×8) leads to convergence difficulties.

## 5. Strengths and Limitations
**Strengths**: First commercial-grade open-source video generation models; supports multiple aspect ratios; scalable architecture with performance improving at larger scales; comprehensive open release including code, checkpoints, VAE, and captioning models.

**Limitations**: Aggressive spatial-temporal compression (16×16×8) causes convergence difficulties; processing long-duration videos requires significant GPU memory addressed only through context parallel; captioning pipeline quality depends on pre-processing strategies.

## 6. Takeaway
CogVideoX demonstrates that 3D causal VAE compression combined with expert transformer architectures enables efficient, high-quality long-form video generation. The open release of commercial-grade models (5B/2B) with supporting components (VAE, captioning) provides a foundation for advancing video generation research. The scalable architecture suggests continued performance gains with increased model size, data volume, and training compute.
