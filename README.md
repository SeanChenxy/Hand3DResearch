# Hand-Object Interaction in the Age of Large Foundation Models: Reconstruction, Generation, and Embodied Transfer – A Survey

This repository accompanies our survey **Hand-Object Interaction in the Age of Large Foundation Models: Reconstruction, Generation, and Embodied Transfer – A Survey** — a prior-centric survey of foundation-model priors for HOI reconstruction, generation, and robot learning.

- 📄 Curated paper list with code, websites, models, benchmarks, and datasets.
- 🤖 Organized around non-foundation-prior methods and three foundation-model prior families.
- 📝 **Each paper includes an paper summary document** for deep reading.
- 🤝 If you find missing papers, outdated links, or incorrect metadata, please feel free to open an issue or submit a pull request!

---

## Table of Contents

- [1. HOI Tasks, Representations, and Non-Foundation-Prior Methods (Chapter 2)](#1-hoi-tasks-representations-and-non-foundation-prior-methods-chapter-2)
  - [1.1 Hand-Object Reconstruction](#11-hand-object-reconstruction)
  - [1.2 Hand-Held Object Reconstruction](#12-hand-held-object-reconstruction)
  - [1.3 Hand-Object Motion Reconstruction](#13-hand-object-motion-reconstruction)
  - [1.4 Hand-Object Grasp Generation](#14-hand-object-grasp-generation)
  - [1.5 Hand-Object Motion Generation](#15-hand-object-motion-generation)
  - [1.6 Hand-Object Image/Video Generation](#16-hand-object-imagevideo-generation)
- [2. 3D Geometry Priors for HOI (Chapter 3)](#2-3d-geometry-priors-for-hoi-chapter-3)
  - [2.1 Prior Sources](#21-prior-sources)
  - [2.2 Shape Completion Priors](#22-shape-completion-priors)
  - [2.3 Shape Retrieval Priors](#23-shape-retrieval-priors)
  - [2.4 Spatial Geometry Priors](#24-spatial-geometry-priors)
- [3. Semantic Priors for HOI (Chapter 4)](#3-semantic-priors-for-hoi-chapter-4)
  - [3.1 Prior Sources](#31-prior-sources)
  - [3.2 Visual Grounding Priors](#32-visual-grounding-priors)
  - [3.3 Language Reasoning Priors](#33-language-reasoning-priors)
- [4. Visual Generative Priors for HOI (Chapter 5)](#4-visual-and-motion-generative-priors-for-hoi-chapter-5)
  - [4.1 Prior Sources](#41-prior-sources)
  - [4.2 Image Generative Priors](#42-image-generative-priors)
  - [4.3 Video Generative Priors](#43-video-generative-priors)
    - [4.3.1 From HOI Video Generation to HOI World Models](#431-from-hoi-video-generation-to-hoi-world-models)
  - [4.4 Visual Representation Priors](#44-visual-representation-priors)
- [5. HOI for Robot Learning (Chapter 6)](#5-hoi-for-robot-learning-chapter-6)
  - [5.1 Generalist Policy Learning: Video-Based Pretraining](#51-generalist-policy-learning-video-based-pretraining)
  - [5.2 Generalist Policy Learning: Structured HOI Supervision](#52-generalist-policy-learning-structured-hoi-supervision)
  - [5.3 Task-Specific Skill Transfer: Dexterous Motion Retargeting](#53-task-specific-skill-transfer-dexterous-motion-retargeting)
  - [5.4 Task-Specific Skill Transfer: Interaction-Guided Policy Learning](#54-task-specific-skill-transfer-interaction-guided-policy-learning)
  - [5.5 HOI-to-Robot Data Engines](#55-hoi-to-robot-data-engines)
- [7. Benchmarks (Chapter 7)](#7-benchmarks-chapter-7)
  - [7.1 Reconstruction Benchmarks](#71-reconstruction-benchmarks)
  - [7.2 Generation Benchmarks](#72-generation-benchmarks)
  - [7.3 Embodied Learning Data Sources](#73-embodied-learning-data-sources)

---

<a id="1-hoi-tasks-representations-and-non-foundation-prior-methods-chapter-2"></a>
## 1. HOI Tasks, Representations, and Non-Foundation-Prior Methods (Chapter 2)

> This chapter covers prior-free supervised baselines and domain-intrinsic prior methods across six HOI tasks. These methods do NOT invoke foundation-model knowledge, serving as the baseline against which foundation-model prior gains are measured.

<a id="11-hand-object-reconstruction"></a>
### 1.1 Hand-Object Reconstruction

> Recovering hand-only or hand-object spatial state from single-frame observations. Output: 2D/3D keypoints, joint angles, MANO parameters, hand mesh, object 6D pose, or hand-object spatial configuration. Methods with contact-field outputs are marked `[+contact]`.

- **HandGCAT** — *HandGCAT: Occlusion-Robust 3D Hand Mesh Reconstruction from Monocular Images*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.07912-b31b1b.svg)](http://arxiv.org/abs/2403.07912) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/HandGCAT_arXiv2023.md)
- **Keypoint Transformer** — *Keypoint Transformer: Solving Joint Identification in Challenging Hands and Object Interactions for Accurate 3D Pose Estimation*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9880464/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52688.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52688.2022.01081) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Keypoint_Transformer_arXiv2022.md)
- **Collaborative HOI Recon** — *Collaborative Learning for Hand and Object Reconstruction with Attention-guided Graph Convolution*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9878674/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52688.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52688.2022.00171) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Collaborative_Learning_for_Hand_and_Object_arXiv2022.md)
- **MobRecon** — *MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular Image*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9878887/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52688.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52688.2022.01989) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/MobRecon_arXiv2022.md)
- **Semi-Supervised HOI Pose** — *Semi-Supervised 3D Hand-Object Poses Estimation with Interactions in Time*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9577481/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR46437.20-4B5D67.svg)](https://doi.org/10.1109/CVPR46437.2021.01445) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Semi_Supervised_3D_Hand_Object_Poses_Estimation_with_arXiv2021.md)
- **HOPE-Net** — *HOPE-Net: A Graph-Based Model for Hand-Object Pose Estimation*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9156657/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR42600.20-4B5D67.svg)](https://doi.org/10.1109/CVPR42600.2020.00664) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/HOPE_Net_arXiv2020.md)
- **Photometric Consistency HOI** — *Leveraging Photometric Consistency Over Time for Sparsely Supervised Hand-Object Reconstruction*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9156936/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR42600.20-4B5D67.svg)](https://doi.org/10.1109/CVPR42600.2020.00065) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Leveraging_Photometric_Consistency_Over_Time_for_arXiv2020.md)
- **H+O** — *H+O: Unified Egocentric Recognition of 3D Hand-Object Poses and Interactions*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/8953449/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR.2019.00-4B5D67.svg)](https://doi.org/10.1109/CVPR.2019.00464) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/H_O_arXiv2019.md)
- **Joint HOI Recon** — *Learning Joint Reconstruction of Hands and Manipulated Objects*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/8954029/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR.2019.01-4B5D67.svg)](https://doi.org/10.1109/CVPR.2019.01208) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Joint_Reconstruction_of_Hands_and_Manipulated_arXiv2019.md)
- **A Simple Baseline for Efficient Hand Mesh Reconstruction** — *A Simple Baseline for Efficient Hand Mesh Reconstruction*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10657733/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00136) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Simple_Baseline_for_Efficient_Hand_Mesh_arXiv2024.md)
- **Reconstructing Hands in 3D with Transformers** — *Reconstructing Hands in 3D with Transformers*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10655481/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00938) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Reconstructing_Hands_in_3D_with_Transformers_arXiv2024.md)
- **ContactArt** — *ContactArt: Learning 3D Interaction Priors for Category-level Articulated Object and Hand Poses Estimation*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10550760/) [![DOI](https://img.shields.io/badge/DOI-10.1109/3DV62453.202-4B5D67.svg)](https://doi.org/10.1109/3DV62453.2024.00028) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/ContactArt_arXiv2024.md)
- **NCRF** — *NCRF: Neural Contact Radiance Fields for Free-Viewpoint Rendering of Hand-Object Interaction*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10550790/) [![DOI](https://img.shields.io/badge/DOI-10.1109/3DV62453.202-4B5D67.svg)](https://doi.org/10.1109/3DV62453.2024.00091) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/NCRF_arXiv2024.md)
- **DeepSimHO** — *DeepSimHO: Stable Pose Estimation for Hand-Object Interaction via Physics Simulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2310.07206-b31b1b.svg)](http://arxiv.org/abs/2310.07206) [![Paper](https://img.shields.io/badge/Paper-NeurIPS-4B5D67.svg)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fbdaea4878318e214c0577dae4b8bc43-Abstract-Conference.html) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/DeepSimHO_arXiv2023.md)
- **S²Contact** — *S²Contact: Graph-Based Network for 3D Hand-Object Contact Estimation with Semi-supervised Learning*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://link.springer.com/10.1007/978-3-031-19769-7_33) [![DOI](https://img.shields.io/badge/DOI-10.1007/978-3-031-19-4B5D67.svg)](https://doi.org/10.1007/978-3-031-19769-7_33) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/S_2_Contact_arXiv2022.md)
- **CPF** — *CPF: Learning a Contact Potential Field to Model the Hand-Object Interaction*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9710247/) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICCV48922.20-4B5D67.svg)](https://doi.org/10.1109/ICCV48922.2021.01091) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/CPF_arXiv2021.md)
- **THOR-Net** — *THOR-Net: End-to-end Graformer-based Realistic Two Hands and Object Reconstruction with Self-supervision*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10030162/) [![DOI](https://img.shields.io/badge/DOI-10.1109/WACV56688.20-4B5D67.svg)](https://doi.org/10.1109/WACV56688.2023.00106) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/THOR_Net_arXiv2023.md)
- **QORT-Former** — *QORT-Former: Query-optimized Real-time Transformer for Understanding Two Hands Manipulating Objects*
  [![arXiv](https://img.shields.io/badge/arXiv-2502.19769-b31b1b.svg)](https://arxiv.org/abs/2502.19769) [![Paper](https://img.shields.io/badge/Paper-AAAI-4B5D67.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/32407) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/QORT_Former_arXiv.md)
- **CHOIR [WACV 2025]** — *A Versatile and Differentiable Hand-Object Interaction Representation*
  [![arXiv](https://img.shields.io/badge/arXiv-2409.16855-b31b1b.svg)](https://arxiv.org/abs/2409.16855) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://www.openaccess.thecvf.com/content/WACV2025/html/Morales_A_Versatile_and_Differentiable_Hand-Object_Interaction_Representation_WACV_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Versatile_and_Differentiable_Hand_Object_Interaction_Represe_arXiv.md)
- **WiLoR** — *WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild*
  [![arXiv](https://img.shields.io/badge/arXiv-2409.12259-b31b1b.svg)](https://arxiv.org/abs/2409.12259) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/WiLoR_arXiv.md)
- **MaskHand** — *MaskHand: Generative Masked Modeling for Robust Hand Mesh Reconstruction in the Wild*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.13393-b31b1b.svg)](https://arxiv.org/abs/2412.13393) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/MaskHand_arXiv.md)
- **Hamba** — *Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba*
  [![arXiv](https://img.shields.io/badge/arXiv-2407.09646-b31b1b.svg)](https://arxiv.org/abs/2407.09646) [![Paper](https://img.shields.io/badge/Paper-NeurIPS-4B5D67.svg)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/03e9a69e5b686c316a07d73f0cf5e225-Abstract-Conference.html) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/humansensinglab/Hamba) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_1_hand_object_recon/Hamba_arXiv.md)

<a id="12-hand-held-object-reconstruction"></a>
### 1.2 Hand-Held Object Reconstruction

> Recovering the shape, surface, mesh, or implicit field of hand-held objects under severe hand occlusion. Emphasizes object-agnostic / open-world object recovery beyond known-template 6D pose estimation.

- **gSDF** — *gSDF: Geometry-Driven Signed Distance Functions for 3D Hand-Object Reconstruction*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10203349/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52729.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52729.2023.01239) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/gSDF_arXiv2023.md)
- **In-Hand 3D Object Scanning from an RGB Sequence** — *In-Hand 3D Object Scanning from an RGB Sequence*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10203411/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52729.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52729.2023.01638) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/In_Hand_3D_Object_Scanning_from_an_arXiv2023.md)
- **What's in Your Hands?** — *What's in your hands? 3D Reconstruction of Generic Objects in Hands*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9879463/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52688.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52688.2022.00387) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/What_s_in_your_hands_3D_Reconstruction_arXiv2022.md)
- **HOISDF** — *HOISDF: Constraining 3D Hand-Object Pose Estimation with Global Signed Distance Fields*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10657272/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00989) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/HOISDF_arXiv2024.md)
- **In-Hand 3D Object Reconstruction from a Monocular RGB Video** — *In-Hand 3D Object Reconstruction from a Monocular RGB Video*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/28029) [![DOI](https://img.shields.io/badge/DOI-10.1609/aaai.v38i3.2-4B5D67.svg)](https://doi.org/10.1609/aaai.v38i3.28029) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/In_Hand_3D_Object_Reconstruction_from_a_AAAI2024.md)
- **Chord** — *Chord: Category-level Hand-held Object Reconstruction via Shape Deformation*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10377806/) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICCV51070.20-4B5D67.svg)](https://doi.org/10.1109/ICCV51070.2023.00866) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/Chord_arXiv2023.md)
- **Reconstructing Hand-Held Objects from Monocular Video** — *Reconstructing Hand-Held Objects from Monocular Video*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://dl.acm.org/doi/10.1145/3550469.3555401) [![DOI](https://img.shields.io/badge/DOI-10.1145/3550469.3555-4B5D67.svg)](https://doi.org/10.1145/3550469.3555401) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/Reconstructing_Hand_Held_Objects_from_Monocular_Video_arXiv2022.md)
- **AlignSDF** — *AlignSDF: Pose-Aligned Signed Distance Fields for Hand-Object Reconstruction*
  [![arXiv](https://img.shields.io/badge/arXiv-2207.12909-b31b1b.svg)](http://arxiv.org/abs/2207.12909) [![Paper](https://img.shields.io/badge/Paper-Springer-4B5D67.svg)](https://link.springer.com/chapter/10.1007/978-3-031-19769-7_14) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://zerchen.github.io/projects/alignsdf.html) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/AlignSDF_arXiv2022.md)
- **MOHO** — *MOHO: Learning Single-view Hand-held Object Reconstruction with Multi-view Occlusion-Aware Supervision*
  [![arXiv](https://img.shields.io/badge/arXiv-2310.11696-b31b1b.svg)](https://arxiv.org/abs/2310.11696) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_MOHO_Learning_Single-view_Hand-held_Object_Reconstruction_with_Multi-view_Occlusion-Aware_Supervision_CVPR_2024_paper.html) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/ZhangCYG/MOHO) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/Zhang_MOHO_Learning_Single_view_Hand_held_Object_Reconstruct_CVPR.md)
- **TexHOI** — *TexHOI: Reconstructing Textures of 3D Unknown Objects in Monocular Hand-Object Interaction Scenes*
  [![arXiv](https://img.shields.io/badge/arXiv-2501.03525-b31b1b.svg)](https://arxiv.org/abs/2501.03525) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_2_hand_held_object_recon/TexHOI_arXiv.md)

<a id="13-hand-object-motion-reconstruction"></a>
### 1.3 Hand-Object Motion Reconstruction

> Recovering hand, object, camera, contact, and interaction state evolution over time from video or temporal observations. Includes 4D/renderable outputs as a representation branch, not a separate task.

- **Interaction-Aware 4DGS** — *Interaction-Aware 4D Gaussian Splatting for Dynamic Hand-Object Interaction Reconstruction*
  [![arXiv](https://img.shields.io/badge/arXiv-2511.14540-b31b1b.svg)](https://arxiv.org/abs/2511.14540) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/Interaction_Aware_4D_Gaussian_Splatting_for_Dynamic_arXiv2025.md)
- **SIGHT** — *SIGHT: Synthesizing Image-Text Conditioned and Geometry-Guided 3D Hand-Object Trajectories*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.22869-b31b1b.svg)](https://arxiv.org/abs/2503.22869) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/SIGHT_arXiv2025.md)
- **HOLD** — *HOLD: Category-Agnostic 3D Reconstruction of Interacting Hands and Objects from Video*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10658613/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00054) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/HOLD_arXiv2024.md)
- **BOTH2Hands** — *BOTH2Hands: Inferring 3D Hands from Both Text Prompts and Body Dynamics*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10658110/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00232) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/BOTH2Hands_arXiv2024.md)
- **InterHandGen** — *InterHandGen: Two-Hand Interaction Generation via Cascaded Reverse Diffusion*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10658183/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00057) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/InterHandGen_arXiv2024.md)
- **HandNeRF** — *HandNeRF: Learning to Reconstruct Hand-Object Interaction Scene from a Single RGB Image*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10611230/) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICRA57147.20-4B5D67.svg)](https://doi.org/10.1109/ICRA57147.2024.10611230) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/HandNeRF_arXiv2024.md)
- **Gaze-guided Hand-Object Interaction Synthesis** — *Gaze-guided Hand-Object Interaction Synthesis: Dataset and Method*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.16169-b31b1b.svg)](https://arxiv.org/abs/2403.16169) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://takiee.github.io/gaze-hoi/) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/Gaze_guided_Hand_Object_Interaction_Synthesis_arXiv2024.md)
- **PhysHOI** — *PhysHOI: Physics-Based Imitation of Dynamic Human-Object Interaction*
  [![arXiv](https://img.shields.io/badge/arXiv-2312.04393-b31b1b.svg)](http://arxiv.org/abs/2312.04393) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/PhysHOI_arXiv2023.md)
- **Model-based 3D Hand Recon** — *Model-based 3D Hand Reconstruction via Self-Supervised Learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2103.11703-b31b1b.svg)](http://arxiv.org/abs/2103.11703) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/Model_based_3D_Hand_Reconstruction_via_Self_Supervised_arXiv2021.md)
- **SeqHAND** — *SeqHAND: RGB-Sequence-Based 3D Hand Pose and Shape Estimation*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://link.springer.com/10.1007/978-3-030-58610-2_8) [![DOI](https://img.shields.io/badge/DOI-10.1007/978-3-030-58-4B5D67.svg)](https://doi.org/10.1007/978-3-030-58610-2_8) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/SeqHAND_arXiv2020.md)
- **InteractionFusion** — *InteractionFusion: real-time reconstruction of hand poses and deformable objects in hand-object interactions*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://dl.acm.org/doi/10.1145/3306346.3322998) [![DOI](https://img.shields.io/badge/DOI-10.1145/3306346.3322-4B5D67.svg)](https://doi.org/10.1145/3306346.3322998) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/InteractionFusion_arXiv2019.md)
- **Unconstrained HOI Recon** — *Towards Unconstrained Joint Hand-Object Reconstruction From RGB Videos*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9665955/) [![DOI](https://img.shields.io/badge/DOI-10.1109/3DV53792.202-4B5D67.svg)](https://doi.org/10.1109/3DV53792.2021.00075) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/Unconstrained_Joint_Hand_Object_Reconstruction_From_RGB_arXiv2021.md)
- **BIGS** — *BIGS: Bimanual Category-agnostic Interaction Reconstruction from Monocular Videos via 3D Gaussian Splatting*
  [![arXiv](https://img.shields.io/badge/arXiv-2504.09097-b31b1b.svg)](https://arxiv.org/abs/2504.09097) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/BIGS_arXiv.md)
- **HOGSA** — *HOGSA: Bimanual Hand-Object Interaction Understanding with 3D Gaussian Splatting Based Data Augmentation*
  [![arXiv](https://img.shields.io/badge/arXiv-2501.02845-b31b1b.svg)](https://arxiv.org/abs/2501.02845) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/HOGSA_arXiv.md)
- **LatentHOI** — *LatentHOI: On the Generalizable Hand Object Motion Generation with Latent Hand Diffusion.*
  [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/LatentHOI_arXiv.md)
- **LatentAct** — *How Do I Do That? Synthesizing 3D Hand Motion and Contacts for Everyday Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2504.12284-b31b1b.svg)](https://arxiv.org/abs/2504.12284) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/ap229997/latentact) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/How_Do_I_Do_That_Synthesizing_arXiv.md)
- **BimArt** — *BimArt: A Unified Approach for the Synthesis of 3D Bimanual Interaction with Articulated Objects*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.05066-b31b1b.svg)](https://arxiv.org/abs/2412.05066) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/BimArt_arXiv.md)
- **HandDiffuse** — *HandDiffuse: Generative Controllers for Two-Hand Interactions via Diffusion Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2312.04867-b31b1b.svg)](https://arxiv.org/abs/2312.04867) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_3_3_hand_object_motion_recon/HandDiffuse_arXiv.md)

<a id="14-hand-object-grasp-generation"></a>
### 1.4 Hand-Object Grasp Generation

> Generating plausible static hand grasp poses, hand meshes, or contact configurations given object, language, contact, or functional targets.

- **ContactOpt** — *ContactOpt: Optimizing Contact to Improve Grasps*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9578455/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR46437.20-4B5D67.svg)](https://doi.org/10.1109/CVPR46437.2021.00152) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/ContactOpt_arXiv2021.md)
- **GanHand** — *GanHand: Predicting Human Grasp Affordances in Multi-Object Scenes*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9156512/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR42600.20-4B5D67.svg)](https://doi.org/10.1109/CVPR42600.2020.00508) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/GanHand_arXiv2020.md)
- **UGG** — *UGG: Unified Generative Grasping*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://link.springer.com/10.1007/978-3-031-72855-6_24) [![DOI](https://img.shields.io/badge/DOI-10.1007/978-3-031-72-4B5D67.svg)](https://doi.org/10.1007/978-3-031-72855-6_24) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/UGG_arXiv2025.md)
- **GEARS** — *GEARS: Local Geometry-Aware Hand-Object Interaction Synthesis*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10657454/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.01950) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/GEARS_arXiv2024.md)
- **ClickDiff** — *ClickDiff: Click to Induce Semantic Contact Map for Controllable Grasp Generation with Diffusion Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2407.19370-b31b1b.svg)](http://arxiv.org/abs/2407.19370) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/ClickDiff_arXiv2024.md)
- **ContactGen** — *ContactGen: Generative Contact Modeling for Grasp Generation*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10378632/) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICCV51070.20-4B5D67.svg)](https://doi.org/10.1109/ICCV51070.2023.01884) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/ContactGen_arXiv2023.md)
- **Contact2Grasp** — *Contact2Grasp: 3D Grasp Synthesis via Hand-Object Contact Constraint*
  [![arXiv](https://img.shields.io/badge/arXiv-2210.09245-b31b1b.svg)](http://arxiv.org/abs/2210.09245) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/Contact2Grasp_arXiv2023.md)
- **Grasping Field** — *Grasping Field: Learning Implicit Representations for Human Grasps*
  [![arXiv](https://img.shields.io/badge/arXiv-2008.04451-b31b1b.svg)](http://arxiv.org/abs/2008.04451) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/Grasping_Field_arXiv2020.md)
- **Contact Consistency Grasp** — *Hand-Object Contact Consistency Reasoning for Human Grasps Generation*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9710689/) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICCV48922.20-4B5D67.svg)](https://doi.org/10.1109/ICCV48922.2021.01092) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/Hand_Object_Contact_Consistency_Reasoning_for_Human_arXiv2021.md)
- **MGD** — *Multi-Modal Diffusion for Hand-Object Grasp Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2409.04560-b31b1b.svg)](https://arxiv.org/abs/2409.04560) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/Multi_Modal_Diffusion_for_Hand_Object_Grasp_Generation_arXiv.md)
- **SNS-Grasp** — *SNS-Grasp: Semantic-guided Noise Scaling for Grasp Generation*
  [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_1_grasp_generation/SNS_Grasp_arXiv.md)

<a id="15-hand-object-motion-generation"></a>
### 1.5 Hand-Object Motion Generation

> Generating hand motion, object motion, interaction stages, and state transitions over time as manipulation sequences.

- **D-Grasp** — *D-Grasp: Physically Plausible Dynamic Grasp Synthesis for Hand-Object Interactions*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9880342/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52688.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52688.2022.01992) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_2_motion_generation/D_Grasp_arXiv2022.md)
- **G-HOP** — *G-HOP: Generative Hand-Object Prior for Interaction Reconstruction and Grasp Synthesis*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10658436/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00187) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_2_motion_generation/G_HOP_arXiv2024.md)
- **ArtiGrasp** — *ArtiGrasp: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2309.03891-b31b1b.svg)](http://arxiv.org/abs/2309.03891) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://eth-ait.github.io/artigrasp/) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_2_motion_generation/ArtiGrasp_arXiv2024.md)

<a id="16-hand-object-imagevideo-generation"></a>
### 1.6 Hand-Object Image/Video Generation

> Generating or editing images/videos containing hand-object interactions. Editing, inpainting, object swap, and reenactment are treated as conditional generation, not separate tasks.

- **Hand-Object Interaction Image Generation** — *Hand-Object Interaction Image Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2211.15663-b31b1b.svg)](http://arxiv.org/abs/2211.15663) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://play-with-hoi-generation.github.io/) [📝 Paper Summary](papers_summaries/chapter2_non_foundation/2_4_3_image_video_generation/Hand_Object_Interaction_Image_Generation_arXiv2022.md)

---

<a id="2-3d-geometry-priors-for-hoi-chapter-3"></a>
## 2. 3D Geometry Priors for HOI (Chapter 3)

> Foundation models provide open-world 3D shape and spatial geometry knowledge to mitigate occlusion, unseen regions, and dynamic-camera geometry uncertainty.

<a id="21-prior-sources"></a>
### 2.1 Prior Sources

> `[Prior Source]` These are the upstream foundation models that provide 3D geometry knowledge, not HOI methods per se.

<!-- PAPER_LIST_PRIOR_SOURCES_CH3 -->

<a id="22-shape-completion-priors"></a>
### 2.2 Shape Completion Priors

> Foundation models (Zero-1-to-3, Wonder3D, TripoSR, InstantMesh, etc.) provide open-world shape completion knowledge for occluded objects. These priors initialize complete object geometry before HOI-specific optimization with hand pose, silhouette, temporal, and contact constraints.

- **AGILE** — *AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.04672-b31b1b.svg)](http://arxiv.org/abs/2602.04672) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/AGILE_arXiv2026.md)
- **Grasp in Gaussians** — *Grasp in Gaussians: Fast Monocular Reconstruction of Dynamic Hand-Object Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2604.12929-b31b1b.svg)](https://arxiv.org/abs/2604.12929) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://aidilayce.github.io/GraG-page/) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/Grasp_in_Gaussians_arXiv2026.md)
- **ForeHOI** — *ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.06226-b31b1b.svg)](https://arxiv.org/abs/2602.06226) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://tao-11-chen.github.io/project_pages/ForeHOI/) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/ForeHOI_arXiv2026.md)
- **DynHOR** — *Hand-held Object Reconstruction from RGB Video with Dynamic Interaction*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/11092778/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52734.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52734.2025.01141) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/Hand_held_Object_Reconstruction_from_RGB_Video_arXiv2025.md)
- **Follow My Hold** — *Follow My Hold: Hand-Object Interaction Reconstruction through Geometric Guidance*
  [![arXiv](https://img.shields.io/badge/arXiv-2508.18213-b31b1b.svg)](http://arxiv.org/abs/2508.18213) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://aidilayce.github.io/FollowMyHold-page/) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/Follow_My_Hold_arXiv2025.md)
- **MagicHOI** — *MagicHOI: Leveraging 3D Priors for Accurate Hand-object Reconstruction from Short Monocular Video Clips*
  [![arXiv](https://img.shields.io/badge/arXiv-2508.05506-b31b1b.svg)](http://arxiv.org/abs/2508.05506) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/MagicHOI_arXiv2025.md)
- **Diffusion-Guided HOI Recon** — *Diffusion-Guided Reconstruction of Everyday Hand-Object Interaction Clips*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10377618/) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICCV51070.20-4B5D67.svg)](https://doi.org/10.1109/ICCV51070.2023.01806) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/Diffusion_Guided_Reconstruction_of_Everyday_Hand_Object_Inte_arXiv2023.md)
- **EasyHOI** — *EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild*
  [![arXiv](https://img.shields.io/badge/arXiv-2411.14280-b31b1b.svg)](https://arxiv.org/abs/2411.14280) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_EasyHOI_Unleashing_the_Power_of_Large_Models_for_Reconstructing_Hand-Object_CVPR_2025_paper.html) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/lym29/EasyHOI) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/EasyHOI_arXiv.md)

**Prior Source Papers:**
- **Hunyuan3D 2.0** — *Hunyuan3D 2.0: Scaling Diffusion Models for High Resolution Textured 3D Assets Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2501.12202-b31b1b.svg)](https://arxiv.org/abs/2501.12202) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/hunyuan3d_20_scaling_diffusion_models_for_high_resolution_textured_3d_assets_gen_arXiv2026.md)
- **TripoSR** — *TripoSR: Fast 3D Object Reconstruction from a Single Image*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.02151-b31b1b.svg)](https://arxiv.org/abs/2403.02151) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/VAST-AI-Research/TripoSR) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/triposr_fast_3d_object_reconstruction_from_a_single_image_arXiv2024.md)
- **InstantMesh** — *InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models* [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/TencentARC/InstantMesh) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/instantmesh_efficient_3d_mesh_generation_from_a_single_image_with_sparse_view_la_arXiv2024.md)
- **DreamGaussian** — *DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation*
  [![arXiv](https://img.shields.io/badge/arXiv-2309.16653-b31b1b.svg)](https://arxiv.org/abs/2309.16653) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/dreamgaussian_generative_gaussian_splatting_for_efficient_3d_content_creation_arXiv2024.md)
- **Wonder3D** — *Wonder3D: Single Image to 3D Using Cross-Domain Diffusion* [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/wonder3d_single_image_to_3d_using_cross_domain_diffusion_arXiv2024.md)
- **Zero-1-to-3** — *Zero-1-to-3: Zero-shot One Image to 3D Object*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10378322/) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICCV51070.23-4B5D67.svg)](https://doi.org/10.1109/ICCV51070.2023.00853) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/zero_1_to_3_zero_shot_one_image_to_3d_object_arXiv2023.md)
- **Objaverse** — *Objaverse: A Universe of Annotated 3D Objects*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10205395/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52729.23-4B5D67.svg)](https://doi.org/10.1109/CVPR52729.2023.01263) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/objaverse_a_universe_of_annotated_3d_objects_arXiv2023.md)
- **ShapeNet** — *ShapeNet: An Information-Rich 3D Model Repository*
  [![arXiv](https://img.shields.io/badge/arXiv-1512.03012-b31b1b.svg)](https://arxiv.org/abs/1512.03012) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/shapenet_an_information_rich_3d_model_repository_arXiv2015.md)

<a id="23-shape-retrieval-priors"></a>
### 2.3 Shape Retrieval Priors

> External 3D asset databases (Objaverse, ShapeNet, ABO) and vision-language embeddings (CLIP, DINOv2, OpenShape) provide topologically stable shape candidates for hand-held object reconstruction.

- **GHOST** — *GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.18912-b31b1b.svg)](http://arxiv.org/abs/2603.18912) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/GHOST_arXiv2026.md)
- **Reconstructing Hand-Held Objects in 3D from Images and Video** — *Reconstructing Hand-Held Objects in 3D from Images and Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2404.06507-b31b1b.svg)](http://arxiv.org/abs/2404.06507) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://janehwu.github.io/mcc-ho) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/Reconstructing_Hand_Held_Objects_in_3D_from_arXiv2025.md)
- **DynHOR** — *Hand-held Object Reconstruction from RGB Video with Dynamic Interaction*
  [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/Hand_held_Object_Reconstruction_from_RGB_Video_arXiv.md)
- **PICO** — *PICO: Reconstructing 3D People In Contact with Objects*
  [![arXiv](https://img.shields.io/badge/arXiv-2504.17695-b31b1b.svg)](https://arxiv.org/abs/2504.17695) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/PICO_arXiv.md)

**Prior Source Papers:**
- **DINOv2** — *DINOv2: Learning Robust Visual Features without Supervision*
  [![arXiv](https://img.shields.io/badge/arXiv-2304.07193-b31b1b.svg)](https://arxiv.org/abs/2304.07193) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/dinov2_learning_robust_visual_features_without_supervision_arXiv2024.md)
- **Objaverse** — *Objaverse: A Universe of Annotated 3D Objects*
  [![arXiv](https://img.shields.io/badge/arXiv-2212.08051-b31b1b.svg)](https://arxiv.org/abs/2212.08051) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/objaverse_a_universe_of_annotated_3d_objects_arXiv2023.md)
- **ABO** — *Amazon Berkeley Objects (ABO): Dataset and Benchmarks for Real-World 3D Object Understanding*
  [![arXiv](https://img.shields.io/badge/arXiv-2111.01078-b31b1b.svg)](https://arxiv.org/abs/2111.01078) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/abo_dataset_and_benchmarks_for_real_world_3d_object_understanding_arXiv2022.md)
- **Google Scanned Objects** — *Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9811809) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICRA46639.20-4B5D67.svg)](https://doi.org/10.1109/ICRA46639.2022.9811809) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/google_scanned_objects_a_high_quality_dataset_of_3d_scanned_household_items_arXiv2022.md)
- **CLIP** — *Learning Transferable Visual Models From Natural Language Supervision*
  [![arXiv](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg)](https://arxiv.org/abs/2103.00020) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/learning_transferable_visual_models_from_natural_language_supervision_arXiv2021.md)

<a id="24-spatial-geometry-priors"></a>
### 2.4 Spatial Geometry Priors

> Pre-trained visual geometry models (DUSt3R, MASt3R, VGGT, MoGe2, Depth Anything 3) inject depth, camera, point maps, cross-view correspondence, and world-space alignment into HOI pipelines.

- **GeoHand** — *GeoHand: Unlocking Prior Geometry Knowledge for Monocular 3D Hand Reconstruction*
  [![arXiv](https://img.shields.io/badge/arXiv-2605.17354-b31b1b.svg)](http://arxiv.org/abs/2605.17354) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/GeoHand_arXiv2026.md)
- **Grasp in Gaussians** — *Grasp in Gaussians: Fast Monocular Reconstruction of Dynamic Hand-Object Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2604.12929-b31b1b.svg)](http://arxiv.org/abs/2604.12929) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://aidilayce.github.io/GraG-page/) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/Grasp_in_Gaussians_arXiv2026.md)
- **ArtHOI** — *ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.25791-b31b1b.svg)](http://arxiv.org/abs/2603.25791) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/ArtHOI_arXiv2026.md)
- **HGGT** — *HGGT: Robust and Flexible 3D Hand Mesh Reconstruction from Uncalibrated Images*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.23997-b31b1b.svg)](http://arxiv.org/abs/2603.23997) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://lym29.github.io/HGGT/) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/HGGT_arXiv2026.md)
- **EgoGrasp** — *EgoGrasp: World-Space Hand-Object Interaction Estimation from Egocentric Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2601.01050-b31b1b.svg)](http://arxiv.org/abs/2601.01050) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/EgoGrasp_arXiv2026.md)
- **WHOLE** — *WHOLE: World-Grounded Hand-Object Lifted from Egocentric Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.22209-b31b1b.svg)](http://arxiv.org/abs/2602.22209) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://judyye.github.io/whole-www) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/WHOLE_arXiv2026.md)
- **Hand3R** — *Hand3R: Online 4D Hand-Scene Reconstruction in the Wild*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.03200-b31b1b.svg)](http://arxiv.org/abs/2602.03200) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/Hand3R_arXiv2026.md)
- **HOSt3R** — *HOSt3R: Keypoint-free Hand-Object 3D Reconstruction from RGB images*
  [![arXiv](https://img.shields.io/badge/arXiv-2508.16465-b31b1b.svg)](https://arxiv.org/abs/2508.16465) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://www.openaccess.thecvf.com/content/ICCV2025W/HANDS/html/Swamy_Host3R_Keypoint-free_Hand-Object_3D_Reconstruction_from_RGB_images_ICCVW_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/Host3R_arXiv.md)
- **HaWoR** — *HaWoR: World-Space Hand Motion Reconstruction from Egocentric Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2501.02973-b31b1b.svg)](https://arxiv.org/abs/2501.02973) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_HaWoR_World-Space_Hand_Motion_Reconstruction_from_Egocentric_Videos_CVPR_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/HaWoR_arXiv.md)
- **Dyn-HaMR** — *Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.12861-b31b1b.svg)](https://arxiv.org/abs/2412.12861) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/Dyn_HaMR_arXiv.md)

**Prior Source Papers:**
- **Depth Anything 3** — *Depth Anything 3: Recovering the Visual Space from Any Views*
  [![arXiv](https://img.shields.io/badge/arXiv-2511.10647-b31b1b.svg)](https://arxiv.org/abs/2511.10647) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/depth_anything_3_recovering_the_visual_space_from_any_views_arXiv2025.md)
- **MoGe-2** — *MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.12067-b31b1b.svg)](https://arxiv.org/abs/2412.12067) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/moge_2_accurate_monocular_geometry_with_metric_scale_and_sharp_details_arXiv2025.md)
- **VGGT** — *VGGT: Visual Geometry Grounded Transformer*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.11651-b31b1b.svg)](https://arxiv.org/abs/2503.11651) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/vggt_visual_geometry_grounded_transformer_arXiv2025.md)
- **MASt3R** — *Grounding Image Matching in 3D with MASt3R*
  [![arXiv](https://img.shields.io/badge/arXiv-2406.09756-b31b1b.svg)](https://arxiv.org/abs/2406.09756) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/grounding_image_matching_in_3d_with_mast3r_arXiv2024.md)
- **DUSt3R** — *DUSt3R: Geometric 3D Vision Made Easy*
  [![arXiv](https://img.shields.io/badge/arXiv-2312.14132-b31b1b.svg)](https://arxiv.org/abs/2312.14132) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/dust3r_geometric_3d_vision_made_easy_arXiv2024.md)
- **CroCo** — *CroCo: Self-Supervised Pre-training for 3D Vision Tasks by Cross-View Completion*
  [![arXiv](https://img.shields.io/badge/arXiv-2210.10716-b31b1b.svg)](https://arxiv.org/abs/2210.10716) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/croco_self_supervised_pre_training_for_3d_vision_tasks_by_cross_view_completion_arXiv2023.md)
- **CroCo v2** — *CroCo v2: Improved Cross-view Completion Pre-training for Stereo Matching and Optical Flow*
  [![arXiv](https://img.shields.io/badge/arXiv-2211.10408-b31b1b.svg)](https://arxiv.org/abs/2211.10408) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_4_spatial_geometry/croco_v2_improved_cross_view_completion_pre_training_for_stereo_matching_and_optical_flow_arXiv2023.md)

---

<a id="3-semantic-priors-for-hoi-chapter-4"></a>
## 3. Semantic Priors for HOI (Chapter 4)

> Foundation models provide semantic knowledge for visual evidence localization and interaction intent reasoning.

<a id="31-prior-sources"></a>
### 3.1 Prior Sources

> `[Prior Source]` Upstream models providing visual grounding and language reasoning capabilities.

<!-- PAPER_LIST_PRIOR_SOURCES_CH4 -->

<a id="32-visual-grounding-priors"></a>
### 3.2 Visual Grounding Priors

> Open-vocabulary detection, promptable segmentation, and region association models (Grounding DINO, SAM/SAM2, CLIP, Florence-2) convert semantic prompts into visual evidence (boxes, masks, region tracks) for HOI reconstruction.

- **CHOIR** — *CHOIR: Contact-aware 4D Hand-Object Interaction Reconstruction*
  [![arXiv](https://img.shields.io/badge/arXiv-2605.20992-b31b1b.svg)](http://arxiv.org/abs/2605.20992) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/CHOIR.md)
- **AGILE** — *AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.04672-b31b1b.svg)](http://arxiv.org/abs/2602.04672) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/AGILE.md)
- **Grasp in Gaussians** — *Grasp in Gaussians: Fast Monocular Reconstruction of Dynamic Hand-Object Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2604.12929-b31b1b.svg)](http://arxiv.org/abs/2604.12929) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://aidilayce.github.io/GraG-page/) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/GraG.md)
- **GHOST** — *GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.18912-b31b1b.svg)](http://arxiv.org/abs/2603.18912) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/GHOST.md)
- **EgoGrasp** — *EgoGrasp: World-Space Hand-Object Interaction Estimation from Egocentric Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2601.01050-b31b1b.svg)](http://arxiv.org/abs/2601.01050) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/EgoGrasp.md)
- **Reconstructing Hand-Held Objects in 3D from Images and Video** — *Reconstructing Hand-Held Objects in 3D from Images and Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2404.06507-b31b1b.svg)](http://arxiv.org/abs/2404.06507) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://janehwu.github.io/mcc-ho) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/ReconstructingHandHeldObjects.md)
- **EasyHOI** — *EasyHOI: Unleashing the Power of Large Models for Reconstructing Hand-Object Interactions in the Wild*
  [![arXiv](https://img.shields.io/badge/arXiv-2411.14280-b31b1b.svg)](https://arxiv.org/abs/2411.14280) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_EasyHOI_Unleashing_the_Power_of_Large_Models_for_Reconstructing_Hand-Object_CVPR_2025_paper.html) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/lym29/EasyHOI) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/EasyHOI.md)
- **DynHOR** — *Hand-held Object Reconstruction from RGB Video with Dynamic Interaction*
  [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/HandHeldObjReconDynamic.md)
- **HandOS** — *HandOS: 3D Hand Reconstruction in One Stage*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.01537-b31b1b.svg)](https://arxiv.org/abs/2412.01537) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_HandOS_3D_Hand_Reconstruction_in_One_Stage_CVPR_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/HandOS.md)

**Prior Source Papers:**
- **Segment Anything** — *Segment Anything*
  [![arXiv](https://img.shields.io/badge/arXiv-2304.02643-b31b1b.svg)](https://arxiv.org/abs/2304.02643) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/segment_anything_arXiv2023.md)
- **Grounding DINO** — *Grounding DINO: Marrying DINO with Grounded Pre-training for Open-Set Object Detection*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://link.springer.com/10.1007/978-3-031-72970-6_3) [![DOI](https://img.shields.io/badge/DOI-10.1007/978-3-031-72-4B5D67.svg)](https://doi.org/10.1007/978-3-031-72970-6_3) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/grounding_dino_marrying_dino_with_grounded_pre_training_for_open_set_object_dete_arXiv2025.md)
- **Florence-2** — *Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks*
  [![arXiv](https://img.shields.io/badge/arXiv-2312.06148-b31b1b.svg)](https://arxiv.org/abs/2312.06148) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/florence_2_advancing_a_unified_representation_for_a_variety_of_vision_tasks_arXiv2024.md)
- **Qwen-VL** — *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond*
  [![arXiv](https://img.shields.io/badge/arXiv-2308.12966-b31b1b.svg)](https://arxiv.org/abs/2308.12966) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/QwenLM/Qwen-VL) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/qwen_vl_a_versatile_vision_language_model_for_understanding_localization_text_re_arXiv2023.md)
- **BLIP-2** — *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2301.12597-b31b1b.svg)](https://arxiv.org/abs/2301.12597) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/blip_2_bootstrapping_language_image_pre_training_with_frozen_image_encoders_and__arXiv2023.md)
- **SAM 2** — *SAM 2: Segment Anything in Images and Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2408.00714-b31b1b.svg)](https://arxiv.org/abs/2408.00714) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/facebookresearch/segment-anything-2) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/sam_2_segment_anything_in_images_and_videos_arXivunknown.md)
- **Visual Instruction Tuning** — *Visual Instruction Tuning*
  [![arXiv](https://img.shields.io/badge/arXiv-2303.14189-b31b1b.svg)](https://arxiv.org/abs/2303.14189) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/haotian-liu/LLaVA) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_2_visual_grounding/visual_instruction_tuning_arXivunknown.md)

<a id="33-language-reasoning-priors"></a>
### 3.3 Language Reasoning Priors

> LLMs/MLLMs/VLMs (GPT-4, LLaMA, Qwen) provide functional part identification, grasp/contact intent, task decomposition, and motion description knowledge. This knowledge is converted into interaction constraints for grasp and motion generation.

- **AffordGrasp** — *AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.08021-b31b1b.svg)](http://arxiv.org/abs/2603.08021) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/AffordGrasp.md)
- **SynHLMA** — *SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation*
  [![arXiv](https://img.shields.io/badge/arXiv-2510.25268-b31b1b.svg)](http://arxiv.org/abs/2510.25268) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/SynHLMA.md)
- **StructBiHOI** — *StructBiHOI: Structured Articulation Modeling for Long--Horizon Bimanual Hand--Object Interaction Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.08390-b31b1b.svg)](http://arxiv.org/abs/2603.08390) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/StructBiHOI.md)
- **OpenHOI** — *OpenHOI: Open-World Hand-Object Interaction Synthesis with Multimodal Large Language Model*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.18947-b31b1b.svg)](http://arxiv.org/abs/2505.18947) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/OpenHOI.md)
- **TOUCH** — *TOUCH: Text-guided Controllable Generation of Free-Form Hand-Object Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2510.14874-b31b1b.svg)](http://arxiv.org/abs/2510.14874) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/TOUCH.md)
- **SemGrasp** — *SemGrasp : Semantic Grasp Generation via Language Aligned Discretization*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://link.springer.com/10.1007/978-3-031-72627-9_7) [![DOI](https://img.shields.io/badge/DOI-10.1007/978-3-031-72-4B5D67.svg)](https://doi.org/10.1007/978-3-031-72627-9_7) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/SemGrasp.md)
- **DiffH2O** — *DiffH2O: Diffusion-Based Synthesis of Hand-Object Interactions from Textual Descriptions*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.17827-b31b1b.svg)](http://arxiv.org/abs/2403.17827) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://diffh2o.github.io/) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/DiffH2O.md)
- **NL2Contact** — *NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model*
  [![arXiv](https://img.shields.io/badge/arXiv-2407.12727-b31b1b.svg)](http://arxiv.org/abs/2407.12727) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/NL2Contact.md)
- **Text2Grasp** — *Text2Grasp: Grasp synthesis by text prompts of object grasping parts*
  [![arXiv](https://img.shields.io/badge/arXiv-2404.15189-b31b1b.svg)](http://arxiv.org/abs/2404.15189) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/Text2Grasp.md)
- **Text2HOI** — *Text2HOI: Text-guided 3D Motion Generation for Hand-Object Interaction*
  [![arXiv](https://img.shields.io/badge/arXiv-2404.00562-b31b1b.svg)](http://arxiv.org/abs/2404.00562) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/Text2HOI.md)
- **Multi-GraspLLM** — *Multi-GraspLLM: A Multimodal LLM for Multi-Hand Semantic Guided Grasp Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.08468-b31b1b.svg)](https://arxiv.org/abs/2412.08468) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/MultiGraspLLM.md)
- **RAGG** — *RAGG: Retrieval-Augmented Grasp Generation Model*
  [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/RAGG.md)
- **G-DexGrasp** — *G-DexGrasp: Generalizable Dexterous Grasping Synthesis Via Part-Aware Prior Retrieval and Prior-Assisted Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.19457-b31b1b.svg)](https://arxiv.org/abs/2503.19457) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/GDexGrasp.md)
- **HOIGPT** — *HOIGPT: Learning Long-Sequence Hand-Object Interaction with Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.19157-b31b1b.svg)](https://arxiv.org/abs/2503.19157) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/Mingzhen-Huang/HOIGPT) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/HOIGPT.md)
- **MEgoHand** — *MEgoHand: Multimodal Egocentric Hand-Object Interaction Motion Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.16602-b31b1b.svg)](https://arxiv.org/abs/2505.16602) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/MEgoHand.md)

**Prior Source Papers:**
- **GPT-4** — *GPT-4 Technical Report*
  [![arXiv](https://img.shields.io/badge/arXiv-2303.08774-b31b1b.svg)](https://arxiv.org/abs/2303.08774) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/gpt_4_technical_report_arXiv2023.md)
- **LLaMA** — *LLaMA: Open and Efficient Foundation Language Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2302.13971-b31b1b.svg)](https://arxiv.org/abs/2302.13971) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/llama_open_and_efficient_foundation_language_models_arXiv2023.md)
- **Qwen** — *Qwen Technical Report*
  [![arXiv](https://img.shields.io/badge/arXiv-2309.16609-b31b1b.svg)](https://arxiv.org/abs/2309.16609) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/qwen_technical_report_arXiv2023.md)
- **CLIP** — *Learning Transferable Visual Models From Natural Language Supervision*
  [![arXiv](https://img.shields.io/badge/arXiv-2103.00020-b31b1b.svg)](https://arxiv.org/abs/2103.00020) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/learning_transferable_visual_models_from_natural_language_supervision_arXiv2021.md)
- **Qwen-VL** — *Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond*
  [![arXiv](https://img.shields.io/badge/arXiv-2308.12966-b31b1b.svg)](https://arxiv.org/abs/2308.12966) [📝 Paper Summary](papers_summaries/chapter4_semantic_priors/4_3_language_reasoning/qwen_vl_a_versatile_vision_language_model_for_understanding_localization_text_re_arXiv2023.md)

---

<a id="4-visual-and-motion-generative-priors-for-hoi-chapter-5"></a>
## 4. Visual Generative Priors for HOI (Chapter 5)

> Foundation generative models provide image, video, and motion distribution priors to mitigate visual realism, temporal consistency, and motion diversity challenges.

<a id="41-prior-sources"></a>
### 4.1 Prior Sources

> `[Prior Source]` Upstream generative foundation models.

<!-- PAPER_LIST_PRIOR_SOURCES_CH5 -->

<a id="42-image-generative-priors"></a>
### 4.2 Image Generative Priors

> Text/image-conditioned diffusion models (Stable Diffusion, SDXL, FLUX, ControlNet) provide single-frame visual distribution knowledge for HOI image synthesis, editing, and data augmentation.

- **Affordance Diffusion** — *Affordance Diffusion: Synthesizing Hand-Object Interactions*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10204191/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52729.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52729.2023.02153) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/AffordanceDiffusion.md)
- **RHanDS** — *RHanDS: Refining Malformed Hands for Generated Images with Decoupled Structure and Style Guidance*
  [![arXiv](https://img.shields.io/badge/arXiv-2404.13984-b31b1b.svg)](http://arxiv.org/abs/2404.13984) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/RHanDS.md)
- **HandBooster** — *HandBooster: Boosting 3D Hand-Mesh Reconstruction by Conditional Synthesis and Sampling of Hand-Object Interactions*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/10656712/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR52733.20-4B5D67.svg)](https://doi.org/10.1109/CVPR52733.2024.00968) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/HandBooster.md)
- **Hand1000** — *Hand1000: Generating Realistic Hands from Text with Only 1,000 Images*
  [![arXiv](https://img.shields.io/badge/arXiv-2408.15461-b31b1b.svg)](http://arxiv.org/abs/2408.15461) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/Hand1000.md)
- **AttentionHand** — *AttentionHand: Text-driven Controllable Hand Image Generation for 3D Hand Reconstruction in the Wild*
  [![arXiv](https://img.shields.io/badge/arXiv-2407.18034-b31b1b.svg)](http://arxiv.org/abs/2407.18034) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/AttentionHand.md)
- **Prompt-Propose-Verify** — *Prompt-Propose-Verify: A Reliable Hand-Object-Interaction Data Generation Framework using Foundational Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2312.15247-b31b1b.svg)](http://arxiv.org/abs/2312.15247) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/PromptProposeVerify.md)
- **HOIDiffusion** — *HOIDiffusion: Generating Realistic 3D Hand-Object Interaction Data*
  [![arXiv](https://img.shields.io/badge/arXiv-2403.12011-b31b1b.svg)](https://arxiv.org/abs/2403.12011) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_HOIDiffusion_Generating_Realistic_3D_Hand-Object_Interaction_Data_CVPR_2024_paper.html) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/HOIDiffusion.md)
- **HO123** — *Single-view Image to Novel-view Generation for Hand-Object Interactions*
  [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/HO123.md)

**Prior Source Papers:**
- **FLUX.1 Kontext** — *FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space*
  [![arXiv](https://img.shields.io/badge/arXiv-2506.15742-b31b1b.svg)](https://arxiv.org/abs/2506.15742) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/flux1_kontext_flow_matching_for_in_context_image_generation_and_editing_in_laten_arXiv2025.md)
- **SDXL** — *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*
  [![arXiv](https://img.shields.io/badge/arXiv-2307.01952-b31b1b.svg)](https://arxiv.org/abs/2307.01952) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/sdxl_improving_latent_diffusion_models_for_high_resolution_image_synthesis_arXiv2024.md)
- **PixArt-α** — *PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis*
  [![arXiv](https://img.shields.io/badge/arXiv-2310.00426-b31b1b.svg)](https://arxiv.org/abs/2310.00426) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/pixart_fast_training_of_diffusion_transformer_for_photorealistic_text_to_image_s_arXiv2023.md)
- **LAION-5B** — *LAION-5B: An open large-scale dataset for training next generation image-text models*
  [![arXiv](https://img.shields.io/badge/arXiv-2210.08272-b31b1b.svg)](https://arxiv.org/abs/2210.08272) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/laion_5b_an_open_large_scale_dataset_for_training_next_generation_image_text_mod_arXiv2022.md)
- **Imagen** — *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding*
  [![arXiv](https://img.shields.io/badge/arXiv-2205.11487-b31b1b.svg)](https://arxiv.org/abs/2205.11487) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/photorealistic_text_to_image_diffusion_models_with_deep_language_understanding_arXivunknown.md)
- **ControlNet** — *Adding Conditional Control to Text-to-Image Diffusion Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2302.05543-b31b1b.svg)](https://arxiv.org/abs/2302.05543) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/lllyasviel/ControlNet) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/adding_conditional_control_to_text_to_image_diffusion_models_arXiv2023.md)
- **LDM / Stable Diffusion** — *High-Resolution Image Synthesis with Latent Diffusion Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2112.10752-b31b1b.svg)](https://arxiv.org/abs/2112.10752) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_2_image_generative/high_resolution_image_synthesis_with_latent_diffusion_models_arXiv2022.md)

<a id="43-video-generative-priors"></a>
### 4.3 Video Generative Priors

> Video diffusion models (SVD, AnimateDiff, CogVideoX, HunyuanVideo) provide temporal appearance and identity persistence priors for HOI video generation, inpainting, and reenactment.

- **PAM** — *PAM: A Pose-Appearance-Motion Engine for Sim-to-Real HOI Video Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.22193-b31b1b.svg)](http://arxiv.org/abs/2603.22193) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/GasaiYU/PAM) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/PAM.md)
- **HVG-3D** — *HVG-3D: Bridging Real and Simulation Domains for 3D-Conditional Hand-Object Interaction Video Synthesis*
  [![arXiv](https://img.shields.io/badge/arXiv-2604.03305-b31b1b.svg)](http://arxiv.org/abs/2604.03305) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://hvg3d.github.io/) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/HVG3D.md)
- **ByteLoom** — *ByteLoom: Weaving Geometry-Consistent Human-Object Interactions through Progressive Curriculum Learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.22854-b31b1b.svg)](http://arxiv.org/abs/2512.22854) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/ByteLoom.md)
- **Open-world HOI VideoGen** — *Open-world Hand-Object Interaction Video Generation Based on Structure and Contact-aware Representation*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.01677-b31b1b.svg)](http://arxiv.org/abs/2512.01677) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/OpenWorldHOIVideo.md)
- **AnchorCrafter** — *AnchorCrafter: Animate Cyber-Anchors Selling Your Products via Human-Object Interacting Video Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2411.17383-b31b1b.svg)](http://arxiv.org/abs/2411.17383) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/AnchorCrafter.md)
- **iDiT-HOI** — *iDiT-HOI: Inpainting-based Hand Object Interaction Reenactment via Video Diffusion Transformer*
  [![arXiv](https://img.shields.io/badge/arXiv-2506.12847-b31b1b.svg)](http://arxiv.org/abs/2506.12847) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/iDiTHOI.md)
- **SViMo** — *SViMo: Synchronized Diffusion for Video and Motion Generation in Hand-object Interaction Scenarios*
  [![arXiv](https://img.shields.io/badge/arXiv-2506.02444-b31b1b.svg)](http://arxiv.org/abs/2506.02444) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/SViMo.md)
- **ManiVideo** — *ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.16212-b31b1b.svg)](http://arxiv.org/abs/2412.16212) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/ManiVideo.md)
- **HOI-Swap** — *HOI-Swap: Swapping Objects in Videos with Hand-Object Interaction Awareness*
  [![arXiv](https://img.shields.io/badge/arXiv-2406.07754-b31b1b.svg)](http://arxiv.org/abs/2406.07754) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/HOISwap.md)
- **Re-HOLD** — *Re-HOLD: Video Hand Object Interaction Reenactment via adaptive Layout-instructed Diffusion Model*
  [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/ReHOLD.md)

**Prior Source Papers:**
- **HunyuanVideo** — *HunyuanVideo: A Systematic Framework For Large Video Generative Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.03603-b31b1b.svg)](https://arxiv.org/abs/2412.03603) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/hunyuanvideo_a_systematic_framework_for_large_video_generative_models_arXiv2025.md)
- **CogVideoX** — *CogVideoX: Text-to-Video Diffusion Model*
  [![arXiv](https://img.shields.io/badge/arXiv-2408.06072-b31b1b.svg)](https://arxiv.org/abs/2408.06072) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/cogvideox_text_to_video_diffusion_model_arXiv2025.md)
- **Open-Sora Plan** — *Open-Sora Plan: Open-Source Large Video Generation Model*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.00131-b31b1b.svg)](https://arxiv.org/abs/2412.00131) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/open_sora_plan_open_source_large_video_generation_model_arXiv2024.md)
- **AnimateDiff** — *AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning*
  [![arXiv](https://img.shields.io/badge/arXiv-2307.04785-b31b1b.svg)](https://arxiv.org/abs/2307.04785) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/animatediff_animate_your_personalized_text_to_image_diffusion_models_without_spe_arXiv2024.md)
- **Stable Video Diffusion** — *Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets*
  [![arXiv](https://img.shields.io/badge/arXiv-2311.15127-b31b1b.svg)](https://arxiv.org/abs/2311.15127) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/stable_video_diffusion_scaling_latent_video_diffusion_models_to_large_datasets_arXiv2023.md)
- **Frozen in Time** — *Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval*
  [![arXiv](https://img.shields.io/badge/arXiv-2104.00650-b31b1b.svg)](https://arxiv.org/abs/2104.00650) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/frozen_in_time_a_joint_video_and_image_encoder_for_end_to_end_retrieval_arXiv2021.md)

<a id="431-from-hoi-video-generation-to-hoi-world-models"></a>
#### 4.3.1 From HOI Video Generation to HOI World Models

> World-model methods that use video prediction or interaction dynamics to model future hand-object states. Include only papers with an explicit HOI world-model role.

<!-- PAPER_LIST_HOI_WORLD_MODELS -->

- **Dexterous World Models** — *Dexterous World Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.17907-b31b1b.svg)](https://arxiv.org/abs/2512.17907) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/Dexterous_World_Models_arXiv2025.md)
- **Hand2World** — *Hand2World: Autoregressive Egocentric Interaction Generation via Free-Space Hand Gestures*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.09600-b31b1b.svg)](https://arxiv.org/abs/2602.09600) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/Hand2World_arXiv2026.md)
- **Generated Reality** — *Generated Reality: Human-centric World Simulation using Interactive Video Generation with Hand and Camera Control*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.18422-b31b1b.svg)](https://arxiv.org/abs/2602.18422) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/Generated_Reality_arXiv2026.md)
- **Egocentric World Model for Photorealistic Hand-Object Interaction Synthesis** — *Egocentric World Model for Photorealistic Hand-Object Interaction Synthesis*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.13615-b31b1b.svg)](https://arxiv.org/abs/2603.13615) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/Egocentric_World_Model_arXiv2026.md)
- **Wh0** — *Wh0: Generative World Models as Scalable Sources of Egocentric Human Hand Manipulation Data*
  [![arXiv](https://img.shields.io/badge/arXiv-2606.22136-b31b1b.svg)](https://arxiv.org/abs/2606.22136) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/Wh0_arXiv2026.md)
- **HandsOnWorld** — *HandsOnWorld: Unconstrained Egocentric Video Generation with Camera-Disentangled Hand Control*
  [![arXiv](https://img.shields.io/badge/arXiv-2607.02075-b31b1b.svg)](https://arxiv.org/abs/2607.02075) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_3_video_generative/HandsOnWorld_arXiv2026.md)

<a id="44-visual-representation-priors"></a>
### 4.4 Visual Representation Priors

> Pre-trained visual representations used to improve HOI reconstruction, grasping, or interaction understanding. Papers are listed here when representation transfer is central to their HOI use.

<!-- PAPER_LIST_VISUAL_REPRESENTATION_PRIORS -->

- **HopFormer** — *HopFormer: Sparse Graph Transformers with Explicit Receptive Field Control*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.02268-b31b1b.svg)](https://arxiv.org/abs/2602.02268) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_5_visual_representation_priors/HopFormer_arXiv2026.md)
- **HORT** — *HORT: Monocular Hand-held Objects Reconstruction with Transformers*
  [![Paper](https://img.shields.io/badge/Paper-IEEE-4B5D67.svg)](https://doi.org/10.1109/ICCV51701.2025.00571) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_5_visual_representation_priors/HORT_ICCV2025.md)
- **ForeHOI** — *ForeHOI: Feed-forward 3D Object Reconstruction from Daily Hand-Object Interaction Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.06226-b31b1b.svg)](https://arxiv.org/abs/2602.06226) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_5_visual_representation_priors/ForeHOI_arXiv2026.md)
- **Reconstructing Hand-Held Objects in 3D from Images and Videos** — *Reconstructing Hand-Held Objects in 3D from Images and Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2404.06507-b31b1b.svg)](https://arxiv.org/abs/2404.06507) [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_3_shape_retrieval/Reconstructing_Hand_Held_Objects_in_3D_from_arXiv2025.md)
- **Hand-held Object Reconstruction from RGB Video with Dynamic Interaction** — *Hand-held Object Reconstruction from RGB Video with Dynamic Interaction*
  [📝 Paper Summary](papers_summaries/chapter3_3d_geometry_priors/3_2_shape_completion/Hand_held_Object_Reconstruction_from_RGB_Video_arXiv2025.md)
- **HUG** — *Human Universal Grasping*
  [![arXiv](https://img.shields.io/badge/arXiv-2606.17054-b31b1b.svg)](https://arxiv.org/abs/2606.17054) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_5_visual_representation_priors/HUG_arXiv2026.md)

**Prior Source Papers:**
- **DINOv2** — *DINOv2: Learning Robust Visual Features without Supervision*
  [![arXiv](https://img.shields.io/badge/arXiv-2304.07193-b31b1b.svg)](https://arxiv.org/abs/2304.07193) [📝 Paper Summary](papers_summaries/chapter5_visual_motion_generative_priors/5_5_visual_representation_priors/DINOv2_arXiv2024.md)

---

<a id="5-hoi-for-robot-learning-chapter-6"></a>
## 5. HOI for Robot Learning (Chapter 6)

> Visual HOI reconstruction/generation results serve as privileged information for robot policy learning. This chapter validates whether HOI outputs are physically executable.

<a id="51-generalist-policy-learning-video-based-pretraining"></a>
### 5.1 Generalist Policy Learning: Video-Based Pretraining

> Learning reusable policies from web-scale or egocentric human interaction videos through latent actions, image-goal representations, or video dynamics — without explicit hand/object structure supervision.

- **CLAP** — *CLAP: contrastive latent action pretraining for learning vision-language-action models from human videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2601.04061-b31b1b.svg)](https://arxiv.org/abs/2601.04061) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/CLAP_arXiv2026.md)
- **mimic-video** — *mimic-video: video-action models for generalizable robot control beyond VLAs*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.15692-b31b1b.svg)](http://arxiv.org/abs/2512.15692) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/mimic_video_arXiv2025.md)
- **UniVLA** — *UniVLA: learning to act anywhere with task-centric latent actions*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.06111-b31b1b.svg)](http://arxiv.org/abs/2505.06111) [![GitHub](https://img.shields.io/badge/GitHub-code-181717.svg?logo=github)](https://github.com/OpenDriveLab/UniVLA) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/UniVLA_arXiv2025.md)
- **villa-X** — *villa-X: enhancing latent action modeling in vision-language-action models*
  [![arXiv](https://img.shields.io/badge/arXiv-2507.23682-b31b1b.svg)](http://arxiv.org/abs/2507.23682) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/villa_X_arXiv2025.md)
- **FLARE** — *FLARE: robot learning with implicit world modeling*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.15659-b31b1b.svg)](http://arxiv.org/abs/2505.15659) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/FLARE_arXiv2025.md)
- **Latent action pretraining from videos** — *Latent action pretraining from videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2410.11758-b31b1b.svg)](http://arxiv.org/abs/2410.11758) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://latentactionpretraining.github.io) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/Latent_action_pretraining_from_videos_arXiv2025.md)
- **Video prediction policy** — *Video prediction policy: a generalist robot policy with predictive visual representations*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.14803-b31b1b.svg)](http://arxiv.org/abs/2412.14803) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/Video_prediction_policy_arXiv2025.md)
- **GR00T N1** — *GR00T N1: an open foundation model for generalist humanoid robots*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.14734-b31b1b.svg)](http://arxiv.org/abs/2503.14734) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/GR00T_N1_arXiv2025.md)
- **ViPRA** — *ViPRA: video prediction for robot actions*
  [![arXiv](https://img.shields.io/badge/arXiv-2511.07732-b31b1b.svg)](https://arxiv.org/abs/2511.07732) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://vipra-project.github.io) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/ViPRA_arXiv2025.md)
- **Motus** — *Motus: a unified latent action world model*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.13030-b31b1b.svg)](https://arxiv.org/abs/2512.13030) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/Motus_arXiv2025.md)
- **IGOR** — *IGOR: image-GOal representations are the atomic control units for foundation models in embodied AI*
  [![arXiv](https://img.shields.io/badge/arXiv-2411.00785-b31b1b.svg)](http://arxiv.org/abs/2411.00785) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/IGOR_arXiv2024.md)
- **GR-2** — *GR-2: a generative video-language-action model with web-scale knowledge for robot manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2410.06158-b31b1b.svg)](http://arxiv.org/abs/2410.06158) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/GR_2_arXiv2024.md)
- **Unleashing Video Gen Pretraining** — *Unleashing large-scale video generative pre-training for visual robot manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2312.13139-b31b1b.svg)](http://arxiv.org/abs/2312.13139) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/Unleashing_large_scale_video_generative_pre_training_for_arXiv2023.md)
- **Moto** — *Moto: latent motion token as the bridging language for learning robot manipulation from videos*
  [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Moto_Latent_Motion_Token_as_the_Bridging_Language_for_Learning_ICCV_2025_paper.html) [![DOI](https://img.shields.io/badge/DOI-10.1109/ICCV51701.20-4B5D67.svg)](https://doi.org/10.1109/ICCV51701.2025.01837) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_1_video_based_pretraining/Moto_arXiv.md)

<a id="52-generalist-policy-learning-structured-hoi-supervision"></a>
### 5.2 Generalist Policy Learning: Structured HOI Supervision

> Enhancing generalist VLA/policy pretraining with structured HOI signals: hand pose, object interaction, motion tokens, or frame-aligned action chunks.

- **UniHM** — *UniHM: unified dexterous hand manipulation with vision language model*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.00732-b31b1b.svg)](http://arxiv.org/abs/2603.00732) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/UniHM_arXiv2026.md)
- **Ego-Pi** — *Ego-Pi: VLA Fine-Tuning for Ego-Centric Human and Robot Data*
  [![arXiv](https://img.shields.io/badge/arXiv-2606.08107-b31b1b.svg)](https://arxiv.org/abs/2606.08107) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://egopipaper.github.io/) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/Ego_Pi_arXiv2026.md)
- **In-N-on** — *In-N-on: scaling egocentric manipulation with in-the-wild and on-task data*
  [![arXiv](https://img.shields.io/badge/arXiv-2511.15704-b31b1b.svg)](http://arxiv.org/abs/2511.15704) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/In_N_on_arXiv2025.md)
- **Scalable VLA Pretraining** — *Scalable vision-language-action model pretraining for robotic manipulation with real-life human activity videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2510.21571-b31b1b.svg)](http://arxiv.org/abs/2510.21571) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/Scalable_vision_language_action_model_pretraining_for_roboti_arXiv2025.md)
- **MotionTrans** — *MotionTrans: human VR data enable motion-level learning for robotic manipulation policies*
  [![arXiv](https://img.shields.io/badge/arXiv-2509.17759-b31b1b.svg)](http://arxiv.org/abs/2509.17759) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/MotionTrans_arXiv2025.md)
- **H-RDT** — *H-RDT: human manipulation enhanced bimanual robotic manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2507.23523-b31b1b.svg)](http://arxiv.org/abs/2507.23523) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/H_RDT_arXiv2025.md)
- **Being-H0** — *Being-H0: vision-language-action pretraining from large-scale human videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2507.15597-b31b1b.svg)](http://arxiv.org/abs/2507.15597) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/Being_H0_arXiv2025.md)
- **EgoVLA** — *EgoVLA: learning vision-language-action models from egocentric human videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2507.12440-b31b1b.svg)](http://arxiv.org/abs/2507.12440) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/EgoVLA_arXiv2025.md)
- **Gemini robotics** — *Gemini robotics: bringing AI into the physical world*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.20020-b31b1b.svg)](http://arxiv.org/abs/2503.20020) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_2_2_structured_hoi_supervision/Gemini_robotics_arXiv2025.md)

<a id="53-task-specific-skill-transfer-dexterous-motion-retargeting"></a>
### 5.3 Task-Specific Skill Transfer: Dexterous Motion Retargeting

> Explicitly converting human hand-object demonstrations to robotized demonstrations, robot action trajectories, or cross-embodiment rollouts.

- **DexImit** — *DexImit: learning bimanual dexterous manipulation from monocular human videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.10105-b31b1b.svg)](http://arxiv.org/abs/2602.10105) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/DexImit_arXiv2026.md)
- **RoboWheel** — *RoboWheel: a data engine from real-world human demonstrations for cross-embodiment robotic learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.02729-b31b1b.svg)](http://arxiv.org/abs/2512.02729) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/RoboWheel_arXiv2025.md)
- **DexUMI** — *DexUMI: using human hand as the universal manipulation interface for dexterous manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.21864-b31b1b.svg)](http://arxiv.org/abs/2505.21864) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/DexUMI_arXiv2025.md)
- **HERMES** — *HERMES: human-to-robot embodied learning from multi-source motion data for mobile dexterous manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2508.20085-b31b1b.svg)](http://arxiv.org/abs/2508.20085) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/HERMES_arXiv2025.md)
- **Masquerade** — *Masquerade: Learning from In-the-wild Human Videos using Data-Editing*
  [![arXiv](https://img.shields.io/badge/arXiv-2508.09976-b31b1b.svg)](http://arxiv.org/abs/2508.09976) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/Masquerade_arXiv2025.md)
- **DexMachina** — *DexMachina: functional retargeting for bimanual dexterous manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.24853-b31b1b.svg)](http://arxiv.org/abs/2505.24853) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/DexMachina_arXiv2025.md)
- **Web2Grasp** — *Web2Grasp: Learning Functional Grasps from Web Images of Hand-Object Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.05517-b31b1b.svg)](http://arxiv.org/abs/2505.05517) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/Web2Grasp_arXiv2025.md)
- **You Only Teach Once** — *You Only Teach Once: Learn One-Shot Bimanual Robotic Manipulation from Video Demonstrations*
  [![arXiv](https://img.shields.io/badge/arXiv-2501.14208-b31b1b.svg)](http://arxiv.org/abs/2501.14208) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/You_Only_Teach_Once_arXiv2025.md)
- **ManipTrans** — *ManipTrans: efficient dexterous bimanual manipulation transfer via residual learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.21860-b31b1b.svg)](http://arxiv.org/abs/2503.21860) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/ManipTrans_arXiv2025.md)
- **DexMV** — *DexMV: imitation learning for dexterous manipulation from human videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2108.05877-b31b1b.svg)](http://arxiv.org/abs/2108.05877) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/DexMV_arXiv2022.md)

<a id="54-task-specific-skill-transfer-interaction-guided-policy-learning"></a>
### 5.4 Task-Specific Skill Transfer: Interaction-Guided Policy Learning

> Using visual affordance, contact, pixel/3D interaction trajectory, or generated interaction plans to guide robot grasping and policy execution — without per-frame human-to-robot motion mapping.

- **FlowHOI** — *FlowHOI: Flow-based Semantics-Grounded Generation of Hand-Object Interactions for Dexterous Robot Manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2602.13444-b31b1b.svg)](http://arxiv.org/abs/2602.13444) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_2_interaction_guided_policy/FlowHOI_arXiv2026.md)
- **A0** — *A0: an affordance-aware hierarchical model for general robotic manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2504.12636-b31b1b.svg)](http://arxiv.org/abs/2504.12636) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_2_interaction_guided_policy/A0_arXiv2026.md)
- **GAT-grasp** — *GAT-grasp: gesture-driven affordance transfer for task-aware robotic grasping*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.06227-b31b1b.svg)](http://arxiv.org/abs/2503.06227) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_2_interaction_guided_policy/GAT_grasp_arXiv2025.md)
- **Gen2Act** — *Gen2Act: Human Video Generation in Novel Scenarios enables Generalizable Robot Manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2409.16283-b31b1b.svg)](http://arxiv.org/abs/2409.16283) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_2_interaction_guided_policy/Gen2Act_arXiv2024.md)
- **Any-point trajectory modeling for policy learning** — *Any-point trajectory modeling for policy learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2401.00025-b31b1b.svg)](http://arxiv.org/abs/2401.00025) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_2_interaction_guided_policy/Any_point_trajectory_modeling_for_policy_learning_arXiv2024.md)
- **VidBot** — *VidBot: Learning Generalizable 3D Actions from In-the-Wild 2D Human Videos for Zero-Shot Robotic Manipulation*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.07135-b31b1b.svg)](https://arxiv.org/abs/2503.07135) [![Paper](https://img.shields.io/badge/Paper-CVF-4B5D67.svg)](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_VidBot_Learning_Generalizable_3D_Actions_from_In-the-Wild_2D_Human_Videos_CVPR_2025_paper.html) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_2_interaction_guided_policy/VidBot_arXiv.md)
- **AffordDexGrasp** — *AffordDexGrasp: Open-set Language-guided Dexterous Grasp with Generalizable-Instructive Affordance*
  [![arXiv](https://img.shields.io/badge/arXiv-2503.07360-b31b1b.svg)](https://arxiv.org/abs/2503.07360) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_2_interaction_guided_policy/AffordDexGrasp_arXiv.md)

<a id="55-hoi-to-robot-data-engines"></a>
### 5.5 HOI-to-Robot Data Engines

> Data engines that turn human hand-object interaction observations into scalable robot-learning demonstrations, aligned trajectories, or executable supervision.

<!-- PAPER_LIST_HOI_TO_ROBOT_DATA_ENGINES -->

- **Human2Robot** — *Human2Robot: Learning Robot Actions from Paired Human-Robot Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2502.16587-b31b1b.svg)](https://arxiv.org/abs/2502.16587) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_4_hoi_to_robot_data_engines/Human2Robot_arXiv2025.md)
- **TraceGen** — *TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2511.21690-b31b1b.svg)](https://arxiv.org/abs/2511.21690) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_4_hoi_to_robot_data_engines/TraceGen_arXiv2025.md)
- **RoboWheel** — *RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.02729-b31b1b.svg)](https://arxiv.org/abs/2512.02729) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_3_1_dexterous_motion_retargeting/RoboWheel_arXiv2025.md)
- **H2R-Grounder** — *H2R-Grounder: A Paired-Data-Free Paradigm for Translating Human Interaction Videos into Physically Grounded Robot Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2512.09406-b31b1b.svg)](https://arxiv.org/abs/2512.09406) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_4_hoi_to_robot_data_engines/H2R_Grounder_arXiv2025.md)
- **EgoEngine** — *EgoEngine: From Egocentric Human Videos to High-Fidelity Dexterous Robot Demonstrations*
  [![arXiv](https://img.shields.io/badge/arXiv-2606.12604-b31b1b.svg)](https://arxiv.org/abs/2606.12604) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_4_hoi_to_robot_data_engines/EgoEngine_arXiv2026.md)
- **EgoInfinity** — *EgoInfinity: A Web-Scale 4D Hand-Object Interaction Data Engine for Any-View Robot Retargeting and Video-to-Action Robot Learning*
  [![arXiv](https://img.shields.io/badge/arXiv-2606.17385-b31b1b.svg)](https://arxiv.org/abs/2606.17385) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_4_hoi_to_robot_data_engines/EgoInfinity_arXiv2026.md)
- **Qwen-RobotManip** — *Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models*
  [![arXiv](https://img.shields.io/badge/arXiv-2606.17846-b31b1b.svg)](https://arxiv.org/abs/2606.17846) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_4_hoi_to_robot_data_engines/Qwen_RobotManip_arXiv2026.md)
- **Do as I Do** — *Do as I Do: Dexterous Manipulation Data from Everyday Human Videos*
  [![arXiv](https://img.shields.io/badge/arXiv-2606.19333-b31b1b.svg)](https://arxiv.org/abs/2606.19333) [📝 Paper Summary](papers_summaries/chapter6_robot_learning/6_4_hoi_to_robot_data_engines/Do_as_I_Do_arXiv2026.md)

---



<a id="7-benchmarks-chapter-7"></a>
## 7. Benchmarks (Chapter 7)

> Key benchmark datasets for HOI reconstruction, generation, and interaction understanding, organized by evaluation purpose. Each dataset comes with a detailed AI summary covering data composition, annotation types, supported evaluation tasks, strengths, and limitations.

<a id="71-reconstruction-benchmarks"></a>
### 7.1 Reconstruction Benchmarks

- **FreiHAND** — *FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape From Single RGB Images*
  [![Paper](https://img.shields.io/badge/Paper-ICCV-4B5D67.svg)](https://ieeexplore.ieee.org/document/9010946/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/FreiHAND_ICCV2019.md)
- **HO3D** — *HOnnotate: A Method for 3D Annotation of Hand and Object Poses*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/9157405/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://www.tugraz.at/index.php?id=40231) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HO3D_CVPR2020.md)
- **HOnnotate** — *HOnnotate: A Method for 3D Annotation of Hand and Object Poses*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/9157405/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR42600.2020.00-4B5D67.svg)](https://doi.org/10.1109/CVPR42600.2020.00326) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HOnnotate_CVPR2020.md)
- **HO-3D v3** — *HO-3D v3: Improving the Accuracy of Hand-Object Annotations of the HO-3D Dataset*
   [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HO3Dv3_2021.md)
- **DexYCB** — *DexYCB: A Benchmark for Capturing Hand Grasping of Objects*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/9578786/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://dex-ycb.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/DexYCB_CVPR2021.md)
- **ARCTIC** — *ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/10203858/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://arctic.is.tue.mpg.de/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/ARCTIC_CVPR2023.md)
- **HOT3D** — *HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos*
  [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://facebookresearch.github.io/hot3d/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HOT3D_CVPR2025.md)
- **H2O** — *H2O: Two Hands Manipulating Objects for First Person Interaction Recognition*
  [![Paper](https://img.shields.io/badge/Paper-ICCV-4B5D67.svg)](https://ieeexplore.ieee.org/document/9710699/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://taeinkwon.com/projects/h2o/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/H2O_ICCV2021.md)
- **InterHand2.6M** — *InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image*
  [![Paper](https://img.shields.io/badge/Paper-ECCV-4B5D67.svg)](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_33) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://mks0601.github.io/InterHand2.6M/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/InterHand2_6M_ECCV2020.md)
- **AssemblyHands** — *AssemblyHands: Towards Egocentric Activity Understanding via 3D Hand Pose Estimation*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/10203338/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://assemblyhands.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/AssemblyHands_CVPR2023.md)
- **ArtHOI Datasets** — *ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.25791-b31b1b.svg)](https://arxiv.org/abs/2603.25791) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/ArtHOI_Datasets_2026.md)
- **FPHA** — *First-Person Hand Action Benchmark with RGB-D Videos and 3D Hand Pose Annotations*
  [![Paper](https://img.shields.io/badge/Paper-DOI-4B5D67.svg)](https://ieeexplore.ieee.org/document/8578148/) [![DOI](https://img.shields.io/badge/DOI-10.1109/CVPR.2018.00050-4B5D67.svg)](https://doi.org/10.1109/CVPR.2018.00050) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/FPHA_2018.md)
- **GigaHands** — *GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities*
  [![arXiv](https://img.shields.io/badge/arXiv-2412.04244-b31b1b.svg)](https://arxiv.org/abs/2412.04244) [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ivl.cs.brown.edu/research/gigahands.html) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://ivl.cs.brown.edu/research/gigahands.html) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/GigaHands_CVPR2025.md)

<a id="72-generation-benchmarks"></a>
### 7.2 Generation Benchmarks

- **GRAB** — *GRAB: A Dataset of Whole-Body Human Grasping of Objects*
  [![Paper](https://img.shields.io/badge/Paper-ECCV-4B5D67.svg)](https://link.springer.com/chapter/10.1007/978-3-030-58574-7_21) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://grab.is.tue.mpg.de/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/GRAB_ECCV2020.md)
- **OakInk** — *OakInk: A Large-scale Knowledge Repository for Understanding Hand-Object Interaction*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/9878658/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://oakink.net/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/OakInk_CVPR2022.md)
- **OakInk2** — *OakInk2: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion* [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/OakInk2_CVPR2024.md)
- **ContactPose** — *ContactPose: A Dataset of Grasps with Object Contact and Hand Pose*
  [![Paper](https://img.shields.io/badge/Paper-ECCV-4B5D67.svg)](https://link.springer.com/chapter/10.1007/978-3-030-58568-6_21) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://contactpose.cc.gatech.edu/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/ContactPose_ECCV2020.md)
- **ContactDB** — *ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/8954000/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://contactdb.cc.gatech.edu/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/ContactDB_CVPR2019.md)
- **AffordPose** — *AffordPose: A Large-scale Dataset of Hand-Object Interactions with Affordance-driven Hand Pose*
  [![Paper](https://img.shields.io/badge/Paper-ICCV-4B5D67.svg)](https://ieeexplore.ieee.org/document/10378060/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://affordpose.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/AffordPose_ICCV2023.md)
- **RH20T** — *RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot*
  [![arXiv](https://img.shields.io/badge/arXiv-2307.00595-b31b1b.svg)](https://arxiv.org/abs/2307.00595) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://rh20t.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/RH20T_2023.md)
- **HandX** — *HandX: Scaling Bimanual Motion and Interaction Generation*
  [![arXiv](https://img.shields.io/badge/arXiv-2603.28766-b31b1b.svg)](https://arxiv.org/abs/2603.28766) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://handx-project.github.io) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HandX_arXiv2026.md)

<a id="73-embodied-learning-data-sources"></a>
### 7.3 Embodied Learning Data Sources

> Large-scale video and robot-learning datasets that provide egocentric interaction data for HOI-related pretraining and transfer learning.

- **HOI4D** — *HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/9879533/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://hoi4d.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HOI4D_CVPR2022.md)
- **EgoDex** — *EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video*
  [![arXiv](https://img.shields.io/badge/arXiv-2505.11709-b31b1b.svg)](https://arxiv.org/abs/2505.11709) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/EgoDex_2026.md)
- **Ego4D** — *Ego4D: Around the World in 3,000 Hours of Egocentric Video*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/9878708/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://ego4d-data.org/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/Ego4D_CVPR2022.md)
- **Ego-Exo4D** — *Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/10656221/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://ego-exo4d-data.org/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/EgoExo4D_CVPR2024.md)
- **EPIC-KITCHENS** — *The EPIC-KITCHENS Dataset: Collection, Challenges and Baselines*
  [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://epic-kitchens.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/EPIC_KITCHENS_2020.md)
- **HoloAssist** — *HoloAssist: an Egocentric Human Interaction Dataset for Interactive AI Assistants in the Real World*
  [![Paper](https://img.shields.io/badge/Paper-ICCV-4B5D67.svg)](https://ieeexplore.ieee.org/document/10377281/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://holoassist.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HoloAssist_ICCV2023.md)
- **Assembly101** — *Assembly101: A Large-Scale Multi-View Video Dataset for Understanding Procedural Activities*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/9880358/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://assembly-101.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/Assembly101_CVPR2022.md)
- **TACO** — *TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding*
  [![Paper](https://img.shields.io/badge/Paper-CVPR-4B5D67.svg)](https://ieeexplore.ieee.org/document/10655531/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://taco2024.github.io/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/TACO_CVPR2024.md)
- **Something-Something** — *The "Something Something" Video Database for Learning and Evaluating Visual Common Sense*
  [![Paper](https://img.shields.io/badge/Paper-ICCV-4B5D67.svg)](https://ieeexplore.ieee.org/document/8237509/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://developer.qualcomm.com/software/ai-datasets/something-something) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/SomethingSomething_ICCV2017.md)
- **HowTo100M** — *HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips*
  [![Paper](https://img.shields.io/badge/Paper-ICCV-4B5D67.svg)](https://ieeexplore.ieee.org/document/9010066/) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://www.di.ens.fr/willow/research/howto100m/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HowTo100M_ICCV2019.md)
- **HumanNet** — *HumanNet: Scaling Human-centric Video Learning to One Million Hours*
  [![arXiv](https://img.shields.io/badge/arXiv-2605.06747-b31b1b.svg)](https://arxiv.org/abs/2605.06747) [![Website](https://img.shields.io/badge/Website-page-0A66C2.svg)](https://dagroup-pku.github.io/HumanNet/) [📝 Paper Summary](papers_summaries/chapter7_datasets_metrics/datasets/HumanNet_arXiv2026.md)


## Citations

If you find this repository useful, please consider citing the original papers and our survey:

```bibtex
@misc{hoi-survey-2026,
  title = {Hand-Object Interaction in the Age of Large Foundation Models: Reconstruction, Generation, and Embodied Transfer – A Survey},
  year = {2026},
}
```
