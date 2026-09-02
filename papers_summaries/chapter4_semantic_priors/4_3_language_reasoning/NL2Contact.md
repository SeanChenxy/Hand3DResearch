# NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model

**Authors:** Zhongqun Zhang, Hengfei Wang, Ziwei Yu, Yihua Cheng, Angela Yao, Hyung Jin Chang  
**Date:** 2024-07-17  
**Identifier:** [arXiv:2407.12727](https://arxiv.org/abs/2407.12727)  
**Zotero item:** `L32WWDAD` ([Zotero](zotero://select/library/items/L32WWDAD))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
NL2Contact introduces the task of controlling 3D hand-object contact modeling with natural language. It pairs a new dataset, ContactDescribe (LLM-generated, multi-level contact descriptions), with a two-stage latent diffusion model that first generates a text-consistent hand pose and then a hand-object contact map, and applies the predicted contact to grasp pose optimization and novel grasp generation, outperforming geometry-only baselines such as ContactOpt and S2Contact.

## Background and Problem
Physical hand-object contact is central to refining inaccurate hand poses and generating human grasps in 3D hand-object reconstruction. Existing contact modeling methods rely on geometric constraints (e.g., PointNet-based inference from hand-object point clouds, as in ContactOpt, GraspTTA, Contact2Grasp) that are physically plausible but cannot be specified or controlled, often producing unrealistic patterns such as all five fingers contacting scissors. Prior controllable approaches use high-level intent verbs or coarse object-part affordances, which are too vague to specify precise contact patterns. The paper defines a new task: given a natural language description of a hand-object interaction, generate physically realistic contact maps aligned with the text. Two challenges are identified: cross-modal modeling from language to 3D contact, and the absence of datasets with descriptive text for contact patterns.

## Method
The method has three components. (1) Text-to-Hand-Object fusion: a BERT text encoder, VPoser hand-pose encoder, and PointNet++ object encoders produce global and local features that are fused via two cascaded multi-head attention modules, switching between grasp pose optimization and grasp generation modes. (2) Staged latent diffusion: stage 1 denoises a VPoser latent hand pose conditioned on text and object features (U-Net denoiser); stage 2 freezes this model and denoises a contact latent (pre-trained PointNet contact encoder) conditioned on the text and the generated hand pose rendered onto the hand-object point cloud. (3) Contact optimization: the generated contact map refines the MANO hand parameters by minimizing the discrepancy between current and generated contact maps plus a penetration loss, following ContactOpt.

## Contributions
- A new task formulation: natural language-guided 3D hand-object contact modeling, the first work to model 3D hand-object contacts from text descriptions.
- ContactDescribe, the first dataset with sentence-level hand-centered contact descriptions: 2,300 unique grasps of 25 household objects from 50 participants, 11,500 descriptions (five sentences per grasp), built on ContactPose with multi-level (action / grasp type / contact location / free-finger status) prompts authored manually and diversified with ChatGPT.
- A staged latent diffusion model with a text-to-hand-object fusion network that guides both hand pose and contact map denoising.
- Demonstrated applications to grasp pose optimization and novel human grasp generation from a textual contact description.

## Experimental Setup
Training uses ContactDescribe (the Perturbed ContactPose split: 22,624 training and 1,416 testing grasps with ~80 mm perturbed hand error) and a manually annotated Described HO3D subset (~14k grasps, 10 subjects, 10 YCB objects) to test generalization to unseen objects. Object meshes are sampled to 2,048 points. Training uses Adam, batch size 64, learning rate 1.5e-3, 50 epochs for hand pose denoising and 100 epochs for contact generation, taking about 11 hours on a single V100 GPU (9.5 GB). Generating one contact map takes around 3 seconds. Metrics: MPJPE (mm) over 21 joints, intersection volume (voxel size 0.5 cm), contact coverage (points within ±2 mm of the object surface), contact precision/recall (threshold 0.4), grasp diversity (K-means with 20 clusters), and simulation displacement (SD) for physical stability.

## Results
On ContactDescribe grasp pose optimization, NL2Contact reaches 21.7 mm MPJPE versus 25.1 mm for ContactOpt and 29.4 mm for S2Contact, while also lowering intersection volume to 7.1 (vs 12.8 and 12.2) and achieving 30.5% coverage, 49.2% contact precision, and 59.9% recall. On Described HO3D it attains 8.4 mm MPJPE, best among compared methods (ContactOpt 9.5, TOCH 9.3, S2Contact 8.7). For grasp generation on unseen HO3D objects, it achieves the lowest intersection volume (5.89 vs 9.96 for ContactGen), 99% coverage, highest diversity (5.91), and competitive stability (SD 2.31). Ablations show low-level text is most precise (21.7 mm vs 23.9 mm with high-level text only), removing staged diffusion degrades MPJPE to 27.3 mm, and text guidance plus text-to-hand fusion each contribute measurable gains. An interactive ChatGPT pipeline turns high-level user requests into low-level contact descriptions for grasp editing.

## Limitations
The paper's conclusion notes that its modeling targets contact in this static setting and states that exploring the modeling of dynamic contacts with text is left to future work. The paper does not report a dedicated limitations section, and no other limitations are explicitly discussed.
