# GHOST: Fast Category-agnostic Hand-Object Interaction Reconstruction from RGB Videos using Gaussian Splatting

## Summary
GHOST (Gaussian Hand-Object Splatting) is a fast, category-agnostic framework for reconstructing dynamic hand-object interactions from monocular RGB videos using 2D Gaussian Splatting, where both hands and objects are represented as dense, view-consistent Gaussian discs, with three key innovations (geometric-prior retrieval, grasp-aware alignment, hand-aware background loss) enabling complete, physically consistent, and animatable reconstructions an order of magnitude faster than prior category-agnostic methods.

## 1. Problem and Setting
- Fast, category-agnostic 3D reconstruction of dynamic hand-object interactions from monocular RGB videos.
- Input: monocular RGB video of hand-object interaction.
- Output: 3D hand mesh (MANO parameters), 3D object represented as 3D Gaussians, and 6D object pose trajectory.
- Task: hand-held object reconstruction from video. Classified under shape retrieval priors because the geometric-prior retrieval step bootstraps the object shape representation.

## 2. Core Method
- 2D Gaussian Splatting is used to represent both hands and objects as dense, view-consistent Gaussian discs.
- Three key innovations:
  1. Geometric-prior retrieval and consistency loss that completes occluded object regions (using foundation-model-based retrieval to bootstrap a 3D shape prior).
  2. Grasp-aware alignment that refines hand translations and object scale to ensure realistic contact.
  3. Hand-aware background loss that prevents penalizing hand-occluded object regions during optimization.
- How FM prior is injected: a geometric prior is retrieved (likely via foundation model features) to initialize the object shape, then a consistency loss maintains plausibility with the prior during optimization.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: geometric-prior retrieval uses foundation model embeddings; the 3D Gaussian Splatting framework is the efficient representation backbone.
- Domain knowledge: hand model (MANO); physical plausibility constraints; assumption that the object's rough shape category can be retrieved from a database.
- Training data: no HOI-specific training. The FM embedding models are pre-trained; the 3D object database is constructed from large 3D asset collections.
- Assumption: object is rigid; video captures sufficient viewpoint coverage.

## 4. Experiments and Findings
- Datasets: ARCTIC, HO3D, and in-the-wild datasets.
- Metrics: 3D reconstruction accuracy, 2D rendering quality, runtime.
- Achieves state-of-the-art accuracy in 3D reconstruction and 2D rendering quality while running an order of magnitude faster than prior category-agnostic methods.
- The three innovations (retrieval, grasp-aware alignment, hand-aware background loss) each contribute to the final performance.

## 5. Strengths and Limitations
### Strengths
- Fast reconstruction (order of magnitude faster than prior methods) suitable for interactive applications.
- Category-agnostic: works on any object with a reasonable match in the database.
- 3D Gaussian representation supports real-time novel-view rendering.
- Three tailored innovations specifically address the hand-object reconstruction challenges.

### Limitations
- Retrieval quality depends on the object being represented in the database.
- Foundation model embeddings may fail for heavily occluded, textureless, or unusual objects.
- Retrieved shape may not match the instance-level geometry details.
- Physical constraints are heuristic; no physics simulation.
- Dynamic motion modeling is limited.

## 6. Takeaway
GHOST demonstrates a practical synthesis of the retrieval and optimization paradigms, specifically tailored for hand-object interactions: FM-based retrieval provides the semantic knowledge to bootstrap from a database, while efficient 2D Gaussian Splatting with hand-aware design choices provides the optimization framework for fitting to observations. The three innovations (retrieval, grasp-aware alignment, hand-aware background) exemplify how domain-specific design choices matter for HOI, even when leveraging general-purpose FMs.
