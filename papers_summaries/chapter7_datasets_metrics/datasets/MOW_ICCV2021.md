# Reconstructing Hand-Object Interactions in the Wild (MOW)

**Authors:** Zhe Cao, Ilija Radosavovic (equal contribution), Angjoo Kanazawa, Jitendra Malik
**Date:** 2021 (ICCV 2021)
**Identifier:** none printed on the PDF page 1; the paper mentions an extended arXiv version and a project page but prints no arXiv ID or URL in the text.
**Zotero item:** `8QJ6KRM6` ([Zotero](zotero://select/library/items/8QJ6KRM6))
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.

## Summary
This paper targets joint 3D reconstruction of hands and objects from single RGB images in the wild, and identifies the evaluation gap that all existing 3D hand-object datasets were collected in the lab with few participants and objects (e.g., HO3D: 10 subjects, 10 objects; ContactPose: 50 subjects, 25 objects; GRAB: 10 subjects, 51 objects). The authors propose RHO (Reconstructing Hands and Objects), an optimization-based method that combines 2D image cues with 3D contact priors learned from MoCap data, and use it with human intervention to build MOW (Manipulating Objects in the Wild), a dataset of 500 in-the-wild images annotated with 3D hand pose, 3D object pose, and object models spanning 121 categories. On lab benchmarks RHO performs better or on par with the state of the art (e.g., 9.7 mm vs 14.7 mm hand MAE on HO3D), while existing learning methods trained on lab data struggle on wild images, reinforcing the need for the dataset.

## Background and Motivation
Understanding hand-object interactions requires 3D reasoning, but single-image 3D recovery is underconstrained and made harder by heavy mutual occlusion, a wide range of small daily objects absent from labeled recognition datasets, and fine-grained contact. Because 3D annotation in the wild is difficult, prior data collection focused on in-the-lab settings (thermal or marker-based MoCap, multi-view rigs), leaving a large reality gap between lab datasets and the richness of real-world interaction images (first- and third-person, varied viewpoints and environments). The paper argues this limits both evaluation of reconstruction methods and learning about human manipulation, and addresses it by turning a reconstruction technique into a semi-automatic annotation tool.

## Dataset Construction
MOW provides 500 annotated in-the-wild images (not videos; temporal information is not exploited).
- Source: static frames selected from EPIC Kitchens and 100 Days of Hands using a hand-object detector that picks images with high hand-object bounding-box overlap; viewpoints include both first-person and third-person.
- Scale and diversity: 500 images covering 121 distinct object categories with a long-tailed category distribution, and 450 participants (compared to 10-50 in prior datasets), with much greater diversity of manipulation actions than previous datasets.
- Objects: 3D models chosen by annotators from an existing collection or found online; primary sources are the YCB dataset and the Free3D online platform.
- Annotation pipeline: (1) model selection for the manipulated object; (2) reconstruction with RHO, semi-automatic in that the annotator can adjust loss weights, with default settings usually a good starting point; (3) verification, where the annotator accepts the result, returns it to reconstruction, or discards it, iterating until convergence.
- Annotations: per image, a 3D object model, 3D object pose, and 3D hand pose (MANO); amodal masks and contact maps are derivable from the reconstructions.
- Quality evaluation on a 100-image sample: amodal masks derived from the 3D annotations reach a mean IoU of 0.77 for objects (0.84 large, 0.78 medium, 0.64 small) and 0.68 for hands against human-labeled amodal masks; a user study rates reconstruction quality 4.16 out of 5, and the 3D object model matches the true object in 92% of cases (most mismatches due to imprecise mesh topology, e.g., a cylinder fitted to a handled mug).

## Evaluation Protocol
- Task (method): from a single RGB image with a known 3D object model, reconstruct the 3D hand (MANO) and object pose. RHO proceeds in four steps: (a) hand pose initialization with FrankMocap refined by fitting to 2D keypoints; (b) object pose estimation by differentiable rendering against an estimated instance mask (PointRend trained on COCO, instance chosen by IoU with a detected hand box) and a monocularly estimated depth map; (c) joint optimization with interaction (bidirectional Chamfer) and collision (hand SDF) losses to resolve depth/scale ambiguity; (d) hand pose refinement by a small network trained on 3D contact priors from the GRAB MoCap dataset.
- Metrics: hand MAE (mm) over 21 joints after root alignment and global scale alignment; object accuracy as Chamfer distance between ground-truth object vertices and the predicted-posed CAD vertices; for interaction quality, hand-object center distance and an SDF-based collision score.
- Lab benchmarks: HO3D (68 sequences, 10 subjects, 10 objects) and FPHA (4 annotated objects), following the same testing split as the compared method for FPHA.
- Baselines: the state-of-the-art feed-forward method of Hasson et al. (photometric-consistency training) with the same monocular RGB input and known 3D object model; RHO is tested without 3D supervision. In-the-wild comparison against the same method is qualitative.
- Ablations: cumulative addition of interaction, depth, collision losses and the refinement stage.

## Findings and Analysis
- In the lab, RHO outperforms the state of the art on HO3D with hand MAE of 9.7 mm vs 14.7 mm and object Chamfer distance of 19.9 vs 26.8. On FPHA, hand MAE is 14.2 mm vs 18.0 mm; its object MAE is slightly worse (23.9 vs 22.3), attributed to the compared method using the FPHA action split where the same objects appear in train and test, whereas RHO uses no 3D supervision.
- Ablations show individual hand/object reconstruction yields a hand-object distance of 414.8 mm and no collision; the interaction loss cuts distance to 71.5 mm but inflates collision to 39.8; adding depth and collision losses reduces collision to 7.7 while keeping distance at 76.4 mm; refinement slightly improves both (6.5 collision, 75.8 mm distance).
- In the wild, the compared lab-trained method struggles due to limited object categories in training, while RHO produces better reconstructions across diverse categories (qualitative comparison).
- Dataset analysis shows a long-tailed object distribution; an Isomap embedding of hand poses reveals a pen cluster but no other clear category clusters, indicating multiple grasp types per category and shared grasps across categories (e.g., pen and spoon); the first embedding dimension corresponds to grasp closure, spanning fully closed to fully open.

## Contributions
- RHO, an optimization-based procedure for reconstructing hand-object interactions in the wild across diverse object categories without 3D supervision, combining 2D keypoint/mask/depth cues with 3D contact priors learned from MoCap data.
- Quantitative and qualitative evidence that lab-trained methods do not transfer to in-the-wild images, while RHO is better or on par in the lab and markedly better in the wild.
- MOW, a new 3D dataset of 500 in-the-wild images spanning 121 object categories with instance category, 3D object model, 3D object pose, and 3D hand pose annotations, produced by a semi-automatic RHO-based annotation pipeline with human verification, plus an analysis of grasp diversity.

## Limitations
The annotations are produced by the authors' own reconstruction method with human intervention, so ground truth inherits RHO's errors and ambiguities; the paper mitigates this via iterative verification and amodal-mask/user-study checks on a 100-image sample, but 3D accuracy against independent measurements is not verified. RHO assumes a known 3D object model and a single hand-object pair per image, requires per-example optimization, and can yield imperfect results across viewpoints due to ambiguity, which is why the verification step exists. The dataset is small (500 images) and static (single frames, no temporal sequences), and the object model mismatch rate is 8%, mostly imprecise mesh topology. These points are partly acknowledged in the paper; the small scale and annotation-provenance caveats are evident from the construction.
