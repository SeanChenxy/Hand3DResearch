# HOnnotate (CVPR 2020) — Method View

> Hampali, Rad, Oberweger, Lepetit. *HOnnotate: A Method for 3D Annotation of Hand and Object Poses.* CVPR 2020. DOI: 10.1109/CVPR42600.2020.00326. Zotero Key: `V9JIHJUS` (same paper as HO-3D, viewed from the method angle).

## Summary
The HOnnotate method view: a multi-view RGB-D joint optimization algorithm that automatically annotates RGB images with 3D hand pose (MANO) and 6D object pose simultaneously, bypassing the intrusiveness of mocap / markers, and constructs the HO-3D dataset (77,558 frames) as a proof of concept. It is the prototype for nearly all subsequent 3D HOI annotation pipelines.

## 1. Dataset Purpose
- The core contribution is methodological: "how to attach 3D hand + 6D object ground truth to real images simultaneously", not just data release.
- Tasks: 6D object pose estimation from RGB; 3D hand pose and shape from RGB; joint hand-object reconstruction.
- Canonical scene: single hand + rigid object (YCB set); does not cover bi-manual, articulated, or in-the-wild.
- For the community, the methodological value of HOnnotate far exceeds the dataset size itself.

## 2. Data Composition
- Source: real capture (exactly the same data as HO-3D v1/v2).
- Viewpoint: multi-view RGB-D (Intel RealSense, 1–5 cameras) with point-cloud fusion.
- Scale: 77,558 frames / 68 sequences / 10 subjects / 10 YCB objects.
- Object and action coverage: identical to HO-3D v2; focused on single-hand grasping, lifting, and manipulation.
- Includes natural hand-object mutual occlusion; sequences range from ~1 s to over 30 s.

## 3. Annotation and Supervision
- Pipeline: multi-view depth fusion → scene point cloud → model-based 6D object pose initialization with YCB CAD → joint optimization of 3D hand + 6D object pose under hand-object, silhouette, and temporal smoothness constraints → manual spot check.
- Hand: 3D 21 joints, MANO β / θ, hand mesh.
- Object: 6D pose (rotation + translation), aligned to YCB CAD.
- Contact: no explicit contact map; can be inferred via a distance threshold.
- No affordance, no language, no force / tactile.

## 4. Supported Evaluation
- The evaluation is not the algorithm itself, but "training/evaluating downstream models on data annotated by HOnnotate". Downstream tasks include hand pose, object pose, and joint HOI.
- In-paper ablations: (1) without HOnnotate annotation vs with → hand-pose accuracy drops by a large MPJPE margin; (2) effect of each optimization term (contact, silhouette, depth, temporal smoothness).
- An "unseen objects" split serves as an early cross-object generalization protocol.
- The whole paper argues for "the feasibility of data-driven HOI research".

## 5. Why It Matters
- Establishes "3D hand + 6D object" joint optimization as the annotation paradigm, in contrast to the then-common "label hand, then label object" two-stage approach.
- The combination of contact prior / silhouette prior / depth prior / temporal smoothness becomes a reference recipe for follow-up multi-object datasets (e.g., ArticulatedData, HOI4D).
- Demonstrates the feasibility of replacing mocap / physical markers with algorithmic annotation for large-scale 3D HOI datasets.
- Serves as the standard training-set source for HMP, iHOI, AlignSDF, and many other methods on HO-3D v1/v2.

## 6. Limitations and Biases
- Assumes objects have CAD models, and the 10-object set is small: no solution for in-the-wild objects without accessible CAD.
- Depth-sensor noise propagates to MANO fitting; failure rate is high on highly reflective / transparent objects.
- The optimization depends on an initial pose estimator; a bad initialization leads to local optima.
- Not robust to bi-manual, articulated, tool-use, or dynamic-motion scenarios.
- Missing contact / affordance ground truth — systematic error is introduced if downstream methods consume it directly.

## 7. Takeaway
HOnnotate should be treated in this survey as a "paradigm entry for 3D HOI annotation" rather than a plain dataset — it proves the basic assumption that hand and object can be annotated simultaneously on real RGB-D. In Ch7 of survey v5, it complements the HO-3D dataset entry: HO-3D describes "what the dataset looks like", HOnnotate describes "how it was built".
