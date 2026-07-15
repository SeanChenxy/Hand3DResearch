# FreiHAND (ICCV 2019)

> Zimmermann, Ceylan, Yang, Russell, Argus, Brox. *FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape From Single RGB Images.* ICCV 2019. DOI: 10.1109/ICCV.2019.00090. Zotero Key: `UPJ9BN4I`.

## Summary
FreiHAND is the first large-scale benchmark that provides 3D hand pose and full MANO shape ground truth for single RGB images. Using a multi-view green-screen capture and a semi-automatic iterative annotation pipeline, it addresses the scarcity of hand-only 3D data and the failure of cross-dataset generalization.

## 1. Dataset Purpose
- Fills the gap of a "single RGB → 3D hand pose & shape" benchmark: prior datasets (e.g., CMU Panoptic) were mostly from mocap studios, depth sensors, or synthetic data and exhibited a domain gap to real RGB images.
- Tasks: 3D hand joint estimation from a monocular RGB image, MANO shape-parameter regression, and articulated hand mesh reconstruction.
- A hand-only benchmark: no objects and no bi-manual interaction; focused purely on hand shape recovery.
- Provides both hand pose and hand shape annotations, unlike earlier datasets such as STB or RHD that only provided joints.

## 2. Data Composition
- Source: real capture. 32 subjects perform 32 prescribed hand-pose categories against a green screen, shot by 8 calibrated multi-view RGB cameras simultaneously (256 viewpoints in total).
- Viewpoint: third-person multi-view; the test set is in-the-wild single-view RGB.
- Scale: 130,240 training frames, 3,960 evaluation frames, 13,272 test frames. About 4 subjects do not appear in the training set.
- Object and action coverage: 32 hand-pose categories (reaching, grasping, pinching, gesturing, etc.), with no object interaction.
- No hand-object occlusion, no contact, no bi-manual interaction, no tool use.

## 3. Annotation and Supervision
- Hand: 3D 21-joint annotations obtained via multi-view + iterative human-in-the-loop optimization; MANO shape β and pose θ.
- Object: none.
- Interaction: none.
- Scene: green-screen masks, camera intrinsics/extrinsics; an in-the-wild test set for cross-domain evaluation.
- Robot-related annotations: none; a purely visual benchmark.

## 4. Supported Evaluation
- Benchmark task: monocular RGB → MANO hand shape / pose estimation.
- Key metrics: mesh error (vertex-to-vertex distance), joint position error (3D / 2D), F-score @ 5mm/15mm, PA-MPJPE.
- Role: used as both training and evaluation set; its evaluation split is the de facto standard hand-mesh evaluation protocol.
- Cross-dataset capability: the paper explicitly tests generalization trained on FreiHAND and evaluated on STB / RHD, serving as an "in-the-wild generalization" indicator.

## 5. Why It Matters
- Provides, for the first time, a large-scale and reproducible ground truth for "single RGB → full hand mesh".
- The multi-view iterative annotation pipeline becomes the methodological blueprint for later datasets such as HO-3D and HOT3D.
- Demonstrates that 3D hand shape can be regressed end-to-end from a single RGB image, removing the need for depth sensors.
- Networks trained on FreiHAND clearly outperform those trained on STB / RHD when evaluated on those datasets, establishing the "unified large training set + cross-domain evaluation" paradigm.
- Acts as one of the training-data sources for nearly all hand-only mesh-reconstruction methods (METRO, I2L-MeshNet, etc.).

## 6. Limitations and Biases
- No objects and no interaction: a purely hand-shape benchmark, not reflective of recovery under hand-object occlusion.
- 32 prescribed poses and 32 subjects: limited motion and subject diversity.
- Green-screen background: domain gap to in-the-wild backgrounds remains.
- Annotation depends on MANO: any hand shape outside the MANO low-dimensional space (infants, deformities, severe injury) cannot be represented.
- No contact, affordance, language, or scene context.

## 7. Takeaway
FreiHAND is best suited to demonstrate RGB-only 3D hand pose / MANO shape estimation and cross-dataset generalization. **Not suitable** for evaluating hand-object interaction, hand-object occlusion, bi-manual manipulation, or in-hand manipulation. In this survey, FreiHAND plays the role of a "hand-only baseline / hand-only upper bound" and serves as the standard evaluation source for all hand-object methods on hand-only tasks.
