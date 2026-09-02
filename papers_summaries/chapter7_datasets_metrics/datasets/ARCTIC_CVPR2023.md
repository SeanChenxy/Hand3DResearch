# ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation

**Authors:** Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, Otmar Hilliges  
**Date:** CVPR 2023 (June 2023, per Zotero record)  
**Identifier:** DOI `10.1109/CVPR52729.2023.01244`  
**Zotero item:** `P7QSDW4P` ([Zotero](zotero://select/library/items/P7QSDW4P))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

ARCTIC is the first large-scale dataset of two hands dexterously manipulating articulated objects: 339 sequences, 10 subjects, 11 articulated objects, 2.1M RGB images from 8 static allocentric plus 1 egocentric view at 30 fps (2800 x 2000), paired with accurate marker-based mocap ground truth for both hands, the full body (SMPL-X), articulated object pose, and dynamic hand-object contact. It defines two tasks with baselines: consistent motion reconstruction (reconstruct two hands and an articulated object from monocular video with spatio-temporally consistent contact, via ArcticNet) and interaction field estimation (dense per-vertex hand-object distances from images, via InterField).

## Background and Motivation

- Humans understand that object state changes are caused by manipulation (e.g., a book opening), but machines lack data for this: no prior dataset had ground-truth 3D annotations for physically consistent, synchronized motion of hands and articulated objects.
- Existing hand-object datasets focus on grasping rigid objects, where hand poses are largely static over time; dexterous, unconstrained manipulation of articulated objects (scissors, laptops) with jointly evolving hand pose and object state is absent. HOI4D includes articulated objects but has a single egocentric view (introducing ambiguity for occluded fingers), one hand, no full-body capture, and mainly grasping-style interaction.
- Capturing dexterous interaction is hard because fast motion and heavy self- and mutual occlusion defeat the 1-8 commodity RGB-D cameras used by prior datasets, forcing those datasets to slow, controlled motion. ARCTIC instead uses high-end marker-based mocap with small markers that minimally interfere.

## Dataset Construction

- **Scale and views:** 339 sequences of 10 subjects (5 female, 5 male) manipulating 11 articulated objects, 2.1M RGB images from 8 static allocentric views and 1 moving egocentric view at 30 fps, image resolution 2800 x 2000. Subjects either "use" the objects (1.7M images) or merely "grasp" them (457K images). Rendered depth of hands, body, and objects is also provided.
- **Objects:** 11 objects, each consisting of two rigid parts rotating about a shared estimated axis (e.g., flip phone, scissors, laptop); meshes are scanned with an Artec 3D handheld scanner and split into two articulated parts in Blender.
- **Capture pipeline (five steps):** (1) canonical geometry: personalized SMPL-X templates per subject from 3dMD 3D scans registered in T-pose and varying poses; (2) estimation of each object's rotation axis; (3) synchronized capture with 54 Vicon Vantage-16 infrared mocap cameras and the multi-view RGB rig, using small hemispherical markers (1.5 mm radius) placed on the dorsal side of hands and objects to avoid impeding interaction; (4) surface solving: marker-to-vertex correspondences are initialized and refined with MoSh++, SMPL-X pose is optimized to fit body, hands, and realistic wrist articulation (MANO alone lacks wrist articulation), and object pose is recovered per frame as the 6D rigid transform of the base part plus a 1D articulation angle about the axis; (5) hand-object contact is computed from mesh proximity.
- **Annotation payload:** 3D hand and object meshes per frame, full-body SMPL-X pose, per-vertex dynamic contact, calibrated cameras for 9 views. T-SNE clustering of 3D hand joints shows ARCTIC covers a significantly larger hand-pose range than DexYCB, HO-3D, H2O, H2O-3D, and FPHA, and contact heatmaps show higher palm-region contact than HO-3D and GRAB, reflecting dexterous manipulation rather than static grasping.
- **Positioning:** among compared datasets, ARCTIC is the only one with articulated objects, both hands, full human body, dexterous manipulation, and mocap-grade annotation; it provides the only setup where monocular, multi-view, and egocentric reconstruction can be studied on the same interactions.

## Evaluation Protocol

- **Splits:** data is split by subjects: 8 for training, 1 male for validation, 1 female for testing (gender-balanced evaluation with one male and one female subject). Two protocols: allocentric (models see and are evaluated on static third-person views only) and egocentric (training may use all views, but evaluation uses only the egocentric view).
- **Task 1, consistent motion reconstruction:** given a monocular video, reconstruct two MANO hands and the articulated object per frame such that hand-object contact, articulation, and motion are spatio-temporally consistent. Object pose is 7D: 1D articulation plus 6D rigid pose; the object model maps this pose to the scanned mesh. Baseline ArcticNet (single-frame SF and recurrent LSTM variants): a CNN encoder produces image features; hand decoders regress MANO parameters and translations for both hands; an object decoder regresses the articulated object pose; trained with ground-truth 3D keypoints, 2D projected keypoints, and model parameters.
- **Task 1 metrics:** contact deviation (CDev, mm; average distance between predicted pairs of vertices that are in contact, under 3 mm, in ground truth), motion deviation (MDev, mm; disagreement of frame-to-frame motion of hand-object vertex pairs in stable-contact windows of at least 15 frames / 0.5 s, detected with a 3 mm proximity threshold), acceleration error (ACC, m/s2; smoothness of hand and object vertex motion, root-subtracted, object root at the base-part center), MPJPE (mm; root-relative 21-joint error per hand), average articulation error (AAE, degrees), success rate (percentage of object vertices with root-subtracted L2 error under 5% of object diameter), and MRRPE (mm; relative root translation between hand-hand and hand-object).
- **Task 2, interaction field estimation:** for every vertex of each hand, estimate the shortest distance to the object mesh and vice versa (fields hand-to-object and object-to-hand for both hands), capturing proximity even when hands are not in contact. Baseline InterField (SF and LSTM variants): CNN image features are concatenated to subsampled canonical-pose vertices, passed through a PointNet, and regressed to distances. Metrics: average distance error (mm) and acceleration error of the predicted field sequence.

## Findings and Analysis

- **Temporal modeling improves physical consistency:** on the allocentric test split, ArcticNet-LSTM beats ArcticNet-SF on CDev (38.9 vs. 41.6 mm), MDev (9.3 vs. 10.4 mm), and acceleration error (hand 5.0 vs. 5.7, object 6.1 vs. 7.6 m/s2), with similar MPJPE (21.5 mm) and slightly better articulation (5.2 vs. 5.4 degrees) and success rate (73.5% vs. 71.4%). This shows recurrent reasoning matters for temporally consistent contact and smooth motion, not just per-frame pose accuracy.
- **Egocentric reconstruction is much harder:** the egocentric protocol drops object success rate to 53.5% (LSTM) versus 73.5% allocentric on test, with egocentric MRRPE (28.3-31.8 mm) lower than allocentric (47.1-52.4 mm) but articulation error higher (6.6 vs. 5.2 degrees), quantifying the difficulty of first-person dexterous reconstruction.
- **Interaction fields:** InterField-LSTM reaches 8.7/9.1 mm average distance error (hand-to-object/object-to-hand) on the allocentric test split and 8.0/9.1 mm egocentric, with smoother fields than the single-frame variant (ACC 1.8/1.8 vs. 2.1/2.0 m/s2 egocentric test). Predicted fields qualitatively correlate with ground truth and mark plausible contact regions.
- **Contact diversity:** fingertips remain the most contact-prone regions, as in HO-3D and GRAB, but dexterous manipulation raises palm contact; contact heatmaps on objects match their function (e.g., bottom support regions for a small waffle iron).

## Contributions

- The first large-scale dataset of two hands dexterously manipulating articulated objects, with synchronized multi-view and egocentric video and mocap-grade 3D ground truth for hands, full body, articulated object pose, and dynamic contact.
- Two novel articulated hand-object interaction tasks: consistent motion reconstruction (spatio-temporally consistent hand-object reconstruction) and interaction field estimation (dense relative hand-object distance estimation beyond binary contact).
- Baselines for both tasks (ArcticNet and InterField, in single-frame and temporal variants) evaluated under allocentric and egocentric protocols to seed future comparison.

## Limitations

The paper has no dedicated limitations section; the following are evidenced by its text and results.

- Object scope is narrow: 11 objects, each with exactly two rigid parts rotating about a single shared axis (1-DoF articulation), and 339 sequences; interaction classes are limited to "use" and "grasp" of these objects.
- Ground truth relies on physical markers (1.5 mm hemispheres) attached to hands and objects; the paper explicitly frames this as a trade-off between accuracy and marker intrusiveness, mitigated by dorsal-side placement rather than eliminated.
- The introduced baselines are preliminary and far from solving the tasks: contact deviation around 39-45 mm, egocentric object success rate near 54%, and egocentric articulation error around 6.4-8.0 degrees, leaving substantial headroom.
- Evaluation covers only the two proposed tasks and two baseline families; the authors position depth-based articulated object pose estimation and bimanual motion generation as future directions enabled by the data rather than benchmarked here.
