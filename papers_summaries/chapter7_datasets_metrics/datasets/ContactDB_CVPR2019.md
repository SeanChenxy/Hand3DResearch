# ContactDB (CVPR 2019)

> Brahmbhatt, Ham, Kemp, Hays. *ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging.* CVPR 2019. DOI: 10.1109/CVPR.2019.00891. Zotero Key: `NIPUAPML`.

## Summary
ContactDB uses thermal imaging as a "contact sensor" to record the heat map (contact imprint on the object surface) after 3,750 grasps of 50 household objects, plus 3D object meshes. It provides 375K frames of synchronized RGB-D + thermal video, and is a pioneering dataset for "inferring contact from grasp".

## 1. Dataset Purpose
- Solves the fundamental difficulty that "existing datasets cannot directly measure hand-object contact". ContactDB uses thermal imaging as the contact proxy — after a hand grasps an object, the object surface leaves a thermal imprint, which is read by a thermal camera.
- Tasks: (1) contact map (on object surface) prediction; (2) predicting grasp contact patterns from object shape; (3) image-to-3D / 3D-to-3D contact synthesis.
- Anchors the "contact as physical measurement" paradigm, in contrast to ContactPose (multi-view-optimization contact).
- 50 objects / 3,750 grasps is comparable in scale to ContactPose, but the contact ground truth comes from physical measurement (more reliable).

## 2. Data Composition
- Source: real capture. Multiple subjects grasp 50 3D-printed objects (with embedded sensors + synchronized thermal imaging) in a controlled studio.
- Viewpoint: multi-view RGB-D + 1 thermal camera (FLIR); the subject's hand leaves before the thermal map of the object surface is collected.
- Scale: 50 objects × multiple grasps × 3D mesh textured with contact map; 3,750 3D meshes + 375K frames of synchronized RGB-D + thermal images.
- Object and action: 50 household objects (mug, bottle, box, kitchen items) of different sizes / shapes / functions.
- Each object is grasped by multiple subjects, and a contact heatmap is left after each grasp.

## 3. Annotation and Supervision
- Contact: vertex-level contact intensity on the object surface (from thermal imaging).
- Object: 3D mesh (high-precision 3D printing), each mesh attached with a contact texture.
- Hand: no direct 3D hand annotation (only RGB-D video).
- Scene: multi-view RGB-D + thermal, camera intrinsics / extrinsics.
- No language, no robot, no MANO fitting.

## 4. Supported Evaluation
- Benchmark tasks: (1) object surface contact map prediction (vertex-level F-score); (2) image-to-contact prediction (image-to-image translation); (3) 3D shape-to-contact prediction (3D ConvNet).
- Key metrics: contact F-score @ different thresholds (low / high temperature); image-to-contact IOU.
- Provides an "unseen object" split.
- The standard evaluation source for contact-from-shape / shape-to-contact research.

## 5. Why It Matters
- The first dataset to use thermal imaging to directly measure hand-object contact.
- Contact data comes from physical measurement (rather than multi-view optimization), serving as a ground-truth verification reference for methods such as ContactPose / CP3.
- Inspired basic research on "active area" (the area of an object that the hand tends to touch), "the relationship between functional intent and object size", etc.
- The core citation in the "shape → contact" research of Ch3 "shape prior" and Ch4 "affordance".
- 50 objects + 3,750 grasps is the representative "physical contact measurement" dataset of the early stage.

## 6. Limitations and Biases
- Only 50 objects: object diversity is limited.
- No 3D hand pose annotation: insufficient compared to the joint "contact + hand" annotation of ARCTIC, ContactPose, etc.
- Thermal imaging measures the "location of the heat imprint left by the hand", not "real-time contact" — there is a 1–2 second delay; dynamic contact does not apply.
- The objects need to be 3D-printed: the object set is limited to the 50 selected by the authors.
- No articulated object, no bi-manual, no tool use, no language.
- No robot annotation, and the direct connection to imitation learning / manipulation learning is limited.

## 7. Takeaway
ContactDB is best for demonstrating the predictive capability of "object shape → contact pattern" and the feasibility of contact as a physically measurable signal. **Not suitable** for evaluating hand pose, 6D object pose, bi-manual, articulated, dynamic contact, language-conditioned, or in-the-wild tasks. In this survey, ContactDB plays the role of "physical contact measurement benchmark" and serves as a reference for the "shape → contact" research in the "shape prior" of Ch3 and "affordance" of Ch4. It complements ContactPose: ContactPose provides "real-time contact", and ContactDB provides "physically measured contact".
