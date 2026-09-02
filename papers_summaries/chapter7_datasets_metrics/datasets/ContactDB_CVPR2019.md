# ContactDB: Analyzing and Predicting Grasp Contact via Thermal Imaging

**Authors:** Samarth Brahmbhatt, Cusuh Ham, Charles C. Kemp, James Hays  
**Date:** 2019 (CVPR 2019)  
**Identifier:** DOI `10.1109/CVPR.2019.00891`  
**Zotero item:** `NIPUAPML` ([Zotero](zotero://select/library/items/NIPUAPML))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

ContactDB is the first large-scale dataset recording detailed hand-object contact maps for human grasps, obtained by texture-mapping thermal images onto 3D object meshes after a grasp. Fifty participants grasped fifty 3D-printed household objects with two post-grasp functional intents, yielding 3750 contact-map-textured meshes and 375K synchronized RGB-D plus thermal frames. Analysis shows grasps depend on functional intent, object size, and non-fingertip contact, and benchmark experiments predict diverse contact maps from object shape using single-view pix2pix and 3D PointNet/VoxNet models.

## Background and Motivation

Hand-object contact is fundamental to grasping, but contact regions are usually occluded from visual-light imaging, so prior datasets recorded joint configurations with gloves or trackers, coarse grasp-type labels from videos, or single-point-per-fingertip tactile estimates. The paper argues that object-centric contact maps enable analysis of grasping preferences by intent, shape, size, and category, learning shape features for grasp prediction, and re-targeting grasps to different hand models. Thermal imaging makes contact observable: heat from the hand leaves a thermally visible imprint on the object surface after release, and the authors verified empirically that heat conduction from contact dominates the thermal measurement under their protocol.

## Dataset Construction

Fifty household objects were 3D printed in white PLA at 15% infill (chosen for heat retention), selected from YCB plus additions such as flashlight, eyeglasses, computer mouse, Stanford bunny, and Utah teapot, and five geometric primitives (cube, cylinder, pyramid, torus, sphere) at 12, 8, and 4 cm scales. Fifty participants (mostly 20-25 years old) grasped each object in commonly encountered orientations, held it 5 seconds, and handed it to an experimenter wearing an insulating glove; participants used chemical hand warmers and avoided in-hand manipulation. A FLIR Boson 640 thermal camera rigidly mounted on a Kinect v2 RGB-D sensor recorded the object rotating on a turntable that paused at 9 equally spaced angles. Objects were grasped with two functional intents: hand-off (48 objects, 2400 textured meshes, 240K frames) and use (27 objects, 1350 meshes, 135K frames), for 3750 meshes and 375K frames in total. Processing segments the object from depth, estimates 6D pose per view with ICP plus circle interpolation, and runs color-map optimization to produce coherently textured contact maps.

## Evaluation Protocol

Contact prediction is evaluated on three held-out object classes (mug, pan, wine glass). The single-view task uses a modified pix2pix GAN taking 4-channel RGB-D input to predict a 2D contact map for the visible surface. The 3D task represents shape as a PointNet point cloud or a VoxNet 64-cubed voxel occupancy grid and trains with two one-to-many strategies: sMCL ensembles (k=1 and k=10) and DiverseNet with a control variable (k=10). Ground-truth contact maps are thresholded at 0.4; prediction error is the percentage mismatch after matching each ground-truth map with the closest of the k diverse predictions, discarding predictions with no contact.

## Findings and Analysis

Functional intent strongly shifts contact: for example, 100% of participants touched the scissors handle when using them versus 38% in hand-off, while the hammer head was touched by 38% in hand-off but 0% in use, and eyeglass temples by 64.58% in use versus 4% in hand-off. Object size changes grasp topology: small objects elicit two-or-three-fingertip grasps, large objects produce bimanual or fingertip-only grasps, and participants with smaller hands prefer bimanual grasps, with no bimanual grasps for medium and small objects. Contact area for many objects exceeds a loose upper bound on five-fingertip contact, demonstrating the importance of palm and proximal-finger contact. In prediction, 3D models beat single-view prediction, voxel grids beat point clouds, and diverse-prediction training is essential: sMCL with k=1 averages 55.37% error for hand-off and 44.48% for use, while VoxNet-sMCL k=10 falls to 11.64% and 17.27%, and VoxNet-DiverseNet reaches 8.72% on the use intent.

## Contributions

The paper contributes the first large-scale contact-map dataset from functional grasping with paired RGB-D-thermal data, an analysis of intent, size, and non-fingertip contact effects, and benchmark protocols and baselines for single-view and 3D diverse contact-map prediction from shape.

## Limitations

The method records contact only after object release rather than during the grasp, and thermal intensity depends on contact duration, pressure, and heat conduction, which the protocol keeps roughly constant. Only heat-retaining 3D-printed PLA objects are covered, participants avoid in-hand manipulation to prevent smudging, and only two functional intents are captured; the prediction baselines also leave a wide accuracy gap depending on representation and diversity strategy.
