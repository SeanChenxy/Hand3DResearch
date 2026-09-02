# TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding

**Authors:** Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, Li Yi  
**Date:** 2024-01-16  
**Identifier:** [arXiv:2401.08399](https://arxiv.org/abs/2401.08399)  
**Zotero item:** No record found in the Zotero library; identity verified against arXiv metadata.  
**Evidence status:** Verified against arXiv metadata and full-text PDF extraction; no Zotero record exists for this dataset paper.  

## Summary

TACO is a bimanual tool-use dataset of 2.5K motion sequences and 5.2M video frames covering 131 tool-action-object triplets, 20 tool categories, 196 tool instances, and 15 actions performed by 14 participants, captured with 12 allocentric cameras, one egocentric camera, and motion capture, and annotated with MANO hand poses, 6D object poses, and object meshes. Around this data the authors build four generalization benchmarks: compositional action recognition, hand-object motion forecasting, cooperative grasp synthesis, and interaction field estimation, each with train/test splits that isolate novel tool geometry, novel triplets, and compound generalization.

## Background and Motivation

Tool-use manipulation is inherently compositional: an action is defined jointly by the tool, the action, and the target object, and bimanual tool use additionally requires hand-hand and hand-tool cooperation. Existing datasets either lack tool affordance structure, cover few triplets, or are single-hand, so models cannot be evaluated on whether they generalize to unseen tools or recombinations of familiar triplets. TACO is designed to expose this compositional structure explicitly, with dense multi-view capture and marker-based ground truth enabling precise 3D annotation of two hands, tools, and target objects.

## Dataset Construction

Each capture rig combines 12 FLIR allocentric cameras (4096 x 3000), one egocentric RealSense L515 mounted on a helmet (1920 x 1080), and a NOKOV motion capture system with 6 infrared cameras, all at 30 Hz; objects are scanned with an EinScan scanner into meshes of up to 100K faces. 14 participants perform 15 actions on 196 tool instances across 20 categories, yielding 131 distinct tool-action-object triplets and 2.5K sequences with 5.2M frames. Annotation is marker-based: object pose is recovered by a marker-to-surface optimization inspired by Mosh++; hand poses come from YOLOv3 hand detection, MMPose keypoint estimation, and RANSAC-based triangulation, followed by contact-aware MANO optimization combining 2D/3D keypoint, angle, temporal, and contact losses. Hands and objects are automatically segmented with SAM plus a track-anything model, and markers are removed from images by a U-Net trained for marker segmentation with LAMA inpainting; the removal network reduces marker mIoU from 63.8% to 11.1% on inpainted images. Cross-dataset evaluation with DexYCB shows that combining TACO with DexYCB for training CMR and MobRecon improves hand pose accuracy on both datasets.

## Evaluation Protocol

Sequences are divided into a training set and four test sets that factor out generalization axes: S1 (no generalization; tool geometries and triplets seen in training), S2 (geometry-level; novel tool geometry within seen triplets), S3 (triplet-level; novel tool-action-object triplets), and S4 (compound; novel geometry and novel triplet), with a train-to-test ratio of 4:1:1:1.5:2.5. Compositional action recognition is benchmarked with AIM and CACNF reporting Top-1/Top-5 accuracy per test set. Hand-object motion forecasting conditions on 10 observed frames and predicts 10 future frames for both hands, the tool, and the target object, with baselines InterVAE, MDM, InterRNN, and CAHMP scored by hand joint error (Je), object translation error (Te), and rotation error (Re). Cooperative grasp synthesis is evaluated with ContactGen, HALO-VAE, and an environment-ablated HALO-VAE-variant, using penetration volume, contact ratio, collision ratio, and FID on familiar (S1, S3) and unseen (S2, S4) geometries. Interaction field estimation predicts six distance fields between the left hand, right hand, tool, and target object from an RGB image with InterField-SF-based baselines, scored by mean distance error and acceleration error.

## Findings and Analysis

Action recognition is strong on S1 (CACNF 86.15% Top-1) and nearly unchanged on S2, showing that models already generalize across tool geometries within a category, but drops sharply on S3 (63.02%) and collapses on S4 (44.00%), demonstrating that novel compositions of tools, actions, and objects, especially their combination, are the real bottleneck. In motion forecasting, the tool and the hand holding it are the hardest elements to predict, and the two generative baselines (MDM, InterVAE) yield significantly larger errors than the predictive ones, which the authors attribute to the fast, complex motion distribution of tool manipulation; CAHMP performs best among the four. In grasp synthesis, conditioning on the environment (the full HALO-VAE) materially improves contact and collision ratios over the ablated variant, and performance declines on unseen geometries. The interaction field benchmark quantifies how bimanual, multi-object contact structure differs from prior single hand-object settings.

## Contributions

A large-scale bimanual tool-action-object dataset with 131 triplets, 196 scanned tool instances, and dense multi-modal capture; a marker-based annotation pipeline delivering MANO hand poses, 6D object poses, cleaned meshes, and marker-free images; a compositional generalization protocol with four test sets that separate geometry-level from triplet-level and compound generalization; and four benchmark tasks with baselines spanning recognition, forecasting, grasp synthesis, and interaction field estimation.

## Limitations

The paper acknowledges that articulated objects are not covered, so tool parts with moving joints remain outside the annotation; scene diversity is limited compared to in-the-wild recordings because data are captured in a controlled studio; and marker removal, although effective (mIoU reduced from 63.8% to 11.1%), is imperfect, leaving residual artifacts in the released images.
