# Web2Grasp: Learning Functional Grasps from Web Images of Hand-Object Interactions

**Authors:** Hongyi Chen, Yunchao Yao, Yufei Ye, Zhixuan Xu, Homanga Bharadhwaj, Jiashun Wang, Shubham Tulsiani, Zackory Erickson, Jeffrey Ichnowski  
**Date:** 2025-05-13  
**Identifier:** [arXiv:2505.05517](https://arxiv.org/abs/2505.05517); DOI `10.48550/arXiv.2505.05517`  
**Zotero item:** `D86WB9EK` ([Zotero](zotero://select/library/items/D86WB9EK))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Web2Grasp learns functional multi-finger grasping—grasps that enable an object's intended use, such as placing a finger on a spray-bottle trigger—from web images of human hand-object interactions instead of costly teleoperated demonstrations. It reconstructs human HOI meshes from RGB images, retargets the hand to a robot hand, aligns the noisy reconstructed object mesh with an accurate text-to-3D shape, and trains an interaction-centric DRO grasp model on this low-quality but inexpensive data. An IsaacGym self-augmentation loop that collects disturbance-stable successes lifts simulation success from 61.8% to 83.4% over seen and unseen objects, surpasses baselines by 6.7% success with 1.8x functionality ratings, and transfers to a real LEAP Hand at an 85% success rate.

## Background and Problem
Functional grasping requires diverse, task-specific hand poses (e.g., fingers on a power drill's trigger), whereas most dexterous grasping work targets power grasps that merely hold objects, and functional approaches depend on expensive human-collected datasets or teleoperation. Web images naturally depict functional human-object interactions, but reconstructed HOI from them is noisy—mutual occlusion yields improper contacts such as fingers inside a drill rather than on its surface. The task is to convert such inexpensive, imperfect HOI reconstructions into training supervision for a multifinger grasping model that generalizes to unseen object categories.

## Method
The pipeline crawls web images per category and reconstructs HOI with Ye et al.'s pretrained model: MANO hand pose via FrankMocap and an object SDF conditioned on the hand pose. Hand poses are retargeted to the robot (ShadowHand in simulation, LEAP Hand for the real robot) with AnyTeleop position-based keypoint optimization, while the low-quality reconstructed object mesh is aligned by ICP to an accurate category mesh generated from the object name with Meshy AI or Genie. The interaction-centric DRO model (a CVAE predicting a dense point-to-point distance matrix between robot and object point clouds, followed by multilateration and optimization for joint configurations) is trained on the reconstructed data, chosen because it tolerates imperfect labels. A three-step simulation loop then deploys the web-trained model in IsaacGym, retains grasps that stay stable under force disturbances (object displacement below 2 mm), and retrains on the accumulated physically feasible successes, expanding both dataset size and object coverage.

## Contributions
- A Web2Grasp pipeline that turns reconstructed HOI from web images into training data for functional grasping models on multifinger robot hands.
- Simulator-augmented dataset expansion that filters penetration and unstable contacts while preserving learned functionality.
- Simulation and real-world evaluations showing high success and functionality on challenging, underrepresented objects such as syringes, pens, spray bottles, and tongs, plus released reconstructed HOI datasets.

## Experimental Setup
The web dataset covers 10 object categories (Power Drill, Pen, Microphone, Phone, Spray Bottle, Wine Glass, Tong, Syringe, Mug, Sword) with 100 reconstructions each; metric filtering plus human inspection keeps 9.6%. Training uses ShadowHand in IsaacGym; evaluation runs 100 trials per object on the 10 seen and 9 unseen categories (Whip, Teapot, Axe, Remote, Torch, Hammer, Whisk, Soap Bottle, Writing Brush), collecting 200 simulator successes per object for augmentation. Baselines are GenDexGrasp, DexGraspNet, and DRO. Real experiments use a uFactory xArm7 with a LEAP Hand over 8 objects, 10 trials each.

## Results
The web-only model reaches 75.8% success on seen and 61.8% across all objects in simulation; simulator augmentation raises the overall rate to 83.4% (e.g., Teapot 12% to 99%). On unseen objects it attains 46.3% success and a 47.7% human-voted functionality score versus 39.6%/19.6% for DexGraspNet, 33.4%/26.3% for DRO, and 20.6%/6.4% for GenDexGrasp, with visible transfer such as Pen to Writing Brush and Spray Bottle to Soap Bottle. In the real world it averages 85% success across eight objects, versus at most 2.5/10 average for baselines, after adding grip tape to the LEAP Hand fingers to increase friction for the Spray Bottle and Syringe.

## Limitations
The authors report that reconstructed HOI exhibits hand-object penetration and occasionally misplaced contacts (e.g., an index finger between tong prongs), object alignment can enlarge meshes and worsen penetration, and reconstruction fails for complex interaction modes such as buckets with thin handles or scissors requiring fingers inside handle holes.
