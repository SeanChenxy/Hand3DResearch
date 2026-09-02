# DexUMI: Using Human Hand as the Universal Manipulation Interface for Dexterous Manipulation

**Authors:** Mengda Xu, Han Zhang, Yifan Hou, Zhenjia Xu, Linxi Fan, Manuela Veloso, Shuran Song  
**Date:** 2025-10-02 (CoRL 2025)  
**Identifier:** [arXiv:2505.21864](https://arxiv.org/abs/2505.21864); DOI `10.48550/arXiv.2505.21864`  
**Zotero item:** `HMWGACS8` ([Zotero](zotero://select/library/items/HMWGACS8))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
DexUMI is a data collection and policy learning framework that uses the human hand itself—rather than teleoperation—as the interface for teaching dexterous manipulation to diverse robot hands. A per-robot wearable hand exoskeleton bridges the kinematics gap by constraining human motion to feasible robot actions while providing direct haptic feedback and precise encoder readings, and a software pipeline bridges the visual gap by segmenting the human hand out of wrist-camera videos, inpainting the background, and compositing a replayed robot hand with correct occlusions. Policies trained on the processed demonstrations reach an average 86% success rate across four real-world tasks on two different robot hands and collect data 3.2 times faster than teleoperation.

## Background and Problem
Dexterous robot hands differ widely in degrees of freedom, actuation, and size, and transferring skills from human hands is hard due to gaps in kinematics, contact geometry, tactile information, and visual appearance. Teleoperation suffers from spatial observation mismatch and lacks direct haptic feedback, while retargeting struggles with morphological differences such as thumb flexibility. The task is to let a human manipulate objects directly, wearing an exoskeleton, and convert those demonstrations into training data whose observations and actions match what a target robot hand would experience, without the robot being present during collection.

## Method
The hardware adaptation is an exoskeleton whose design is optimized per robot hand: a bi-level optimization maximizes similarity between exoskeleton and robot fingertip workspaces in SE(3) while constraining wearability (e.g., moving the thumb backward to avoid collision). It integrates resistive joint encoders at every actuated joint (with a per-joint regression to robot motor values), an iPhone ARKit tracker for 6-DoF wrist pose, an OAK-1 150-degree wide-angle camera rigidly mounted under the wrist in the same pose as on the robot hand, and tactile sensors identical to the target hand's (FSR for Inspire, electromagnetic array for XHand). The software adaptation segments the hand and exoskeleton with SAM2, inpaints the background with ProPainter, records the robot hand replaying the joint actions, and composes final observations using an occlusion-aware mask intersection. A diffusion policy with DINOv2 visual features and tactile input predicts relative end-effector and finger actions.

## Contributions
- A hardware adaptation framework that automatically designs wearable exoskeletons matching each robot hand's fingertip kinematics, enabling robot-free data collection with haptic feedback and precise joint capture.
- A software adaptation pipeline that converts human demonstration videos into visually consistent robot-hand observations with correct occlusion relationships.
- Comprehensive real-world validation on underactuated and fully-actuated hands across precise, contact-rich, and long-horizon tasks.

## Experimental Setup
Two robot hands are used: the Inspire Hand (12 DoF, six active, underactuated) and the XHand (12 active DoF, fully actuated). Four real tasks are evaluated: Cube Pick and Place, Egg Carton Opening, Tea Picking with Tool (both hands), and a four-stage Kitchen task on the XHand (turn off knob, move pan, pick up salt, sprinkle salt). Ablations compare relative versus absolute finger actions, with versus without tactile input, and inpainted versus masked or raw visuals. Each task uses 20 evaluation episodes with matched initial states; training data range from 175 (Egg Carton) to 400 (Tea) trajectories plus 470 for Kitchen.

## Results
The full DexUMI configuration achieves 1.00/0.85/1.00/0.85 (Cube/Carton/Tea tool/Tea leaf) on the Inspire Hand and 1.00/0.85/0.95/0.95/0.75 (Tea tool/Tea leaf/knob/pan/salt) on the XHand, an average success of 86%. Absolute finger actions degrade sharply (e.g., 0.00 on XHand tea leaf and salt), and removing software adaptation hurts most (raw visuals give 0.20 and 0.05 on Cube and Carton). Tactile input helps the visually blocked salt task (fingers insert into the salt before closing) but can hurt on noisy sensors unless relative actions are used. On the tea task, DexUMI collects 3.2 times more successful demonstrations per 15-minute session than teleoperation.

## Limitations
The authors note the exoskeleton requires hardware-specific tuning per hand, matches only fingertip workspaces (not the palm), and 3D-printed links can deform so encoders miss distortion; tactile sensors drift under the stronger human hand forces; the software pipeline still needs real robot hardware for hand images, cannot fully reproduce illumination, and requires a fixed wrist camera; and current robot hands lack precision due to backlash and friction, with hand-size discrepancies limiting wearability.
