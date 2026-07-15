# ContactArt: Learning 3D Interaction Priors for Category-level Articulated Object and Hand Poses Estimation

## Summary
A method that learns 3D interaction priors for category-level articulated objects, jointly estimating hand pose and articulated object state (joint angles) from a single RGB image by leveraging contact-based interaction reasoning.

## 1. Problem and Setting
- Joint estimation of hand pose and articulated object state from a single RGB image, for category-level objects (e.g., different laptop models, different drawer types).
- Input: single RGB image of a hand interacting with an articulated object. Output: hand pose (MANO) and the object's articulation parameters (joint angles, part poses) and the 3D object model.
- Static image setting; both hand and articulated object.
- Category-level generalization: the model should work on object instances not seen during training, within known categories.

## 2. Core Method
- Interaction prior learning: a neural network trained to predict, from image features, both hand MANO parameters and object articulation parameters (e.g., revolute/prismatic joint angles).
- The interaction prior is encoded as a joint distribution over hand pose and object articulation, learned from data containing diverse interaction examples across object categories.
- Contact-based reasoning: the model explicitly reasons about where the hand contacts the object to infer articulation. For instance, hand position relative to a laptop lid suggests the opening angle.
- A shared image encoder extracts features, followed by category-specific decoders that predict articulation parameters and hand pose.
- During inference, the interaction prior constrains the joint prediction to be physically plausible (e.g., hand cannot be inside the articulated parts).

## 3. Knowledge, Supervision, and Assumptions
- Training data: datasets with annotated articulated object states and hand poses (e.g., ContactPose with articulated objects, or custom datasets).
- Supervision: 3D hand joint/vertex positions (MANO), object articulation angles, part 6D poses.
- Uses MANO for hand; articulated object model (e.g., URDF-like parameterization) with known kinematic structure per category.
- Key assumption: the object category and its kinematic structure are known; only the articulation parameters and specific instance geometry vary.
- Contact serves as a strong cue for articulation state inference.

## 4. Experiments and Findings
- Evaluated on datasets with articulated object interactions (laptops, drawers, scissors, etc.).
- Metrics: hand MPJPE, articulation angle error, part pose error.
- The interaction prior significantly improves articulation estimation compared to independent prediction baselines.
- Category-level generalization is demonstrated: the model can estimate articulation for unseen object instances within known categories.
- Ablation: contact-based reasoning contributes more to accuracy than purely visual features, confirming the importance of interaction modeling.

## 5. Strengths and Limitations
### Strengths
- Extends hand-object reconstruction to articulated objects, a significantly more complex setting than rigid objects.
- Category-level generalization reduces the need for per-instance object models.
- Contact-based interaction prior provides a physically grounded cue for articulation estimation.

### Limitations
- Requires known object kinematic structure per category; cannot handle completely novel articulation mechanisms.
- The number of supported articulation types is limited by training data diversity.
- Articulation estimation accuracy degrades under heavy occlusion or when the articulation joint is not visible.
- Single-image setting may be ambiguous for certain articulation states (e.g., a drawer could be partially open without visible cues).

## 6. Takeaway
ContactArt extended hand-object reconstruction to the significantly more challenging domain of articulated objects, introducing the concept of learned 3D interaction priors for category-level generalization. This work laid groundwork for subsequent research on hand-articulated-object interaction, which is critical for robotics and embodied AI applications where objects are not just rigid blocks but have functional moving parts.
