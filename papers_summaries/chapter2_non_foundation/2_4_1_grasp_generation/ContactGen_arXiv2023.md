# ContactGen: Generative Contact Modeling for Grasp Generation

## Summary
Proposes ContactGen, an object-centric contact representation (contact map + hand-part map + direction map) generated via a conditional diffusion model, which then guides MANO parameter optimization to produce diverse and accurate grasps.

## 1. Problem and Setting
- Task: given a 3D object mesh or point cloud, generate diverse and physically plausible static hand grasps.
- Input: 3D object shape; Output: MANO hand parameters achieving realistic contact with the object.
- Key challenge: directly regressing MANO parameters from object shape yields limited diversity; contact-based methods enable diversity but need an expressive, learnable contact representation that captures where, with which hand part, and in what direction the hand should contact.

## 2. Core Method
- ContactGen representation: for each point on the object surface, predict three components: (a) a contact map (binary: is this point contacted by the hand?), (b) a part map (which hand part — palm, thumb, index, etc. — contacts it?), and (c) a direction map (surface normal direction of the contacting hand part).
- Diffusion model: a 3D point-wise diffusion model (DDPM) trained to generate ContactGen maps conditioned on object geometry (encoded via a point-wise transformer).
- Grasp fitting: given generated ContactGen maps, optimize MANO parameters by minimizing an energy that attracts corresponding hand vertices to contact points along predicted directions, with penalties for penetration and joint violation.
- Key innovation: the combination of spatial (where), semantic (which part), and directional (how oriented) contact channels provides a richer, more constraining contact prior than binary contact alone.

## 3. Knowledge, Supervision, and Assumptions
- Training data: GRAB dataset (real human grasps), ObMan (synthetic), and a custom ContactPose dataset.
- Supervision: ground-truth ContactGen maps derived from fitted MANO-object mesh pairs in the training data (contact via proximity, hand part via nearest MANO part segmentation, direction via hand vertex normals).
- Domain knowledge: MANO hand model with part segmentation.
- Assumption: static grasp; rigid, known object mesh.

## 4. Experiments and Findings
- Datasets: GRAB, ObMan, ContactPose.
- Metrics: contact IoU, part accuracy, penetration depth, grasp diversity (coverage), and user preference study.
- Main findings: ContactGen significantly outperforms prior contact-free and simple-contact-based methods on all metrics; the three-channel representation reduces ambiguities in grasp fitting compared to binary-contact-only methods; diffusion-based generation yields diverse grasps that cover the feasible contact manifold.

## 5. Strengths and Limitations
### Strengths
- Rich three-channel contact representation resolves ambiguities that binary contact cannot.
- Diffusion prior naturally captures the multi-modal distribution of plausible contacts.

### Limitations
- Requires per-point hand-part annotation on the object (computed from data but not always available).
- Grasp fitting step is slow (iterative optimization) and can produce failures if diffusion generates inconsistent contact maps.
- GRAB-centric training data limits object diversity.

## 6. Takeaway
ContactGen demonstrates that a richer contact representation (where + which part + which direction) enables far more constrained and accurate grasp fitting than binary contact alone, and that diffusion models are well-suited for capturing the multi-modal nature of plausible human-object contact. The ContactGen representation has become a key reference point for contact-driven grasp generation.
