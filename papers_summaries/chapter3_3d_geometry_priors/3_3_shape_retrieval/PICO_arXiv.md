# PICO: Reconstructing 3D People In Contact with Objects

## Summary
PICO tackles 3D human-object interaction (HOI) reconstruction from single color images in natural, in-the-wild settings by introducing PICO-db — a new dataset of natural images uniquely paired with dense 3D contact on both body and object meshes — and PICO-fit, a render-and-compare fitting method that recovers 3D body and object meshes in interaction, enabling HOI understanding to scale in the wild across many object categories.

## 1. Problem and Setting
- Recovering 3D human-object interaction (HOI) from single color images, where past work requires controlled settings (known object shapes and contacts) and tackles only limited object classes.
- Input: single color image of a person interacting with an object.
- Output: 3D body mesh (SMPL-X) and 3D object mesh in interaction, with contact correspondences.
- Task: hand-object (and full-body) interaction reconstruction with shape retrieval (PICO-db provides the retrieved 3D object prior).

## 2. Core Method
- PICO-db (dataset): natural images paired with dense 3D contact on both body and object meshes. Image-object pairs are built by retrieving an appropriate 3D object mesh from a database using vision foundation models, then projecting DAMON's body contact patches onto the object via a novel method requiring only 2 clicks per patch.
- PICO-fit (method): a render-and-compare fitting pipeline that infers contact for the SMPL-X body, retrieves a likely 3D object mesh and contact from PICO-db, and uses the contact to iteratively fit the 3D body and object meshes to image evidence via optimization.
- How FM priors are injected: vision foundation models (e.g., DINOv2, CLIP) are used for image-based 3D object retrieval from the database; the retrieved object provides the shape prior.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: vision foundation models for object retrieval.
- Domain knowledge: SMPL-X body model, 3D contact modeling, render-and-compare optimization.
- Training data: PICO-db (new dataset) is the primary supervision source; minimal human input (2 clicks per patch) for the contact projection.
- Assumption: object category is approximately retrievable from the database; contact patterns transfer between instances.

## 4. Experiments and Findings
- Datasets: PICO-db (introduced); evaluation on in-the-wild natural images.
- Metrics: 3D body and object mesh accuracy, contact prediction accuracy, cross-category generalization.
- PICO-fit works well for many object categories that no existing method can tackle, enabling HOI understanding to scale in the wild.
- The contact correspondences (body ↔ object) provide strong supervision that bridges the gap between body-only and object-only reconstruction.

## 5. Strengths and Limitations
### Strengths
- First method to handle many object categories in in-the-wild images without controlled settings.
- PICO-db provides a unique resource with body-and-object contact annotations.
- Minimal human input (2 clicks per patch) for the contact projection makes the dataset scalable.
- Generalizes to natural images and novel object classes.

### Limitations
- Depends on the object being represented in the retrieval database.
- The contact-based optimization is iterative and may be slow.
- The 2-clicks-per-patch annotation still requires some human effort.
- Requires SMPL-X body estimation, which may be inaccurate for novel poses.

## 6. Takeaway
PICO tackles the long-standing problem that HOI reconstruction requires controlled settings, by combining a new dataset (PICO-db) with dense body-object contact and a render-and-compare fitting method (PICO-fit) that generalizes to many object categories in the wild. The work demonstrates the value of using vision foundation models for object retrieval and the importance of explicit contact modeling for human-object interaction understanding.
