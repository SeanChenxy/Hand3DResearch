# Reconstructing Hand-Held Objects in 3D from Images and Videos

## Summary
> This work presents a framework that reconstructs hand-held objects in 3D by retrieving matching 3D object models from large-scale 3D asset databases (such as Objaverse) using visual similarity matching based on CLIP embeddings and 2D-3D correspondence, then refining the retrieved shapes to fit the observed hand-object configuration, effectively using the 3D shape database as the foundational prior for object reconstruction.

## 1. Problem and Setting
- **Task**: 3D reconstruction of objects being held by hands in images and videos by retrieving and adapting 3D models from a large asset library.
- **Input**: A single RGB image or short video clip showing a hand interacting with an object.
- **Output**: A 3D object mesh (from the retrieved database) aligned to the observed pose and refined to match the specific instance geometry.
- **Which HOI task**: Hand-held object reconstruction via shape retrieval. This is the canonical example of the "shape retrieval prior" approach in the Ch3 taxonomy.

## 2. Core Method
- **Key innovation**: Instead of generating or completing object shapes from scratch, this method retrieves the most similar 3D model from a large-scale 3D asset database (Objaverse, ShapeNet) and deforms it to match the observed object. The retrieval uses **CLIP-based cross-modal embeddings** to bridge the 2D visual appearance of the object with 3D model renderings.
- **How it works**: (1) The hand is segmented out (using an off-the-shelf hand segmentation model) to isolate the object region. (2) A CLIP embedding is computed from the cropped object image region. (3) This embedding is matched against a pre-computed database of CLIP embeddings from multi-view renderings of 3D models. (4) The top-K retrieved 3D models are evaluated for geometric fit against the observed hand pose and object silhouette. (5) The best-matching model is non-rigidly deformed and pose-aligned to the observation using differentiable rendering and silhouette/depth losses. (6) For video input, temporal consistency of the retrieved model identity is enforced across frames.
- **How FM prior is injected**: CLIP (a large vision-language model pre-trained on 400M image-text pairs) provides the semantic-visual embedding space that enables open-vocabulary retrieval of 3D models from 2D observations. The 3D asset database (Objaverse) provides the shape prior itself.

## 3. Knowledge, Supervision, and Assumptions
- **Which FM prior**: CLIP (Contrastive Language-Image Pre-training) for visual-semantic embedding used in retrieval.
- **How used**: CLIP encodes both the query (cropped object image) and the database (renderings of 3D models) into a shared embedding space, enabling similarity-based retrieval.
- **Domain knowledge**: Hand parametric model (MANO) for pose estimation and physical constraint enforcement; known 3D asset database.
- **Training data**: CLIP is pre-trained; the retrieval database is constructed from Objaverse/ShapeNet. No HOI-specific training required.

## 4. Experiments and Findings
- **Datasets**: HO3D, DexYCB, in-the-wild images and videos from the web.
- **Key metrics**: Retrieval accuracy (top-1, top-5 match rate), Chamfer distance after deformation, and visual similarity between ground truth and reconstructed object.
- **Main quantitative results**: The retrieval+deformation approach achieves competitive shape accuracy compared to generative/completion methods, with the advantage of producing semantically identifiable and complete 3D models (not just geometry).
- **Evidence of FM prior gain**: CLIP-based retrieval significantly outperforms purely geometric (e.g., silhouette matching) retrieval baselines, demonstrating the value of vision-language semantic knowledge for bridging 2D observations to 3D models.

## 5. Strengths and Limitations
### Strengths
- Retrieved models are complete, semantically labeled, and have clean topology (unlike generated shapes).
- The retrieval database can be easily updated with new 3D models.
- Works with single images, not just video.
- CLIP embeddings provide open-vocabulary generalization beyond fixed object categories.
- Deformation step adapts the retrieved model to the specific instance.

### Limitations
- Relies on the object being present in the retrieval database; novel or rare objects may lack matches.
- CLIP embeddings can fail for heavily occluded or textureless objects.
- Deformation may introduce artifacts if the retrieved model topology differs significantly from the actual object.
- Retrieval quality degrades with increasing database size due to nearest-neighbor approximation.
- Cannot handle objects with parts unseen in the database.

## 6. Takeaway
This work exemplifies the "shape retrieval prior" paradigm: rather than generating shape from scratch using a learned prior (diffusion, 3D generative models), it retrieves from an explicit database of 3D models and uses CLIP embeddings to bridge the 2D-3D modality gap. This approach trades the flexibility of generative methods for the reliability and quality of database-derived shapes. It is particularly compelling for applications where object identity matters (e.g., AR/VR, robotics) and the retrieved model's semantic category information can be leveraged downstream.
