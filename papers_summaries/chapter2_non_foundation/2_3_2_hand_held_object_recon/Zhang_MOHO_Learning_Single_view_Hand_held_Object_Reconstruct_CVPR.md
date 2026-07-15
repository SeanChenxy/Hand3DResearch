# MOHO: Learning Single-view Hand-held Object Reconstruction with Multi-view Occlusion-Aware Supervision

## Summary
Leverages multi-view supervision from short video clips to train a single-view hand-held object reconstruction network, using occlusion-aware rendering to handle the hand-object overlap.

## 1. Problem and Setting
- Single-view 3D reconstruction of a hand-held object given one RGB image.
- Input: single RGB image; output: 3D hand mesh (MANO) + 3D object shape (implicit SDF).
- Template-free object reconstruction. Uses multi-view video data only during training to provide richer 3D supervision.

## 2. Core Method
- Two-stage training strategy: (1) use multi-view frames from short video clips to reconstruct per-frame object SDFs via differentiable rendering and multi-view consistency; (2) train a single-view feed-forward network with these multi-view reconstructions as pseudo-ground-truth.
- Occlusion-aware rendering: when computing photometric losses from novel views, the method explicitly models hand-object occlusion relationships, preventing the hand from "bleeding" into the object reconstruction and vice versa.
- The reconstruction network takes an RGB image, predicts MANO parameters and object SDF values (in a hand-aligned coordinate frame), and is trained with both direct 3D losses and multi-view rendering losses.
- Uses a conditional NeRF-like volumetric rendering for the object.

## 3. Knowledge, Supervision, and Assumptions
- Training data: short video clips from HO3D, HOI4D, DexYCB.
- Supervision: multi-view photometric consistency, 2D/3D hand keypoints, depth maps, and object masks.
- Uses MANO for hand.
- Assumes at training time, short video sequences showing the object from multiple views are available; object is rigid.

## 4. Experiments and Findings
- Datasets: HO3D, HOI4D, DexYCB.
- Metrics: Chamfer Distance, F-score for object; MPJPE for hand.
- Multi-view occlusion-aware training significantly improves single-view reconstruction, particularly for occluded object regions. Outperforms prior single-view methods (IHOI, AlignSDF) on real datasets.

## 5. Strengths and Limitations
### Strengths
- Clever use of multi-view video during training to extract richer 3D supervision without manual 3D annotations.
- Occlusion-aware rendering improves hand-object separation.
- Single-view inference is fast (feed-forward).

### Limitations
- Two-stage training pipeline is complex to implement.
- Multi-view pseudo-ground-truth may contain errors that propagate to the single-view network.
- Still struggles with objects that have thin structures or specular surfaces.
- Requires video data during training.

## 6. Takeaway
MOHO demonstrated that multi-view cues (from video) can be distilled into a single-view model through careful occlusion-aware training. This "train on video, test on single image" paradigm is highly practical since video data is easier to collect than 3D annotations, while deployment scenarios often require single-image inference.
