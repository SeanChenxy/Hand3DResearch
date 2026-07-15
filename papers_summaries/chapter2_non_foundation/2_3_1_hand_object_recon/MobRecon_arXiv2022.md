# MobRecon: Mobile-Friendly Hand Mesh Reconstruction from Monocular Image

## Summary
MobRecon is a mobile-friendly hand mesh reconstruction framework that simultaneously achieves high reconstruction accuracy, fast inference speed, and temporal coherence via lightweight stacked structures for 2D encoding, depth-separable spiral convolution for 3D decoding, and a novel MapReg feature lifting module, reaching 83 FPS on Apple A14 CPU.

## 1. Problem and Setting
- Single-view hand mesh reconstruction from a monocular RGB image.
- Input: monocular RGB image (or video).
- Output: 3D MANO hand mesh (pose, shape, vertices).
- Hand-only reconstruction; designed for mobile deployment (real-time on mobile CPUs).

## 2. Core Method
- A framework achieving high accuracy, fast inference, and temporal coherence simultaneously.
- Key components:
  - Lightweight stacked structures for 2D encoding (efficient backbone).
  - Depth-separable spiral convolution for 3D decoding (efficient mesh operation).
  - MapReg feature lifting module: combines heatmap encoding and position regression paradigms; followed by pose pooling and pose-to-vertex lifting to transform 2D pose features to 3D vertex features.
- Temporal smoothing ensures coherence in video inference.
- How the method differs from prior work: holistic optimization for the speed-accuracy-temporality triple; depth-separable spiral convolutions are mobile-friendly.

## 3. Knowledge, Supervision, and Assumptions
- Training data: hand mesh datasets (FreiHAND, RHD, HO3D).
- Supervision: 3D hand mesh labels (MANO), 2D keypoint labels.
- Uses MANO for hand parametric model.
- Fully supervised; the temporal smoothing is applied at test time.
- Key assumption: depth-separable convolutions on the mesh graph are sufficiently expressive for hand mesh.

## 4. Experiments and Findings
- Datasets: FreiHAND, RHD, HO3Dv2.
- Metrics: PA-MPJPE, PA-MPVPE, F-score (accuracy); FPS on Apple A14 (speed); temporal coherence metrics.
- Achieves superior accuracy and temporal coherence compared to prior methods.
- 83 FPS on Apple A14 CPU enables real-time mobile deployment.

## 5. Strengths and Limitations
### Strengths
- Simultaneously achieves high accuracy, fast speed, and temporal coherence.
- Mobile-friendly (real-time on Apple A14).
- Lightweight yet effective design.
- Code publicly available.

### Limitations
- Hand-only; no object reconstruction.
- Mobile optimization trades off some accuracy compared to heaviest server-side models.
- Relies on MANO; cannot represent non-MANO hand details.
- May not handle extreme in-the-wild scenarios.

## 6. Takeaway
MobRecon demonstrates that careful architectural design (lightweight 2D encoding + depth-separable spiral 3D decoding + MapReg lifting) can achieve a strong speed-accuracy-coherence triple for hand mesh reconstruction, suitable for real-time mobile deployment.
