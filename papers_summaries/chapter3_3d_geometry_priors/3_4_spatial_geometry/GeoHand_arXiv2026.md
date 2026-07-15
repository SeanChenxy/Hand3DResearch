# GeoHand: Unlocking Prior Geometry Knowledge for Monocular 3D Hand Reconstruction

## Summary
GeoHand unlocks high-quality geometric priors from a frozen foundational monocular geometry estimator (MoGe2) and adapts them via a GeoAdapter and gated cross-modal token fusion, then refines articulation with a Keypoint-Queried Iterative Refiner (KQIR), achieving state-of-the-art hand reconstruction under severe occlusions and hand-object interactions on FreiHAND, DexYCB, and HO3Dv3.

## 1. Problem and Setting
- Monocular 3D hand reconstruction under severe self-occlusion and hand-object interactions.
- Input: single RGB image, optionally with monocular depth as auxiliary input.
- Output: MANO hand mesh parameters, 3D hand joints, mesh vertices.
- Task: hand-only reconstruction; classified here under 3D geometry priors (MoGe2 provides spatial geometric priors).

## 2. Core Method
- Key innovation: unlock a frozen foundational monocular geometry estimator (MoGe2) to provide spatial geometric priors, then adapt them specifically for hand reconstruction.
- A map-level GeoAdapter recalibrates the general scene-oriented geometric features for detailed hand reconstruction.
- A gated cross-modal token fusion strategy integrates the adapted geometric priors with intrinsic RGB appearance cues without overwhelming them.
- A Keypoint-Queried Iterative Refiner (KQIR) uses projected joint locations to query geometry-aware image features for spatial correction, ensuring precise local articulation.
- How FM prior is injected: MoGe2 (a foundation monocular geometry model) provides dense spatial priors, adapted to the hand domain via learned modules.

## 3. Knowledge, Supervision, and Assumptions
- Foundation model: MoGe2 (monocular geometry estimator).
- Domain knowledge: MANO parametric hand model; per-pixel geometric features from MoGe2.
- Training data: FreiHAND, DexYCB, HO3Dv3 (for evaluation); pre-training uses general scene geometry datasets.
- Assumption: MoGe2's general scene geometry transfers to hands with adapter-based domain adaptation.

## 4. Experiments and Findings
- Datasets: FreiHAND, DexYCB, HO3Dv3.
- Metrics: PA-MPJPE, PA-MPVPE, F-score, with breakdowns under occlusion.
- Achieves state-of-the-art performance, especially under severe occlusions and hand-object interactions.
- The combination of global geometric disambiguation (MoGe2 + GeoAdapter) and local refinement (KQIR) provides complementary benefits.

## 5. Strengths and Limitations
### Strengths
- Effective use of foundation geometry model (MoGe2) to resolve depth ambiguity in hand reconstruction.
- Gated fusion balances geometric and appearance cues adaptively.
- Strong results under severe occlusions and HOI scenarios.
- Modular design with reusable FM components.

### Limitations
- Hand-only; no object reconstruction or interaction modeling.
- Depends on MoGe2's quality for general scene geometry.
- Adapter and refiner need training on hand data; not fully zero-shot.
- The geometric prior may not generalize to unusual hand poses far from training distribution.

## 6. Takeaway
GeoHand demonstrates that high-quality spatial geometric priors from a foundation monocular geometry model (MoGe2) can be unlocked for hand reconstruction through a carefully designed adapter and fusion strategy, especially valuable under severe occlusions. The work exemplifies the "geometry-as-prior" paradigm: leveraging off-the-shelf foundation models not for end-to-end prediction, but as spatial cues that complement appearance-based reasoning in the specialized hand domain.
