# Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items

# Paper Summary

## Summary
Google Scanned Objects (GSO) is a curated collection of over 1,000 high-quality 3D-scanned common household items, captured with a custom structured-light pipeline, and preprocessed into watertight textured meshes that are directly usable in popular physics simulators (Ignition Gazebo, Bullet) for robotics, simulation, and synthetic perception research.

## 1. Problem and Setting
- **Task**: A data resource, not a method — provides 3D-scanned meshes of common household items for use in interactive simulation, sim-to-real perception, and robotic learning.
- **Input/Output**: Real household objects → custom 3D scanning pipeline → watertight, decimated, textured SDF meshes usable in simulators; output is the released SDF/OBJ mesh library.
- **Difficulties addressed**:
  - Synthetic CAD models (ShapeNet, IKEA Models) are stylized and do not represent real geometry; 3D-Scan alternatives (YCB: 77 objects, BigBIRD: 125, KIT: 145) are small and often missing object bases.
  - Public web 3D resources rarely contain watertight, textured, physically-plausible models at scale.
  - Sim-to-real transfer for grasping / manipulation needs high-fidelity object geometry that is not "fixed background scenery".

## 2. Core Method
**Pipeline**: Real household object → custom 3D scanning rig (two machine-vision cameras + DSLR + projector) → structured-light scanning + sub-pixel calibration → raw scan meshes → mesh merging → mesh alignment → mesh decimation → QA curation → SDF / OBJ / mesh release (CC BY 4.0).

**Key components**:
1. **Custom scanning hardware**: A lighting-controlled enclosure with two machine-vision cameras for stereo shape detection, a DSLR for high-resolution HDR color, and a computer-controlled projector for gray-codes and stripe patterns. Scanhead rotates on a turntable.
2. **Structured-light + sub-pixel pipeline**: Projector-camera synchronization, HDR image reconstruction, stripe center finding, sub-pixel camera coordinates, then mesh generation from stereo pairs.
3. **Mesh merging / alignment / decimation**: Raw per-camera scans are calibrated, aligned, merged into a single closed-manifold mesh, decimated for tractable simulation, and texture-mapped.
4. **Quality-assurance curation**: Each scanned model goes through a manual / automated QA pass — closed-manifold meshes are kept; optically uncooperative objects (highly reflective, transparent) that yield invalid meshes are removed.
5. **Simulator-ready release**: Output SDF models are usable in Ignition Gazebo and Bullet; 17 categories of common household items (1,030 objects).

## 3. Knowledge, Supervision, and Assumptions
- **Source**: Real household objects scanned in-house.
- **Supervision**: 3D reconstruction is geometry-driven (calibrated stereo + structured light); no ML training. QA is human-in-the-loop.
- **Foundation-model usage**: Not a method paper. The dataset is intended to be used downstream as object assets in simulation for grasping, navigation, and sim-to-real training.
- **Assumptions**:
  - Structured-light scanning with sub-pixel calibration produces meshes of sufficient quality for simulation.
  - A lighting-controlled enclosure and HDR imaging handle a wide range of object reflectance.
  - Watertight decimated meshes are sufficient proxy geometry for physics simulation.
- **Learned vs. provided**: All 3D geometry is captured by the scanning rig; nothing is learned. Object categories, instance counts, and per-model artifacts are provided as metadata.

## 4. Experiments and Findings
- **Datasets used for context comparison (not benchmarks)**: KIT (145), BigBIRD (125), YCB (77), ShapeNet (51K), 3D-FUTURE (16K), ABO (8K) — see Table I.1.
- **Metrics**:
  - Dataset size: 1,030 objects across 17 categories (year 2021/2022 release).
  - Coverage: chosen to maximize household-object diversity within a manageable scanning budget.
- **Key results stated**:
  - GSO contains over 1,000 unique 3D-scanned household items — an order of magnitude larger than YCB, BigBIRD, and KIT while preserving watertight topology suitable for physics simulation.
  - First scanned dataset to explicitly include the object base on every item (KIT and BigBIRD do not).
  - Models are released under CC BY 4.0 and are simulator-ready (SDF) out of the box.
  - The paper also surveys GSO's downstream impact (Interactive Gibson benchmark uses GSO; grasping pipelines train on GSO; ABO + GSO jointly benchmark differentiable stereopsis).
- **Ablations / experiments**: The paper is primarily a dataset/system paper; quantitative vision-task benchmarks are deferred to downstream work (e.g., Interactive Gibson, Contact-GraspNet, NVIDIA's differentiablestereopsis shape recon benchmark).

## 5. Strengths and Limitations
### Strengths
- **Simulator-ready meshes**: SDF meshes usable in Ignition Gazebo and Bullet with no further preprocessing — closes the gap between research and deployment in robotics.
- **Order-of-magnitude scale up over prior scanned datasets**: 1,030 objects vs. YCB (77) / BigBIRD (125) / KIT (145).
- **Watertight with object bases**: Unlike KIT and BigBIRD, GSO captures the bottom of every object, important for placing objects stably in simulation.
- **Diverse household items**: 17 categories spanning food items, toys, kitchenware, tools, etc.
- **Open license (CC BY 4.0)** and integrated QA pipeline.

### Limitations
- **Modest size by CAD-dataset standards**: 1,030 objects is much smaller than ShapeNet (51K), 3D-FUTURE (16K), or ABO (8K).
- **Optically uncooperative objects fail**: Reflective, transparent, or dark objects can yield invalid meshes; some are removed in QA.
- **No articulated objects**: All meshes are rigid; joints, hinges, and moving parts are not modeled.
- **No multi-view images included**: Unlike ABO, GSO does not provide paired real catalog / multi-view images of the same object.
- **Texture assumes mostly diffuse**: Highly specular objects are scanned with limitations even with HDR imaging.
- **Scan noise remains**: Even with QA, small-scale surface noise, reconstruction artifacts at concave regions, and decimation-induced detail loss are inherent to the structured-light pipeline.

## 6. Takeaway
Google Scanned Objects establishes that **high-quality, watertight, simulator-ready 3D scans of real household objects can be collected at scale using a custom structured-light pipeline**, providing the robotics and 3D-vision communities with a CC-licensed asset library that is an order of magnitude larger than prior scanned datasets (YCB, BigBIRD, KIT) and is uniquely suited to interactive simulation, sim-to-real grasping, and synthetic-perception research. For HOI research, GSO meshes serve as a high-fidelity prior candidate set when the hand-held object in a video can be matched to a GSO scan, providing a topologically-stable and physically-plausible shape prior for reconstruction pipelines.