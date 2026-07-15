# Objaverse: A Universe of Annotated 3D Objects

# Paper Summary

## Summary
Objaverse is a large-scale dataset of 800K+ annotated 3D objects sourced from Sketchfab that enables breakthroughs in 3D vision, 2D long-tail recognition, and embodied AI through unprecedented scale, diversity, and richness of annotations.

## 1. Problem and Setting
- **Task**: Creating a large-scale, diverse, and richly annotated 3D object dataset to support training data-hungry deep learning models
- **Inputs**: 3D models from Sketchfab with Creative Commons licenses
- **Outputs**: A dataset containing 818K+ 3D objects with metadata including names, categories, tags, natural language descriptions, and animations
- **Difficulty**: Existing 3D datasets are severely limited in scale (tens of thousands at most), lack diversity within categories, and have limited annotations. This constrains progress in 3D vision compared to 2D vision which has benefited from massive datasets like ImageNet, LAION, and Conceptual Captions

## 2. Core Method
**Data Collection Pipeline**:
1. **Source Selection**: Objects sourced from Sketchfab using their public API
2. **Filtering**: Only models with distributable Creative Commons licenses are selected; models marked as restricted (objectionable/adult content) are excluded
3. **Metadata Inheritance**: Each object inherits foundational annotations from its creator including name, category assignments, tags, and natural language description
4. **LVIS Categorization** (Objaverse-LVIS subset): A 47K object subset where objects are assigned to one of 1,156 LVIS categories using CLIP classifier predictions and filtering

Key innovation: **Scalability through web sourcing** - leveraging an existing platform (Sketchfab) with 150K+ contributing artists rather than manual curation, enabling 16x more objects than ShapeNet

## 3. Knowledge, Supervision, and Assumptions
- **Training Data**: Uses Sketchfab as the data source - models created by over 150K artists
- **Pretrained Models Used**: CLIP classifier is used to categorize objects into LVIS categories for the Objaverse-LVIS subset
- **Annotations Provided**: 
  - Names from creators
  - 18 coarse categories from Sketchfab's scheme
  - Unrestricted tags from creators
  - Natural language descriptions from creators
  - Animations (where available)
- **Learned vs Provided**: The dataset itself provides the raw data and annotations; models trained on it (e.g., GET3D for generation) learn from this data
- **Assumptions**: Assumes Creative Commons licensing allows distribution and research use; assumes creator-provided metadata is sufficiently accurate for downstream tasks

## 4. Experiments and Findings
**Application 1: 3D Generative Modeling**
- Dataset: Objaverse subset used to train GET3D
- Finding: Generated objects rated by human annotators as more diverse than ShapeNet-trained models in 91% of cases

**Application 2: Long-tail Instance Segmentation**
- Dataset: LVIS benchmark (1,230 categories)
- Method: Copy+Paste augmentation using Objaverse assets
- Finding: Improves performance on tail categories compared to state-of-the-art segmentation methods (specific metrics not mentioned)

**Application 3: Robustness Benchmark**
- Dataset: Rendered Objaverse objects from random orientations
- Finding: State-of-the-art CLIP-style visual backbones show dramatic performance degradation when classifying objects from arbitrary views

**Application 4: Embodied AI - Object Navigation**
- Dataset: ProcTHOR simulated environments populated with Objaverse assets
- Finding: Enables open-vocabulary object navigation for 1.1K semantic categories (~50x increase over previous 108 categories)

## 5. Strengths and Limitations

### Strengths
- **Unprecedented Scale**: 818K objects, 16x larger than ShapeNet
- **Diverse Sources**: Objects from 150K+ artists across different 3D creation platforms, not limited to a single tool like ShapeNet's SketchUp-only models
- **Rich Annotations**: Natural language descriptions, tags, and animations enable multi-modal research
- **Realistic Quality**: Artist-designed and scanned objects with textures/materials, unlike CAD-only datasets
- **Legal Clarity**: Creative Commons licensing ensures free use for research

### Limitations
- **Metadata Noise**: Creator-provided names, categories, and tags have inherent noise and varying specificity
- **Coarse Native Categorization**: Sketchfab's 18-category scheme is too coarse for most applications, requiring additional categorization efforts
- **Rendering Required**: For 2D vision applications, objects must be rendered (computational cost)
- **Not Mentioned**: Quality filtering beyond licensing and content restrictions, potential class imbalance details

## 6. Takeaway
Objaverse demonstrates that **scaling 3D datasets to hundreds of thousands of diverse, artist-created objects with rich annotations** can unlock progress across multiple AI domains: enabling higher-quality 3D generation, improving long-tail 2D recognition through synthetic data augmentation, revealing robustness issues in current vision models, and scaling embodied AI to open-vocabulary navigation. It establishes that web-sourced 3D assets can provide the same scaling benefits for 3D vision that web-scraped image-text datasets provided for multimodal 2D vision.
