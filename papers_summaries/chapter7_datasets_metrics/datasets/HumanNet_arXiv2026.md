# HumanNet: Scaling Human-centric Video Learning to One Million Hours

## Summary
HumanNet is a one-million-hour human-centric video corpus combining first-person and third-person footage with caption, motion, hand and body annotations, paired with a curation pipeline that treats human-centric filtering, viewpoint diversity, and motion-aware annotation as first-class design principles, providing infrastructure for representation learning, motion generation, and human-to-robot transfer.

## 1. Problem and Setting
- Task: Build a large-scale human-centric video dataset that bridges the gap between internet-scale vision-language pretraining and embodied (robot) data, addressing the data limitation of physical interaction learning.
- Input: Heterogeneous web video, video-platform search, general search engines, open-source datasets, and self-collected recordings across first-person and third-person viewpoints.
- Output: A structured, clip-level corpus of ~1,000,000 hours, indexed by source type, viewpoint, task structure, environment, interaction style, motion category, and metadata availability, with caption labels, motion descriptions, 3D hand and body pose, monocular SLAM camera trajectories, and motion retargeting.
- Span: Both first-person and third-person; covers fine-grained activities, human-object interactions, tool use, and long-horizon procedures in homes, workplaces, kitchens, warehouses, public spaces, and outdoor settings.
- Why difficult: Open-world human-centric video is noisy, has ambiguous labels, inconsistent task boundaries, missing metadata, viewpoint imbalance, and variable visual quality; coupling such data with reliable motion, hand, and body annotations while keeping the corpus pretraining-ready at one-million-hour scale is the central design challenge.

## 2. Core Method
The pipeline has three auditable stages.
1. Data collection. Seed keywords are iteratively expanded through keyword expansion, keyword-based crawling, channel-level crawling, and integration of existing sources. Guided by this keyword repository, candidates are pulled from video platforms, general web search engines, directly crawled videos, open-source datasets, and self-collection in real-world environments, then merged into a unified mixed-video pool. Channel- and source-level filtering removes off-topic, low-quality, or passive recordings; for first-person material this yields an ego-video URL pool, and third-person material is retained when human motion/activity is visually central.
2. Data processing. Raw videos are converted into clip-level training samples via: (i) de-duplication and normalization (frame rate, resolution, container); (ii) content filtering to retain clips with meaningful human action; (iii) quality filtering to remove severe motion blur, heavy occlusion, static framing, etc.; (iv) scene splitting that segments long videos at visual changes; (v) video clipping that produces fixed-granularity segments.
3. Annotation. Each clip is enriched with: 3D hand and body pose detection; monocular SLAM camera trajectory for first-person clips meeting stability/parallax requirements; motion retargeting that aligns recovered human motion with a unified humanoid skeleton, flagging a clip as "robot-ready" when retargeting error is below 15 mm and valid-frame coverage exceeds 60%; and LLM-assisted captioning that produces short captions, long descriptions, motion descriptions, and activity classifications, normalized against any inherited source narrations.

The taxonomy is multi-axis rather than flat (Figure 2), allowing the corpus to be sliced by source type, viewpoint, environment, activity category, motion pattern, etc., which is what makes scale coexist with physical specificity.

## 3. Knowledge, Supervision, and Assumptions
- Training data: web video, open-source datasets (Ego4D, EPIC-KITCHENS, Ego-Exo4D, HOI4D, Something-Something, etc.), video-platform and search-engine retrieval, and self-recorded ego + exo footage in real-world settings.
- Supervision signals: 3D hand and body pose detection; monocular SLAM trajectories; humanoid-skeleton motion retargeting (threshold: <15 mm retargeting error and >60% valid-frame coverage → "robot-ready"); LLM-assisted video captions, motion descriptions, activity classifications; hierarchical activity labels (Label 1/2/3); subject ID; appearance.
- Domain knowledge: explicit definition of human-centric clip = footage in which human activity is the organizing signal, covering manipulation, tool use, locomotion, multi-person coordination, multi-step procedures with state changes; explicit exclusion of passive or weakly grounded video.
- Foundation models used: Qwen2.5-VL-style VLM for the validation post-training; LLM-assisted captioning pipeline. The downstream validation uses LingBot-VLA with a Qwen-based VLM backbone.
- Assumptions: (i) actor-centered cues in ego video (contact dynamics, hand-object relations, temporal intent) are transferable to robot representations; (ii) exocentric motion retargeted to a unified humanoid skeleton is meaningful supervision for cross-embodiment learning; (iii) internet-scale human activity is broad enough to expose long-tail physical behaviors that robots need.

## 4. Experiments and Findings
- Headline corpus statistics: 967K hours, 150K+ objects, 720K+ tasks (Figure 1). Subsets are filtered by pose quality, motion magnitude, motion length, etc.
- Comparison vs prior corpora (Table 1): HumanNet (1,000,000 h, first+third, fine-grained activity, direct embodied use) is positioned against EPIC-KITCHENS-100 (~100 h), Ego4D (~3,670 h), HOI4D (2.4M frames), EgoDex (829 h), OpenEgo (1,107 h), EgoScale (20,854 h), EgoVerse (1,362 h), ActivityNet (>648 h), Kinetics (up to 650k clips), Charades (9,848 videos / 68.8 h), AVA, Something-Something V2 (220,847 videos), HACS, FineGym, HowTo100M (136M clips / 1.22M videos), Ego-Exo4D (1,286 h), and Human2Robot (2,600 episodes).
- Controlled VLA post-training validation (Section 3.5, Figure 6): four configurations under the same LingBot-VLA architecture and a fixed 34-hour downstream corpus (100 tasks × 20 episodes): (1) Qwen VLM baseline; (2) Qwen + 100 h real-robot CoBot data; (3) Qwen + 1,000 h egocentric HumanNet video; (4) LingBot (Qwen + 20,000 h real-robot). Result: the 1,000 h egocentric-pretrained variant matches and on several held-out task groups slightly surpasses the 100 h real-robot variant, and substantially closes the gap to the 20,000 h real-robot LingBot baseline (Figure 6 shows per-task-group validation loss).
- Statistical structure (Section 3.4, Figure 5): pose-score distribution concentrates at the high-confidence end after quality filtering; motion-score and motion-length distributions are heavy-tailed but bounded; athletic/outdoor families show longer/higher-magnitude motion while daily activities concentrate on shorter, finer-grained segments.

## 5. Strengths and Limitations
### Strengths
- Largest human-centric video corpus to date at ~1,000,000 hours, with explicit first+third viewpoint indexing, taxonomy, and privacy/quality review built into the release pipeline.
- Three-stage pipeline (collection / processing / annotation) cleanly separates source acquisition from clip-level cleaning and supervision generation, so each stage can be audited, extended, or rerun independently.
- Action-centric annotations: 3D hand + body pose, SLAM camera trajectory, humanoid retargeting with explicit robot-ready thresholds, and LLM-derived caption / motion / activity labels.
- Controlled VLA validation provides a concrete, architecture-fixed evidence point that 1,000 h of egocentric HumanNet video substitutes for ~100 h of real-robot data under matched post-training, supporting the central data-centric claim.

### Limitations
- Human behavior ≠ robot behavior: even at 1 M hours the dataset supplies transferable priors, not a one-to-one replacement of robot data — the embodiment gap (human hands, bodies, tools, mobility vs robot control spaces) is not eliminated.
- Open-world scale brings noise: ambiguous labels, inconsistent task boundaries, missing metadata, viewpoint imbalance, variable visual quality; pose/retarget/caption annotations carry their own errors.
- Coverage can remain uneven (geography, socioeconomic context, occupation, body types, household routines) — 1 M hours can create an illusion of universality while leaving real blind spots.
- Privacy and dual-use risk: first-person and third-person recordings can capture bystanders, private interiors, screens, identifiable people, proprietary workflows; requires license review, redaction, access controls, and clear exclusion documentation.
- The validation study is narrow: a single VLA architecture (LingBot-VLA), a single downstream corpus (100 tasks, 34 h), and only four initialization configurations — broader policy/architecture ablations are not provided.

## 6. Takeaway
HumanNet argues that the bottleneck for general-purpose embodied AI is no longer model architecture but data infrastructure, and demonstrates that systematically curated, multi-axis-indexed, motion- and hand-annotated human-centric video at one-million-hour scale is a scalable and cost-effective substrate for embodied foundation models — the controlled VLA ablation showing 1,000 h of ego video substituting for 100 h of real-robot data is the headline evidence that reframes internet-scale human video as a first-class pretraining resource for physical AI.
