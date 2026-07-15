# HowTo100M (ICCV 2019)

> Miech, Zhukov, Alayrac, Tapaswi, Laptev, Sivic. *HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips.* ICCV 2019. DOI: 10.1109/ICCV.2019.00272. Zotero Key: `Q2ZE92DQ`.

## Summary
HowTo100M is a large-scale "instructional video + natural speech transcription" dataset: 136M video clips, 1.22M instructional videos, 23K visual tasks. It provides video-narration pairs instead of precise manual annotation, and is the founding dataset of video-language pretraining.

## 1. Dataset Purpose
- Solves the fundamental problem that "existing video-language datasets are small in scale, rely on manual captions, and cannot be scaled up". HowTo100M directly uses the automatic transcription of instructional videos as weakly supervised language.
- Tasks: (1) text-video embedding learning; (2) video-language pretraining; (3) text-to-video retrieval; (4) action localization.
- Anchors "web-scale video-narration" as an independent pretraining paradigm.
- The founding data source of video-language foundation model training.

## 2. Data Composition
- Source: publicly available YouTube instructional videos. The ASR (speech recognition) transcription is automatically used as the weak label.
- Viewpoint: third-person (native to YouTube instructional videos).
- Scale: 1.22M videos, 136M video clips, 23K different visual tasks (such as cooking, crafting, repairing).
- Object and action: covering an extremely wide range of daily instructional actions (cooking, crafting, gardening, repairing, etc.).
- No 3D annotation, no fine hand / object annotation.

## 3. Annotation and Supervision
- Video: 1.22M publicly available instructional videos, automatically cut into 136M short clips.
- Annotations: automatic ASR transcription narration (weak label).
- 3D information: none.
- Hand: no annotation.
- Object: objects mentioned in the narration (no bbox / mask).
- Interaction: natural language description of the narration (weak).

## 4. Supported Evaluation
- Benchmark tasks: (1) text-to-video retrieval (finetune on benchmarks such as YouCook2, CrossTask); (2) action localization; (3) video-language pretrained model evaluation.
- Key metrics: R@1 / R@5 / R@10, mAP, mIoU.
- It is not an end-task benchmark, but a pretraining source + multiple downstream benchmark protocols.
- Finetune and evaluate on benchmarks such as YouCook2, CrossTask, MSR-VTT, and LSMDC.

## 5. Why It Matters
- The first to establish "web-scale video + narration" as an extensible video-language pretraining paradigm.
- The 136M video clip scale was the largest in video-language at the time (2019).
- Inspired a large amount of video-language foundation model work (VideoCLIP, CoCa, etc.).
- The core pretraining source of the "video generative prior" in Ch5 and the "video-based pretraining" in Ch6.
- Complements EPIC-KITCHENS: EPIC's strength is fine-grained action labels, while HowTo100M's strength is large-scale weak supervision.

## 6. Limitations and Biases
- ASR-transcribed weak labels: not fully aligned with the video content, and there is misalignment ("the narration talks for 1 minute, but the action happens in the 2nd minute").
- No 3D annotation: not directly comparable to HO-3D v3, ARCTIC, etc.
- No hand pose, 6D object pose, or contact annotation.
- YouTube videos are third-person and cannot be directly used for egocentric tasks.
- The distribution of 23K tasks is affected by YouTube content (many cooking classes), with weak coverage of long-tail tasks.
- Videos may contain non-instructional segments (intro, ads, etc.) and require additional cleaning.

## 7. Takeaway
HowTo100M is best used as a "video-language pretraining large-scale corpus". **Not suitable** as fine-tuning data for 3D HOI reconstruction, nor for egocentric tasks or fine-grained action recognition. In this survey, HowTo100M plays the role of "video-language pretraining main source" and serves as the pretraining data anchor shared by Ch5 / Ch6. After the model is pretrained on HowTo100M, it can be evaluated on downstream tasks such as YouCook2 and CrossTask, indirectly providing a basis for HOI tasks.
