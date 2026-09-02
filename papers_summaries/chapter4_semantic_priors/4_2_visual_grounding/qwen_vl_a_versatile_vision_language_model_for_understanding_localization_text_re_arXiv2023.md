# Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

**Authors:** Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, Jingren Zhou  
**Date:** 2023-08-24  
**Identifier:** [arXiv:2308.12966](https://arxiv.org/abs/2308.12966)  
**Zotero item:** `8YJTICS7` ([Zotero](zotero://select/library/items/8YJTICS7))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

Qwen-VL introduces a series of large vision-language models (9.6B parameters total) built on the Qwen-7B language model, distinguished among open-source LVLMs by supporting bounding-box grounding, OCR/text reading, multi-image and multilingual (English-Chinese) inputs alongside conventional captioning and VQA. Grounding and localization are learned by aligning image-caption-box tuples in which boxes are serialized as ordinary text strings. Through a three-stage training pipeline (large-scale pretraining, multi-task pretraining at higher resolution, and supervised instruction fine-tuning yielding Qwen-VL-Chat), the models set records among generalist models of similar scale on captioning, VQA, text-oriented VQA, referring-expression comprehension, and instruction-following benchmarks, and all weights are publicly released.

## Background and Problem

Large language models excel at text but cannot natively process visual modalities, motivating a wave of large vision-language models. The authors argue that existing open-source LVLMs suffer from inadequate training and optimization and lag far behind proprietary models; moreover, because real-world visual scenarios are complicated, fine-grained visual understanding is crucial, yet most open-source LVLMs perceive images only coarsely and lack abilities such as object grounding or text reading — only a few attempts (the paper cites Kosmos-2/Kosmos-style grounding and Shikra) had addressed this direction. The problem the paper defines is thus to build an open, versatile vision-language foundation model that combines conventional captioning/QA with fine-grained capabilities — grounding, OCR, document understanding, multilingual (English and Chinese) conversation, and multi-image reasoning — at competitive scale and performance.

## Method

Qwen-VL has three components. (1) A large language model initialized from Qwen-7B (7.7B parameters). (2) A Vision Transformer visual encoder (1.9B parameters) initialized from OpenCLIP's ViT-bigG, processing images as patches with stride 14. (3) A position-aware vision-language adapter (0.08B parameters): a single-layer cross-attention module with 256 trainable query embeddings that compresses the variable-length image feature sequence to a fixed length of 256 tokens, with 2D absolute positional encodings incorporated into the query-key pairs to preserve positional details needed for fine-grained comprehension. The total model size is 9.6B parameters. Input images are resized to 224x224 in the first training stage and 448x448 afterward.

For localization, bounding boxes are normalized to the range [0, 1000) and serialized as the string format "(Xtopleft, Ytopleft), (Xbottomright, Ybottomright)", tokenized as ordinary text with no extra positional vocabulary; special tokens <box>/</box> delimit box strings and <ref>/</ref> mark the referred content, so grounding becomes text generation over image-caption-box tuples. Image features are wrapped in <img>/</img> tokens, and ChatML-style tokens structure dialogue in the chat model.

Training proceeds in three stages. Stage 1 (pretraining) uses web-crawled image-text pairs cleaned from 5 billion originals down to 1.4 billion (77.3% English, 22.7% Chinese); the LLM is frozen, the encoder and adapter are trained for 50,000 steps at batch size 30720 (about 1.5 billion samples). Stage 2 (multi-task pretraining) trains the full model at 448x448 resolution on seven simultaneous tasks: captioning (19.7M samples), VQA (3.6M), grounding (3.5M), referring grounding (8.7M), grounded captioning (8.7M), OCR (24.8M, including synthetic data from SynthDoG and Common Crawl PDF/HTML sources), and pure-text autoregression (7.8M), using GRIT, Visual Genome, RefCOCO/RefCOCO+/RefCOCOg for grounding tasks. Stage 3 (supervised fine-tuning) uses 350K instruction-tuning instances — combining LLM self-instructed caption/dialogue data with manually annotated, model-generated, and strategy-concatenated data covering localization and multi-image comprehension, mixed with pure-text dialogue — freezing the visual encoder and optimizing the LLM and adapter to produce Qwen-VL-Chat.

## Contributions

- An open vision-language model series that unifies image captioning, VQA, text-oriented VQA/OCR, document understanding, and visual grounding in one model, addressing the coarse-grained perception gap of prior open-source LVLMs.
- A position-aware adapter design (learnable-query cross-attention compression with 2D absolute positional encodings) that keeps visual sequences short (256 tokens) while retaining localization-relevant positional information.
- Grounding via text serialization: image-caption-box alignment with normalized box coordinates written as plain text strings, enabling both grounding (referring expression comprehension) and grounded captioning without auxiliary detection heads or specialized vocabularies.
- A three-stage training recipe plus a cleaned multilingual multimodal corpus, yielding both the base Qwen-VL and the instruction-tuned Qwen-VL-Chat with multilingual, multi-image, and fine-grained dialogue abilities.
- State-of-the-art results among similarly sized generalist models across a broad benchmark suite, with all models publicly released.

## Experimental Setup

Evaluation covers image captioning (Nocaps val, Flickr30K karpathy-test; CIDEr), general VQA (VQAv2 test-dev, OKVQA val, GQA test-balanced, ScienceQA-Img test, VizWiz test-dev; VQA score/accuracy), text-oriented VQA (TextVQA val, DocVQA test ANLS, ChartQA test relaxed exact match, AI2D test, OCR-VQA test), referring expression comprehension (RefCOCO, RefCOCO+, RefCOCOg, GRIT; accuracy on standard splits), few-shot in-context learning on OKVQA, VizWiz, TextVQA, and Flickr30K against Flamingo-9B/80B, OpenFlamingo-9B, and IDEFICS-9B/80B, and instruction following on TouchStone (English and Chinese, GPT-4 scoring), SEED-Bench (19K multiple-choice questions), and MME (14 perception/cognition subtasks). Captioning uses greedy decoding with the prompt "Describe the image in English:"; VQA uses open-ended generation with "{question} Answer:" prompts (option-constrained top-1 for ScienceQA). Comparisons are against generalist models (Flamingo, BLIP-2, InstructBLIP, Kosmos-1/2, Shikra, mPLUG-Owl, etc.) and specialist state-of-the-art systems (PaLI-X-55B, UNINEXT-H, ONE-PEACE, Grounding DINO-L).

## Results

- Zero-shot captioning: 121.4 CIDEr on Nocaps val and 85.8 CIDEr on Flickr30K karpathy-test (Qwen-VL), the best among generalist models and above Flamingo-80B's 67.2 on Flickr30K.
- General VQA (Qwen-VL): VQAv2 79.5, OKVQA 58.6, GQA 59.3, ScienceQA-Img 67.1, VizWiz 35.2 — each surpassing prior generalist models (e.g., InstructBLIP Vicuna-13B scores 65.0 VQAv2, 49.5 GQA, 33.4 VizWiz).
- Text-oriented VQA (Qwen-VL): TextVQA 63.8, DocVQA 65.1, ChartQA 65.7, AI2D 62.3, OCR-VQA 75.7, frequently by large margins over generalists (e.g., InstructBLIP 50.7 TextVQA), though below specialist fine-tuned PaLI-X-55B on several.
- Referring expression comprehension (Qwen-VL-7B): RefCOCO val/test-A/test-B 89.36/92.26/85.34, RefCOCO+ 83.12/88.25/77.21, RefCOCOg val/test 85.58/85.48, GRIT test 78.22 — top among generalist models, with Shikra-13B (e.g., 87.83 RefCOCO val) behind, though specialist models UNINEXT-H and ONE-PEACE remain higher (e.g., 92.64 RefCOCO val).
- Instruction following (Qwen-VL-Chat): TouchStone 645.2 (best, with a noted advantage in Chinese), SEED-Bench 401.2, MME 1487.58 — ahead of all compared LVLMs (e.g., InstructBLIP 552.4 TouchStone, mPLUG-Owl 605.4).
- Few-shot learning: Qwen-VL improves with in-context examples and outperforms similar-scale Flamingo-9B, OpenFlamingo-9B, and IDEFICS-9B, approaching much larger Flamingo-80B/IDEFICS-80B, using naive random-shot selection.
- Pure-text ability is preserved: after multimodal training Qwen-VL scores 50.7 MMLU, 49.5 CMMLU, and 51.1 C-Eval, comparable to its 7B LLM initialization (49.9 MMLU, 48.5 C-Eval), indicating no catastrophic forgetting.

## Limitations

The paper does not include a dedicated limitations section, but several constraints are stated or evident from its own experiments. Input resolution is capped at 448x448: the authors considered window attention to enable 896x896 inputs but rejected it because convergence loss is significantly higher and 896x896 training is too slow (about 2.5x longer per-iteration at 896x896 window attention settings in their ablation), leaving a resolution gap against the highest-performing specialist systems. The adapter compresses all visual content to a fixed 256-token sequence, and the authors note via ablation that too few queries lose visual information (too many slow convergence), so very information-dense images may be under-served. The grounding vocabulary is limited to the box-annotation formats of the training corpora, and specialist fine-tuned models still exceed Qwen-VL on several text-oriented VQA and referring benchmarks. The authors themselves list future work rather than shortcomings: integrating more modalities such as speech and video, scaling model size, data, and resolution, and extending to multimodal generation.
