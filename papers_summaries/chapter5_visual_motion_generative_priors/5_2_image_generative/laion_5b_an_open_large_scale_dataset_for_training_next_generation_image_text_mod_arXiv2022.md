# LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models

**Authors:** Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, et al.  
**Date:** 2022-10-16  
**Identifier:** [arXiv:2210.08402](https://arxiv.org/abs/2210.08402)  
**Zotero item:** not in the Zotero snapshot (repository-only prior-source card)  
**Evidence status:** Identity verified against Zotero/arXiv metadata; summary content is derived from the paper with in-text caveats where detail is unavailable.  
## Summary
LAION-5B addresses the lack of publicly available image–text data at the scale needed to train modern multimodal models. It constructs a web-scale collection from Common Crawl and filters image–text pairs with a pretrained CLIP model, yielding 5.85 billion publicly accessible pairs. The release also provides language subsets, search and nearest-neighbor tooling, and safety-related scores. A CLIP model trained on the 400-million-pair subset is reported to reach performance comparable to OpenAI's CLIP on several evaluations, while the paper emphasizes that the dataset requires further curation.

## Background and Problem
Large image–text models require hundreds of millions or billions of aligned examples, but datasets of this size had largely been proprietary. LAION-5B takes web image–text associations as input and outputs a public indexed dataset of image URLs, text, and metadata rather than a new generative model. The dataset is intended to support reproducible training and analysis of multimodal models.

## Method
The construction pipeline starts from Common Crawl records, retrieves candidate image–text pairs, and applies CLIP-based image–text similarity filtering. The released collection is divided into 2.32 billion English pairs, 2.26 billion multilingual pairs, and 1.27 billion language-agnostic pairs. The release includes nearest-neighbor indices, an exploration interface, and scores for watermark, NSFW, and toxicity detection.

## Contributions
- A public collection of 5.85 billion CLIP-filtered image–text pairs.
- English, multilingual, and language-agnostic subsets for different training settings.
- Tooling and metadata for searching, inspecting, and curating large-scale multimodal data.

## Experimental Setup
The paper trains CLIP models on LAION-400M, a 400-million-pair subset, and compares them with OpenAI's CLIP. Evaluation includes ImageNet zero-shot classification, distribution-shift benchmarks, VTAB, retrieval, and fine-tuning scenarios. The paper also reports an open implementation of the ViT-L/14 OpenCLIP configuration.

## Results
The LAION-400M-trained CLIP models are reported to perform comparably to OpenAI's CLIP across the listed evaluations. The dataset is 23 times larger than the prior public datasets cited by the paper, according to the reported comparison. The results demonstrate scale and reproducibility rather than a claim that every web pair is correctly aligned.

## Limitations
The dataset is a noisy web-scale resource rather than a fully curated finished product. It contains biases and potentially harmful or copyrighted material inherited from the source web data, and the authors recommend careful investigation and academic use. CLIP filtering can also preserve or amplify errors in the filtering model.
