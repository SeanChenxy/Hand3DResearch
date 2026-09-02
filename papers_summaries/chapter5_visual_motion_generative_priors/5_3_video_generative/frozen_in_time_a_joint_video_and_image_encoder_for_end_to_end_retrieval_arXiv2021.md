# Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval

**Authors:** Max Bain, Arsha Nagrani, Gul Varol, Andrew Zisserman  
**Date:** 2021-04-01  
**Identifier:** [arXiv:2104.00650](https://arxiv.org/abs/2104.00650); DOI `10.1109/ICCV48922.2021.00175`  
**Zotero item:** `95KWJVHE` ([Zotero](zotero://select/library/items/95KWJVHE))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary
Frozen in Time (ICCV 2021) is an end-to-end dual-encoder model for video-text retrieval whose space-time transformer, inspired by ViT and TimeSformer, applies divided space-time attention directly to pixels and treats images as single-frame videos, enabling joint training on image and video captioning data with a curriculum that starts "frozen" on images and gradually expands temporal context. Together with the new WebVid-2M dataset (2.5M web video-text pairs), it sets state-of-the-art results on MSR-VTT, MSVD, DiDeMo, and LSMDC while pretraining on roughly 20x fewer pairs than HowTo100M-based competitors.

## Background and Problem
Joint visual-text models were developing on two separate tracks for images and videos, and dominant video-retrieval methods relied on pre-extracted "expert" features (face, scene, action, sound, speech models). Large video-text datasets such as HowTo100M are noisy, so competitive performance previously required very large compute and scale; the paper targets an efficient, end-to-end trainable joint embedding for text-to-video retrieval that exploits both image and video data.

## Method
The visual encoder is a transformer operating on patched pixels with divided space-time attention and additive spatial plus temporal positional embeddings; images are handled as 1xN single-frame inputs and videos as MxN token grids, so one encoder trains flexibly on both. A text encoder (BERT-style transformer) with a CLS token forms the other branch; the dual encoders are trained end-to-end with a contrastive NCE-style objective over the joint embedding space. A curriculum learning schedule begins with image-only training, then interpolates (or zero-pads) temporal embeddings as the number of training frames grows, letting short-frame models cover more data early. The authors also introduce WebVid-2M, over 2 million videos with weak alt-text captions scraped from stock-footage sites, plus an additional 0.5M pairs in the combined 2.5M-pair pretraining set.

## Contributions
(i) An end-to-end video retrieval model using no expert features, applying divided space-time attention directly to pixels; (ii) an architecture that gracefully handles variable-length inputs, enabling joint image-video training with a temporal curriculum that speeds training and improves accuracy; (iii) the WebVid-2M video-text pretraining dataset; (iv) state-of-the-art video-only retrieval results on MSR-VTT, MSVD, DiDeMo, and LSMDC (LSMDC in the supplementary), beating expert-feature methods and HowTo100M-pretrained systems.

## Experimental Setup
Pretraining uses CC3M (3M image-text pairs), WebVid-2M, and optionally COCO Captions, versus a 17.1M-pair HowTo100M subset (19K hours) used for comparison; downstream finetuning and evaluation are on MSR-VTT (1K-A split), MSVD, DiDeMo, and LSMDC, reporting R@1/R@5/R@10 and Median Rank for text-to-video retrieval, with ablations on frame count, curriculum, and temporal embedding expansion.

## Results
Pretraining ablation on MSR-VTT: CC3M + WebVid2M reaches R@1 27.3 and R@10 68.1 (MedR 4.0), beating a 17.1M-pair HowTo100M subset (24.1/63.9) despite 3x fewer pairs, confirming HowTo100M's noise. Finetuned MSR-VTT (Table 4): R@1 32.5, R@5 61.5, R@10 71.2 with CC3M+WebVid2M+COCO (6.1M pairs), versus Support Set's 30.1/58.5/69.3 pretrained on 136M HowTo100M pairs. Zero-shot MSR-VTT: R@1 23.2 versus 7.5 (MIL-NCE) and 8.7 (Support Set). MSVD: R@1 33.7, R@5 64.7, R@10 76.3 (MedR 3.0), surpassing Support Set (HowTo PT) at 28.4 R@1. On DiDeMo, zero-shot performance matches ClipBERT's finetuned results, and finetuning adds a further 14.2% R@1.

## Limitations
The paper has no dedicated limitations section. The conclusion states performance is not yet saturated and could improve by training on the full HowTo100M, larger weakly paired image datasets such as Google3BN, and multi-dataset combinations. WebVid-2M captions are weak (alt-text) and web-scraped, and the paper notes HowTo100M's noise as a data-quality constraint on the field; other failure modes are not reported in the paper.

