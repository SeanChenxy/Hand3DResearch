# HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips

**Authors:** Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, Josef Sivic  
**Date:** 2019-06-07  
**Identifier:** [arXiv:1906.03327](https://arxiv.org/abs/1906.03327); DOI `10.1109/ICCV.2019.00272`  
**Zotero item:** `Q2ZE92DQ` ([Zotero](zotero://select/library/items/Q2ZE92DQ))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

HowTo100M is a dataset of 136 million video clips sourced from 1.22 million narrated instructional YouTube videos depicting over 23,000 visual tasks, in which each clip is paired with an automatically transcribed narration line, so no manual annotation is needed. At 134,472 hours (about 15 years) of video, it dwarfs prior clip-caption datasets by three orders of magnitude. The authors train a context-gating text-video embedding on this data and show it sets state of the art on instructional video step localization (CrossTask) and text-to-video retrieval (YouCook2), and transfers to generic and movie domains (MSR-VTT, LSMDC) after fine-tuning.

## Background and Motivation

Learning joint text-video embeddings normally requires large sets of manually annotated clip-caption pairs, which are expensive, slow to collect, and subjective. The authors observe that narrated instructional videos are abundant, and that their narration is produced with the explicit intention of explaining what is shown on screen, making it a natural, scalable source of weakly paired supervision. This follows the broader trend of large-scale noisy pretraining (BERT, GPT-2 in NLP; hashtag-supervised pretraining in vision) applied to video-language representation, with the goal of supporting text-to-video retrieval, text-based action localization, and human-robot communication applications.

## Dataset Construction

Visual tasks are harvested from WikiHow, restricted to 12 categories involving physical interaction with the world (Food and Entertaining is the largest with 11,504 tasks and 54.4M clips), with non-physical verbs filtered out, yielding 23,611 tasks. For each task, a "how to [task]" YouTube query retrieves the top 200 results; videos must have English subtitles (manual, ASR, or translated), at least 100 views and 100 words, and no more than 2,000 seconds, and are deduplicated by YouTube ID. Each subtitle line becomes a caption paired with the video clip covering its time interval, giving weakly paired clip-caption data. The full set amounts to 136.6M clips from 1.22M videos (Table 2) covering 134,472 hours. Videos average 6.5 minutes and produce about 110 pairs each, with 4-second clips and captions of about 4 words after stop-word removal. Manual inspection of 400 sampled pairs found visual grounding in 51%, and a 100-video audit found 71% instructional content (12% vlogs, 7% product reviews or advertisements).

## Evaluation Protocol

The embedding model applies non-linear context-gating mappings (Equations 2-3 of the paper) to pre-extracted clip and caption features, with d = 4,096 and 67M parameters; video features are temporally max-pooled ResNet-152 2D (1 fps) concatenated with ResNeXt-101 3D (1.5 fps) features into a 4,096-dimensional vector, and captions are encoded with a shallow 1D-CNN over word2vec embeddings. Training uses the max-margin ranking loss with margin 0.1 and Adam at learning rate 1e-4, and is notable for intra-video negative sampling: half of the negative pairs are caption-clip pairs drawn from the same YouTube video as the positive pair, forcing the embedding to focus on relevant visual content rather than background scene. Training takes less than three days on a single Tesla P100. Evaluation covers text-based action step localization on CrossTask (18 tasks, 2.7K videos, recall metric) and clip retrieval on YouCook2 (validation clips), MSR-VTT, and LSMDC (R@1/5/10 and median rank), with CrossTask test videos removed from the training set; off-the-shelf, dataset-only, and pretrain-then-finetune regimes are all compared, plus a data-scaling study that retrains on subsets from top-2 (15K videos) to top-200 search rank (the full dataset).

## Findings and Analysis

Intra-video negative sampling is critical, raising CrossTask average recall from 25.7 to 33.6 and YouCook2 R@10 from 18.1 to 24.8. With it, the off-the-shelf HowTo100M embedding reaches 33.6 average recall on CrossTask step localization, clearly beating the weakly supervised state of the art of Zhukov et al. (22.4) and even the fully supervised upper bound (31.6), with gains on all tasks except "Make Meringue". On YouCook2 retrieval, the off-the-shelf model attains 24.8 R@10 and fine-tuning boosts this to 35.3 versus 21.6 for the prior HGLMM FV CCA method. On MSR-VTT, pretraining plus fine-tuning reaches 52.8 R@10 versus 43.2 for JSFusion, and comparable state-of-the-art performance is retained using only 20% of MSR-VTT supervision. On LSMDC, fine-tuning lifts R@10 from 25.0 (dataset-only) to 27.9, though direct LSMDC training with JSFusion (34.1) remains higher on this most distant domain. The scaling study shows monotonic improvement on all tasks as training data grows from 15K to the full 1.22M videos, with no saturation, implying that more web video would help further.

## Contributions

The HowTo100M dataset (136M weakly paired clip-caption pairs, 1.22M instructional videos, 23,611 tasks across 12 domains) built with a fully automatic, scalable collection pipeline requiring no manual annotation; a context-gating non-linear text-video embedding trained with intra-video negative sampling; state-of-the-art results on CrossTask step localization and YouCook2 retrieval off the shelf, and after fine-tuning on MSR-VTT and LSMDC; and a data-scaling analysis demonstrating that embedding quality grows steadily with dataset size.

## Limitations

The clip-caption pairs are weakly supervised: only 51% of inspected pairs are visually grounded, captions are often incomplete or ungrammatical (ASR artifacts), and some videos are non-instructional (vlogs, reviews, ads), although the authors retain these as still useful. Deduplication by YouTube ID cannot catch re-uploaded or edited duplicates. The embedding is a shallow (67M-parameter) model over pre-extracted CNN features, and the paper's attempts to automatically filter incorrect positive pairs during training did not yield improvements, which the authors attribute to the model's shallowness and the large data volume; transfers to the most distant domain (LSMDC movies) remain behind domain-specialized training.
