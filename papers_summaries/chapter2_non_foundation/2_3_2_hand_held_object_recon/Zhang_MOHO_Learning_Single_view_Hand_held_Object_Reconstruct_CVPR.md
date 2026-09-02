# MOHO: Learning Single-view Hand-held Object Reconstruction with Multi-view Occlusion-Aware Supervision

**Authors:** Chenyangguang Zhang, Guanlong Jiao, Yan Di, Gu Wang, Ziqin Huang, Ruida Zhang, Fabian Manhardt, Bowen Fu, Federico Tombari, Xiangyang Ji  
**Date:** 2023-10-18  
**Identifier:** [arXiv:2310.11696](https://arxiv.org/abs/2310.11696)  
**Zotero item:** `YHQLXWPS` ([Zotero](zotero://select/library/items/YHQLXWPS))  
**Evidence status:** Identity verified against Zotero/arXiv metadata; the summary was written without full-text extraction, and unavailable details are marked as not reported.  

## Summary

MOHO is a synthetic-to-real framework for reconstructing a hand-held object from a single image, built around the observation that readily accessible hand-object videos can replace hard-to-collect 3D ground-truth models as a training signal — but such videos only provide heavily occluded object observations. The framework addresses the two predominant occlusion types, hand-induced occlusion and object self-occlusion, by pre-training on a large-scale rendered synthetic dataset (SOMVideo) that supplies multi-view occlusion-free supervision in both 2D and 3D spaces, then fine-tuning on real videos with amodal-mask-weighted geometric supervision that discounts unfaithful guidance from hand-occluded views, and augmenting the network with domain-consistent occlusion-aware features to infer complete object shapes under self-occlusion. According to the paper's abstract, a 2D-supervised variant of MOHO outperforms 3D-supervised prior methods by a large margin on the HO3D and DexYCB benchmarks.

## Background and Problem

Single-view hand-held object reconstruction methods conventionally rely on supervision from 3D ground-truth object models, which are difficult and expensive to collect in the real world. Hand-object videos are a far more accessible data source, but the object observations they contain are heavily occluded, in two distinct ways: the grasping hand hides parts of the object (hand-induced occlusion), and the object hides itself from a given viewpoint (self-occlusion), so naive supervision from such views is unfaithful. The problem the paper addresses is how to exploit multi-view occlusion-aware supervision from hand-object videos to train a single-view reconstruction model that nonetheless predicts complete object geometry.

## Method

The framework proceeds in two stages. In the synthetic pre-training stage, the authors render a large-scale synthetic dataset, SOMVideo, consisting of hand-object images paired with multi-view occlusion-free supervisions; this data is used to address hand-induced occlusion in both 2D and 3D spaces. In the real-world fine-tuning stage, MOHO leverages amodal-mask-weighted geometric supervision, in which the amodal mask down-weights the unfaithful guidance caused by hand-occluded supervising views in real videos. In addition, domain-consistent occlusion-aware features are incorporated into the model to resist object self-occlusion when inferring the complete object shape. Further architectural details, loss formulations, and training hyperparameters are not reported in the abstract.

## Contributions

- A synthetic-to-real framework that learns single-view hand-held object reconstruction from multi-view occlusion-aware supervision in hand-object videos instead of 3D ground-truth model supervision.
- The large-scale synthetic SOMVideo dataset of hand-object images with multi-view occlusion-free supervision, used to pre-train against hand-induced occlusion in 2D and 3D spaces.
- An amodal-mask-weighted geometric supervision scheme for the real-data fine-tuning stage, plus domain-consistent occlusion-aware features that improve completeness under object self-occlusion.

## Experimental Setup

According to the abstract, evaluation is conducted on the HO3D and DexYCB hand-object benchmarks, comparing MOHO — including a variant supervised only in 2D — against methods trained with 3D ground-truth supervision. Specific protocol details such as splits, metrics, and baseline configurations are not reported in the abstract.

## Results

The abstract reports that extensive experiments on HO3D and DexYCB demonstrate that the 2D-supervised MOHO gains superior results against 3D-supervised methods by a large margin. No specific quantitative values (metric names or numbers) are available in the verified abstract.

## Limitations

Limitations are not discussed in the abstract. The design implies dependence on the quality of the synthetic-to-real transfer and on amodal mask predictions for weighting real-world supervision, but no limitations are explicitly reported in the verified abstract.

