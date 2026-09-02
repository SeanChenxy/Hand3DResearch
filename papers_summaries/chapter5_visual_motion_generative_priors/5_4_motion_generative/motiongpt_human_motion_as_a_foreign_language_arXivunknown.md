# MotionGPT: Human Motion as a Foreign Language

**Authors:** Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen  
**Date:** 2023-06-26  
**Identifier:** [arXiv:2306.14795](https://arxiv.org/abs/2306.14795)  
**Zotero item:** `B6ILD4TK` ([Zotero](zotero://select/library/items/B6ILD4TK))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

MotionGPT is a unified motion-language model that treats human motion as a foreign language: a VQ-VAE converts 3D motion into discrete "motion tokens" merged with the word vocabulary of a pre-trained T5 language model, which is then motion-language pre-trained and instruction-tuned on prompt-based question-answering. A single 220M-parameter model handles text-to-motion, motion captioning, motion prediction, and motion in-betweening, reaching competitive state-of-the-art results on HumanML3D and KIT (e.g., text-to-motion FID 0.232 on HumanML3D, captioning BLEU@4 12.47 vs. TM2T's 7.00).

## Background and Problem

Prior motion generation systems are task-specific: diffusion or VQ-GPT models handle one task at a time (generation, captioning, prediction), treat motion and language as separate modalities, require strictly paired data, and generalize poorly to unseen tasks. The authors observe that motion exhibits semantic coupling with language ("body language") and follow the vision-language pre-training recipe of BEiT-3 to model motion as an additional language. The problem is to build a single pre-trained motion-language model that learns motion-language correlation from both paired and unpaired data and can serve diverse motion-relevant tasks through natural-language prompts, in the spirit of InstructGPT.

## Method

MotionGPT has two components. (1) A motion tokenizer: a VQ-VAE with 1D convolutional encoder and decoder quantizes motion frames into discrete tokens with a temporal downsampling rate of 4 and a 512x512 codebook, trained with reconstruction (including L1 smooth and velocity regularization), embedding, and commitment losses plus EMA and codebook reset. (2) A motion-aware language model: the motion vocabulary (with special start/end-of-motion tokens) is appended to T5's WordPiece vocabulary, and a transformer encoder-decoder (Flan-T5-Base, 220M parameters, 12-layer encoder and decoder, d_model 768) autoregressively predicts output tokens from mixed text-motion inputs. Training proceeds in three stages: (i) tokenizer training (150K iterations); (ii) motion-language pre-training on T5 with span-corruption denoising (15% masked sentinel spans) plus supervised translation between paired motion and text (300K iterations); (iii) instruction tuning on a multi-task prompt dataset built from HumanML3D and KIT — 15 core motion tasks expanded with dozens of templates each into over one thousand distinct instruction tasks (300K iterations).

## Contributions

- The first unified motion-language generative pre-trained model that treats human motion as a foreign language, handling generation, captioning, prediction, and in-betweening with one model.
- A motion-language training scheme with instruction tuning that learns from task feedback via prompts and improves multi-task and unseen-prompt performance.
- A general multi-task motion benchmark with unified evaluation across the four tasks, released with code and data.

## Experimental Setup

Datasets: HumanML3D (14,616 motions from AMASS with 44,970 descriptions) and KIT (3,911 motions, 6,353 descriptions) for text-to-motion and captioning; a subset of AMASS (motion-only) for motion prediction (condition on the first ~20% of frames) and in-betweening (~50% of frames masked), using the standard HumanML3D joint velocity/position/rotation representation. Metrics: FID, Diversity, R-precision, MM-Dist, and MultiModality for generation; BLEU, ROUGE, CIDEr, and BERTScore for captioning; FID plus Average/Final Displacement Error for completion. Training uses 8 Tesla V100 GPUs, AdamW (tokenizer lr 1e-4, batch 256; language model lr 2e-4 pre-training and 1e-4 instruction tuning, batch 16). Comparisons include MDM, MLD, T2M-GPT, TM2T, MotionDiffuse, and T2M; MDM was re-implemented for prediction under identical settings.

## Results

On HumanML3D text-to-motion, fine-tuned MotionGPT attains FID 0.232, R-precision top-3 0.700, MM-Dist 3.096, and Diversity 9.528 — competitive with T2M-GPT (FID 0.116) and MLD (0.473) and better than MDM (0.544), while the pre-trained-only model already reaches FID 0.160. In the multi-task table, MotionGPT is the only method scoring on all four tasks: captioning BLEU@4 12.47 and CIDEr 29.2 vs. TM2T's 7.00 and 16.8; motion prediction FID 0.905, ADE 4.745, FDE 6.040 vs. MDM's 6.031/5.446/8.561; motion in-between FID 0.214, Diversity 9.560 vs. MDM's 2.698/8.420, with comparable in-between ADE (3.762 vs. 3.787). Ablations across 60M/220M/770M models show the 220M base performs close to the 770M large model (in-between FID 0.214 vs. 0.223), which the authors attribute to the small scale of current motion data (~15K sequences); instruction tuning improves multi-task versatility and text-to-motion quality but slightly degrades pure text-generation metrics.

## Limitations

The paper itself states that MotionGPT covers only articulated human body motion, excluding faces, hands, and animals; it does not model human-object or human-environment interactions or multi-person interaction scenarios, which the authors flag as future work for the motion-language framework. The ablations further indicate that current motion datasets are too small for larger language models to help, and that instruction tuning trades off performance on pure text-generation tasks.
