# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

# Paper Summary

## Summary
CLIP trains an image encoder and a text encoder jointly with a contrastive InfoNCE objective on 400M (image, text) pairs collected from the internet, learning visual representations that enable zero-shot transfer to a wide range of downstream vision tasks by encoding class names (or arbitrary prompts) and matching them to images.

## 1. Problem and Setting
- **Task**: Pretraining a visual encoder that learns from natural-language supervision instead of fixed class labels, enabling zero-shot transfer to downstream tasks via prompt-based classification.
- **Input/Output**: Pretraining — (image, text) pairs; downstream — natural-language class descriptions are embedded by the text encoder, and the image with the highest cosine similarity to a class embedding is predicted.
- **Difficulties**:
  - Prior vision systems were tied to a fixed set of classes (e.g., ImageNet's 1,000 labels) — adding a new class requires new labeled data and retraining.
  - Existing image–text models (VirTex, ICMLM, ConVIRT) trained on hundreds of thousands of image–text pairs and underperformed ImageNet supervised baselines.
  - Caption-prediction objectives with transformer LMs were 3× less compute-efficient than bag-of-words / contrastive baselines at ImageNet zero-shot.
  - Existing image-text datasets (MS-COCO, Visual Genome) were too small; YFCC100M had noisy metadata that shrunk to ImageNet-scale after filtering.

## 2. Core Method
**Pipeline**: Web image–text pairs (WIT, 400M) → image encoder (ResNet or ViT) + text encoder (CBOW or Text Transformer) → contrastive InfoNCE on N×N image-text cosine matrix → trained encoders → at inference, embed candidate class prompts and pick the highest-cosine image.

**Key components**:
1. **WIT (WebImageText) dataset**: 400M (image, text) pairs collected from publicly available internet sources; queries built from English-Wikipedia words, bigrams with high PMI, and WordNet synsets; up to 20K pairs per query.
2. **Contrastive pretraining objective** (CLIP): A symmetric InfoNCE loss that maximizes the cosine similarity of N correctly-paired image and text embeddings and minimizes it for the N²−N incorrect pairs in a batch — instead of predicting the exact caption token-by-token.
3. **Joint training of image and text encoders from scratch**: Both encoders are learned end-to-end with a learned temperature τ; image and text features are projected into a shared multimodal embedding space and L2-normalized.
4. **Zero-shot transfer via prompt engineering**: At test time, the learned text encoder embeds prompts like "a photo of a {object}" to synthesize a zero-shot linear classifier — new classes are added without any retraining.
5. **Scaling**: 5 ResNet sizes (RN50, RN101, RN50x4, RN50x16, RN50x64) and 3 ViT sizes; largest CLIP trains for ~12 days on 592 V100 GPUs (RN50x64).

## 3. Knowledge, Supervision, and Assumptions
- **Training data**: WIT — 400M (image, text) pairs from public web sources; class name coverage is determined by the natural distribution of words on the internet.
- **Supervision**: Weak — image–text co-occurrence rather than class labels. The contrastive objective uses only batch-internal positives/negatives; no external labels.
- **Foundation-model usage**: Builds on ConVIRT (contrastive image–text pretraining) but scales it 100–1000×.
- **Assumptions**:
  - Web-scale natural-language descriptions carry enough semantic supervision to learn transferable visual features.
  - Contrastive learning on noisy image–text pairs is more compute-efficient than caption prediction.
  - Class names can be embedded at test time to recover a "linear classifier" without any training data on the target classes.
- **Learned vs. provided**: Both image and text encoders are learned from scratch; class definitions at test time are provided by the user as text prompts.

## 4. Experiments and Findings
- **Datasets**: 30+ downstream datasets covering classification (ImageNet, ImageNet-A/R/Sketch), fine-grained classification, OCR, action recognition, geo-localization, video understanding, retrieval, etc.
- **Metrics**: Zero-shot top-1 accuracy on each downstream dataset; few-shot accuracy at K labeled examples per class; robustness gap vs supervised baselines.
- **Key results stated**:
  - Zero-shot CLIP matches the accuracy of the original ResNet-50 on ImageNet without using any of the 1.28M ImageNet training examples.
  - CLIP's contrastive objective is 4–28× more compute-efficient than a transformer caption-prediction baseline (3× from switching to BoW + 4× from switching to contrastive; see Fig. 2 of paper).
  - Zero-shot CLIP is significantly more robust than a same-accuracy supervised ImageNet model — the "robustness gap" is largely closed.
  - CLIP transfers non-trivially to OCR, geo-localization, fine-grained recognition, action recognition, and video tasks.
- **Ablations**:
  - Contrastive > caption prediction × BoW > transformer LM (compute-efficiency ordering).
  - Data scale (more pairs) and model scale (more parameters) both improve zero-shot accuracy.
  - Prompt ensembling ("a photo of a {object}" / "a centered photo of a {object}" / etc.) yields small but consistent gains.

## 5. Strengths and Limitations
### Strengths
- **Zero-shot transfer**: Class names or arbitrary natural-language prompts can be used as classifiers without any downstream training.
- **Robustness**: Zero-shot CLIP closes much of the "robustness gap" relative to same-accuracy supervised models on ImageNet-A/R/Sketch.
- **Compute efficiency**: The contrastive objective is ~4× more efficient than transformer caption-prediction.
- **Open-world coverage**: Trained on 400M web pairs, vocabulary is limited by what appears on the public web.
- **Reusable text encoder**: Can embed arbitrary prompts at test time to define new tasks.

### Limitations
- **Prompts matter**: Performance on a downstream dataset depends on prompt engineering and prompt ensembling; not all classes are described naturally in web text.
- **Fine-grained under-performance**: Zero-shot CLIP still trails task-specific supervised models on many fine-grained tasks and structured-prediction tasks (counting, depth, etc.).
- **Out-of-distribution failure modes**: Adversarial robustness is not studied; behavior on extreme domain shifts (medical, scientific imagery) can be poor.
- **Data bias**: WIT inherits internet biases (gender, ethnicity, geography) and harmful associations from web text.
- **Compute scale**: Largest CLIP models require substantial compute (12 days × 592 V100s for RN50x64); reproducibility of the largest models is non-trivial.
- **No explicit pixel-level supervision**: Despite strong image-level features, dense prediction (segmentation, depth) typically benefits from additional fine-tuning.

## 6. Takeaway
CLIP shows that **natural-language supervision at internet scale** is a viable alternative to fixed-label supervised pretraining for vision — by training an image encoder and a text encoder contrastively on 400M (image, text) pairs, it learns visual representations that match supervised baselines zero-shot on ImageNet and generalize to a wide range of downstream tasks via prompt-based classification. The key engineering insight — that a contrastive InfoNCE objective on noisy web pairs is dramatically more compute-efficient than caption prediction — opened the era of large-scale image-text foundation models. For HOI research, CLIP's text encoder and image encoder underpin open-vocabulary category grounding and language-conditioned retrieval priors, providing a robust way to match an in-frame hand-held object or a candidate 3D asset to a category description.