# BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

# Summary

BLIP-2 enables efficient vision-language pre-training by bridging frozen image encoders and frozen large language models through a lightweight Querying Transformer (Q-Former) pre-trained in two stages, achieving state-of-the-art performance with significantly fewer trainable parameters.

## 1. Problem and Setting

- **Task:** Vision-language pre-training (VLP) for multimodal understanding and generation tasks including visual question answering, image captioning, image-text retrieval, and zero-shot instructed image-to-text generation
- **Inputs:** Images (processed by frozen image encoder) and text prompts; **Outputs:** Text responses/captions or similarity scores for retrieval
- **Difficulty:** End-to-end training of large-scale vision-language models is computationally prohibitive. Existing methods that freeze unimodal models struggle with the modality gap—especially when aligning visual features to the text space of frozen LLMs that have never seen images during pre-training. The language modeling loss used by prior works is insufficient to bridge this gap.

## 2. Core Method

**Pipeline:** Image → Frozen Image Encoder → Q-Former (learnable queries) → Frozen LLM → Generated Text

**Architecture:**
- **Q-Former:** A lightweight transformer with two submodules sharing self-attention layers:
  1. **Image Transformer:** Interacts with frozen image encoder via cross-attention
  2. **Text Transformer:** Interacts with text via cross-attention
- Uses a set of learnable query vectors (typically 32) to extract a fixed number of output visual features, independent of input resolution

**Two-Stage Pre-training:**

**Stage 1 - Vision-Language Representation Learning (with frozen image encoder):**
- Bootstraps visual-text alignment from frozen image encoder
- Three objectives:
  1. **Image-Text Contrastive Loss (ITC):** Aligns image and text embeddings; uses hard negatives
  2. **Image-Text Matching Loss (ITM):** Binary classification to determine if image-text pair matches
  3. **Image-Grounded Text Generation Loss (ITG):** Language modeling where text is conditioned on image features

**Stage 2 - Vision-to-Language Generative Learning (with frozen LLM):**
- Connects Q-Former output to frozen LLM
- Full query vectors fed to LLM as soft visual prompts
- Uses only language modeling loss to train Q-Former to output visual representations interpretable by LLM

**Key Innovation:** The Q-Former acts as an information bottleneck that learns to extract the most relevant visual features for both understanding (Stage 1) and generation (Stage 2), while keeping both unimodal models frozen to leverage their pre-trained knowledge without catastrophic forgetting.

## 3. Knowledge, Supervision, and Assumptions

**Training Data:** Large-scale image-text pair datasets (web-collected similar to CLIP/BLIP)
- Used: 146M images for FLAN-based models, 116M for OPT-based models (ImageNet + COCO + LAION + web data)

**Supervision Signals:**
- Image-text contrastive learning (paired supervision)
- Image-text matching (binary classification)
- Language modeling (next token prediction)
- No human annotations beyond image-text pairs

**Pretrained Models Used:**
- **Frozen Image Encoders:** CLIP ViT-L/14, ViT-g/14, ViT-e/14
- **Frozen LLMs:** OPT (2.7B, 6.7B), FlanT5 (XL, XXL, XXL-Cherry)
- Both remain frozen throughout training—only Q-Former parameters are updated

**What is Learned vs Provided:**
- **Learned:** Cross-modal alignment through Q-Former's query vectors; how to extract image-relevant visual features; how to map visual features to LLM's text space
- **Provided:** Visual representation capabilities (from image encoder); language generation and reasoning capabilities (from LLM)

**Assumptions:**
- Pre-trained unimodal models provide high-quality representations
- A fixed-size bottleneck (learnable queries) can capture sufficient visual information
- LLMs can interpret visual features when properly aligned

## 4. Experiments and Findings

**Datasets Evaluated:**
- **VQAv2:** Zero-shot visual question answering
- **COCO:** Image captioning (COCO Captions)
- **Retrieval:** Image-text retrieval on COCO and Flickr30k
- **NLVR2:** Visual reasoning

**Key Metrics:**
- VQAv2: Accuracy
- COCO Captioning: CIDEr, BLEU-4, SPICE, ROUGE-L
- Retrieval: Recall@1 (Image→Text and Text→Image)
- NLVR2: Accuracy

**Quantitative Results:**
- **VQAv2 Zero-shot:** BLIP-2 with FlanT5-XXL (11B) achieves 65.0%, outperforming Flamingo-80B by 8.7% with 54x fewer trainable parameters
- **COCO Captioning:** 113.7 CIDEr (zero-shot) and 129.6 CIDEr (fine-tuned)
- **Image-Text Retrieval:** 84.0% Image→Text TR@1 on COCO (using ViT-L/14 + OPT-2.7B)
- **Efficiency:** Only ~188M trainable parameters (0.4B for Q-Former + ~0.1B adapters) compared to billions in end-to-end methods

**Emerging Capabilities (Qualitative):**
- Zero-shot instructed image-to-text generation following natural language instructions
- Visual knowledge reasoning
- Visual conversation
- Photo-aware responses (e.g., "Write a romantic message for this photo" → contextually appropriate text)

**Ablation Findings:**
- Two-stage pre-training outperforms single-stage
- Both ITC and ITM objectives in Stage 1 are important
- Using image encoder pre-trained with image-text data (CLIP) works better than vision-only encoders
- Model performance scales with better frozen unimodal models (ViT-g/14, ViT-e/14, FlanT5-XXL-Cherry)

## 5. Strengths and Limitations

### Strengths
- **Compute Efficiency:** Achieves SOTA with 54x fewer trainable parameters than Flamingo-80B
- **Leverages Existing Models:** Effectively utilizes off-the-shelf frozen unimodal models without end-to-end retraining
- **Generic Framework:** Works with various combinations of image encoders and LLMs
- **Emergent Capabilities:** Enables zero-shot instructed image-to-text generation not possible with prior VLP methods
- **No Catastrophic Forgetting:** Frozen unimodal models retain their original capabilities
- **Two-Stage Alignment:** More effective than single-stage methods using only language modeling loss

### Limitations
- **Hallucination:** Like other LLM-based methods, can generate plausible but incorrect content
- **Training Data Requirement:** Still requires large-scale image-text pairs (~116M-146M images)
- **Fixed Visual Bottleneck:** The fixed number of learnable queries (32) may limit expressiveness for complex scenes
- **Frozen Components:** Cannot adapt the image encoder or LLM to the vision-language task
- **Limited Visual Reasoning:** Performance on tasks requiring deep visual reasoning (NLVR2) still lags behind some methods

## 6. Takeaway

BLIP-2 demonstrates that effective vision-language pre-training can be achieved by training only a lightweight bridging module (Q-Former) between frozen unimodal models, using a two-stage strategy that first aligns visual features with text and then adapts them for LLM interpretation. This approach dramatically reduces computational cost while achieving state-of-the-art results and enabling new capabilities like zero-shot instructed image-to-text generation. The key insight is that learning to query and extract relevant visual information is more efficient than retraining entire large-scale models end-to-end.
