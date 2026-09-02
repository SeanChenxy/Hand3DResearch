# The "Something Something" Video Database for Learning and Evaluating Visual Common Sense

**Authors:** Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, Florian Hoppe, Christian Thurau, Ingo Bax, Roland Memisevic  
**Date:** 2017-06-13  
**Identifier:** [arXiv:1706.04261](https://arxiv.org/abs/1706.04261); DOI `10.1109/ICCV.2017.622`  
**Zotero item:** `EP38D6RH` ([Zotero](zotero://select/library/items/EP38D6RH))  
**Evidence status:** Zotero metadata, abstract, and PDF extraction were verified.  

## Summary

The "something-something" database is a crowdsourced corpus of 108,499 short videos (2 to 6 seconds, 4.03 s on average) labeled with 174 caption-template classes such as "Dropping [something] into [something]", where crowd workers act out the templates with everyday objects and type in the object nouns themselves. Because success requires reading fine-grained physical interactions between hands and objects rather than scene context or object identity alone, the dataset targets visual common sense about the physical world, and baseline experiments show it is dramatically harder for standard 2D and 3D CNNs than conventional action recognition.

## Background and Motivation

The authors argue that progress toward common sense reasoning about the physical world is blocked by existing video datasets, which label high-level activities (sports, human-centered activities) that can largely be classified from appearance and context, letting networks avoid modeling motion, object properties, affordances, and intuitive physics. Videos of hands manipulating objects contain exactly this information, but labels must force the network to attend to detailed physical aspects. Their solution inverts the usual pipeline: instead of annotating found footage, crowd workers are asked to record videos according to prescribed caption templates, guaranteeing that every video instantiates a fine-grained physical concept. The dataset is framed as an ongoing "curriculum learning" effort, starting with simple verb-noun-preposition templates and increasing complexity as models improve.

## Dataset Construction

Videos are collected through a custom crowdsourcing platform: workers select from grouped action templates, act them out with objects of their choosing, and enter the noun phrases into placeholder masks when uploading; a batch submission workflow lets them gather objects and record over multiple sessions. Quality control combines automated checks on video length and uniqueness with verification by human operators. The result is 108,499 videos over 174 template classes with an average of roughly 620 videos per class (minimum 77 for "Poking a hole into [some substance]", maximum 986 for "Holding [something]"), produced by 1,133 workers with an average of 127.32 workers per class; the data contain 23,137 distinct object name variants (including case, stemming, and determiner differences). The 8:1:1 train/validation/test split is constructed so that all videos from one worker fall into a single split. Two labeling-design mechanisms combat dataset bias and shortcut learning: action groups that bundle visually similar actions with minor physical differences, and contrastive "pretending" actions (for example, pretending versus actually putting something behind something) that force models to track object presence and outcome rather than hand pose or camera cues.

## Evaluation Protocol

The paper defines template-classification baselines on the full 174 classes and on hand-picked subsets of 10 easy classes (28,198 videos) and 40 classes (53,267 videos). Preprocessing samples frames at 24 fps, resizes to 84 x 84, and low-pass filters temporally for a 6 fps target. Encodings compared are: VGG-16 2D-CNN features averaged over frames (trained from scratch and ImageNet-pretrained), the pretrained 2D-CNN followed by an LSTM, a 3D-CNN trained from scratch with stacked clip features, a Sports-1M-pretrained 3D-CNN with averaged features, and a 2D+3D-CNN combination concatenating both encodings. Metrics are top-1 and top-2 error, with top-5 error on the 174-class task. An informal human evaluation asks 10 individuals to classify about 700 test samples across all 174 classes.

## Findings and Analysis

The best baseline, the combined 2D+3D-CNN, reaches 44.9% top-1 error on the 10-class subset and 63.8% on 40 classes, while on all 174 classes the best model (Sports-1M-pretrained 3D-CNN) attains 88.5% top-1 error, 81.5% top-2 error, and 70.0% top-5 error as reported in the results table (the text cites 70.3% top-5). Human evaluators reach approximately 60% accuracy, so even the strongest architecture of the time remains more than 40 error points behind humans, and many of the deliberately confusable classes (within action groups and pretending pairs) are "hardly distinguishable" with these architectures. The authors conclude that the dataset demands genuinely spatiotemporal, physics-aware features, and that 3D convolutions outperform 2D features with temporal pooling, with the combination working best.

## Contributions

A large-scale video database of 108,499 crowdsourced, template-labeled clips emphasizing hand-object interaction and physical common sense; the template-plus-typed-noun labeling scheme that yields open-vocabulary object annotations (23,137 variants) atop 174 classes; dataset design mechanisms (action groups, pretending and contrastive actions, worker-disjoint splits) that suppress shortcut learning; a scalable crowd-acting platform with automated and human quality control; and baseline experiments quantifying the difficulty gap between standard CNNs and human performance.

## Limitations

The paper states that the database is an ongoing collection effort, so the presented version is an early snapshot of a planned curriculum whose label complexity will grow over time. Label ambiguities complicate training and interpretation, which the authors mitigate with top-K error and class subsets rather than eliminate. The reported baselines are restricted to fairly standard 2D/3D CNN architectures, and the human evaluation is informal (about 700 samples, 10 raters), so the cited human accuracy is approximate.
