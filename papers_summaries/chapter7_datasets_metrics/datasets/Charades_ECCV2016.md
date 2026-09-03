# Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding

**Authors:** Gunnar A. Sigurdsson, Gül Varol, Xiaolong Wang, Ali Farhadi  
**Date:** 2016 (ECCV 2016)  
**Identifier:** [arXiv:1604.01753](https://arxiv.org/abs/1604.01753)  
**Zotero item:** No record found in the Zotero library; identity verified against arXiv metadata.  
**Evidence status:** No Zotero record; verified against full-text PDF extraction (arXiv 1604.01753).

## Summary
The paper addresses the absence of large-scale video data covering "boring" everyday indoor activities, which are underrepresented on YouTube, in movies, and in lab recordings. The authors propose the "Hollywood in Homes" approach, which crowdsources the entire video creation process on Amazon Mechanical Turk (AMT): workers write scripts from a controlled vocabulary, act out and record the videos in their own homes, and then verify and annotate the footage. The resulting Charades dataset contains 9,848 videos of average length 30.1 seconds recorded by 267 people on three continents, with free-text descriptions, 66,500 temporally localized intervals for 157 action classes, and 41,104 labels for 46 object classes. Baselines for action classification and sentence prediction are provided, and the low scores (17.2% mAP for the best single method on classification) show that everyday activity understanding with person-object interactions remains far from solved.

## Background and Motivation
Internet video datasets (UCF101, Sports-1M) are biased toward sports and entertaining actions because boring daily activities have no viewership and are rarely uploaded; movies (Hollywood, MPII-MD) remain entertaining and do not capture daily living; in-house datasets (MPII Cooking, TUM Breakfast, TACoS Multi-Level, ADL) control the domain well but lack diversity and scalability. The closest dataset, ActivityNet, is complementary but uncontrolled, biased toward non-boring actions, and professionally edited. The authors' motivation is to learn models biased toward real-world settings where robots operate, including object states, person-object interactions, and contextual common sense. Their solution distributes the "Hollywood filming process" to hundreds of workers' homes, gaining diversity like crowdsourced data while retaining control of scene, object, and action vocabulary like scripted collection.

## Dataset Construction
- **Source:** Worker-recorded videos, not web or movie footage. Workers film themselves for about 30 seconds following a script, in their own homes.
- **Script generation:** The vocabulary was derived by analyzing 549 movie scripts with term-frequency and TF-IDF, yielding 40 objects and 30 actions as seeds across 15 types of indoor rooms (Living Room, Home Office, Kitchen, Bathroom, etc.). Workers were shown one scene, 5 random objects, and 5 random actions, and asked to use 2 objects and 2 actions in a short paragraph about realistic everyday activities.
- **Recording economics:** A pilot found no worker would record videos until pay reached $3 per video; incentive engineering (sign-up bonuses raising new worker rate by 211%, "recruit a friend" bonuses, performance bonuses every 15th video increasing output per worker by 109%) reduced cost to about $1 per video. Collection peaked at 1,225 videos per day from 72 workers; the final cost split was 65% base pay, 21% performance bonuses, 11% recruitment bonuses, 3% verification. Recorded videos were verified by other workers selecting the matching script sentence from a line-up.
- **Annotations:** Per video: multiple free-text descriptions; action classes chosen from 157 (verb, proposition, noun) classes; verified interacted-object lists (46 object classes); and temporal start/end intervals for each present action. Over 15% of videos contain more than one person.
- **Scale:** 9,848 videos (7,985 train, 1,863 test); 27,847 descriptions; 66,500 temporally localized action intervals (49,809 train, 16,691 test); 41,104 object labels; an average of 6.8 relevant actions per video. Label precision is 95.6%, measured by an extra verification step and against ground truth from 19 annotation iterations on 50 videos.

## Evaluation Protocol
- **Splits:** Workers are randomly divided with 80% assigned to training, subject to four constraints: no worker appears in both train and test; category distributions of train and test are similar; each category has at least 6 test and 25 training videos; no single worker dominates the test set.
- **Action classification:** Given a video, identify which of the 157 action classes are present (a video has multiple labels). Metric: mean average precision (mAP). Baselines: random, C3D fc6 features, AlexNet and VGG-16 fc6 frame features, two-stream networks (VGG-16), a class-balanced two-stream variant, improved dense trajectories (IDT) with Fisher vectors, and a late-fusion combination of all methods.
- **Sentence prediction:** Generate a free-text sentence for a video, evaluated against the script (1 ground-truth sentence) and against worker descriptions (2.4 sentences on average), using CIDEr, BLEU1-4, ROUGE-L, and METEOR from the COCO Caption evaluation. Baselines: random words, random sentence, nearest neighbor (AlexNet fc7), S2VT, and human performance.
- The dataset also provides temporal intervals supporting localization, though the paper's baseline experiments cover classification and captioning.

## Findings and Analysis
- **Action classification (mAP on test):** Random 5.9%, C3D 10.9%, AlexNet 11.3%, balanced two-stream 11.9%, two-stream 14.3%, IDT 17.2%, Combined 18.6%. The authors note these scores are much lower than on most existing benchmarks; IDT outperforms all deep baselines at release.
- **Per-class behavior:** Best classes reach reasonable accuracy (Washing a window, 62.1% AP); there is a trend for larger classes to score higher, but small classes like Sitting in chair still perform in the top 15.
- **Error structure:** Confusion is concentrated among actions sharing the interacted object (e.g., putting versus taking clothes; Couch versus Bed actions); actions with no specific object of interaction (e.g., standing up, sneezing) are much easier, reaching 38.9% mAP versus the overall 18.6%. Fine-grained person-object interactions with the same object are the hardest cases.
- **Sentence prediction:** S2VT is the strongest baseline with CIDEr 0.17 on the script task and 0.14 on the description task, versus human CIDEr of 0.51 and 0.53 respectively. Generated captions are coherent but often lack relevance and overfit frequent patterns such as "drinking from a glass/cup"; CIDEr shows the highest agreement with human judgement among the metrics.

## Contributions
- The Hollywood in Homes data collection approach: crowdsourcing not just labeling but the full video creation pipeline (script, acting, verification) at a per-video cost of about one dollar.
- The first large-scale crowdsourced dataset of everyday household activities: 9,848 videos, 157 action classes, 46 object classes, 66,500 temporal intervals, 27,847 descriptions.
- Extensive baselines for action classification and video description generation, exposing the difficulty of daily activity and person-object interaction recognition.
- Public release of the dataset, code, and collection interfaces at http://allenai.org/plato/charades/.

## Limitations
- All videos are scripted and acted by workers rather than spontaneously occurring, so the distribution reflects imagined "boring realistic scenarios" and script vocabulary bias; the paper argues this is a controlled advantage but the enacted nature is inherent to the method.
- State-of-the-art baselines perform poorly (17.2-18.6% mAP classification), and the hardest failure mode—fine-grained actions on the same object—is not resolved by any tested method.
- Caption generation baselines produce coherent but insufficiently relevant sentences and overfit frequent patterns in the data.
- Label precision is 95.6%, leaving roughly 4% residual label noise.
- Recruitment was restricted to the US, Canada, UK, and (for a time) India, so geographic diversity is bounded even though 267 people across three continents participated.
