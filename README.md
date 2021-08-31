# Very Recent Progress in 3D Hand [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)

## Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Papers](#papers)
  - [Hand Mesh/Pose](#hand-meshpose)
  - [Hand-Object Interaction](#hand-object-interaction)
  - [Hand-Hand Interaction](#hand-hand-interaction)
  - [Two-Hand-Object Interaction](#two-hand-object-interaction)
  - [Full Body Reconstruction](#full-body-reconstruction)
  - [Tools](#tools)
- [benchmarks]()

## Overview

![alt img](Hand3DResearch.png)

## Dataset

|  Dataset   | Size  | Shape | Hand int. | Obj. int. | Motion | Synthetic | Link | 
|  ----  | ----  | ---- | ---- | ---- | ---- | ---- | ---- | 
|H2O     | [link](https://arxiv.org/pdf/2104.11181.pdf)   | 571K   | ✅  | ✅  | ✅  | ✅ | |
|HanCO   | [link](https://lmb.informatik.uni-freiburg.de/projects/contra-hand/)     | 860K   | ✅  |  |  | ✅ | |
|H2O     | [link](https://arxiv.org/pdf/2104.11466.pdf)   | 5M   | ✅  | ✅  | ✅  | ✅ | |
|H2O-3D  | [link](https://arxiv.org/pdf/2104.14639.pdf)     | 62K  | ✅ | ✅ | ✅ | ✅ | |
|GRAB    | [link](https://grab.is.tue.mpg.de/)     | 1.6M | ✅ | ✅ | ✅ | ✅ | |
|DexYCB  | [link](https://dex-ycb.github.io/?utm_source=catalyzex.com)     | 528K | ✅ | | ✅ | ✅ | |
|YoutubeHand | [link](https://github.com/arielai/youtube_3d_hands) | 47K  | ✅ |  |  | | |
| YCB-Affordance | [link](https://github.com/enriccorona/YCB_Affordance) | 134K  | ✅ | |✅ | ✅ | |
| HO3D   | [link](https://github.com/shreyashampali/ho3d) | 77K  | ✅ |  | ✅ | ✅ | |
|ContactPose | [link](https://contactpose.cc.gatech.edu/) | 2.9M | ✅ | ✅ | ✅ | ✅ | |
| FreiHAND | [link](https://lmb.informatik.uni-freiburg.de/projects/freihand/) | 134K | ✅ |  | | | |
|ObMan   | [link](https://hassony2.github.io/obman.html)  | 154K | ✅ | | ✅ | | ✅ |
|Interhand2.6M| [link](https://mks0601.github.io/InterHand2.6M) | 2.6M  | ✅ | ✅   |  |✅ | |
|RHD     | [link](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) | 44K  | | |  | | ✅ |
|H3D     | [link](https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html) | 22K  | ✅ | |  | | |
|MHP     | [link](http://www.rovit.ua.es/dataset/mhpdataset) | 80K  |  | |  | | |
|MVHM    | [link](https://github.com/Kuzphi/MVHM) | 320K | ✅ | |  | | ✅ |
|Dexter+Object|[link](https://handtracker.mpi-inf.mpg.de/projects/RealtimeHO/dexter+object.htm)| 3K   | | | ✅ |✅ | |
|EgoDexter |[link](https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm) | 3K   | | | | ✅ | |
|STB     | [link](https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/EgoDexter.htm) | 36K  | ✅ | | | ✅ | |
|FPHA    | [link](https://guiggh.github.io/publications/first-person-hands/) | 105K | | | | ✅ | |
|Tzionas et al.|[link](https://github.com/dimtziwnas/HandObjectInteractionIJCV16_HandMotionViewer?utm_source=catalyzex.com)| 36K | | ✅ | | ✅ | |
|Simon et al. | [link](https://github.com/laobaiswag/openpose1?utm_source=catalyzex.com) | 15K  | | ✅ | | ✅ | |
|GANerated Hands|  [link](https://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/) | 331K | | | ✅ | ✅  | |
|SynthHands | [link](https://handtracker.mpi-inf.mpg.de/projects/OccludedHands/) | 220K | | | ✅ | | ✅ |


## Papers

### Hand Mesh/Pose

#### Regression

+ [H.RG.9] Contrastive Representation Learning for Hand Shape Estimation. arXiv21.
  [[PDF](https://arxiv.org/pdf/2106.04324.pdf)]
  [[Project](https://lmb.informatik.uni-freiburg.de/projects/contra-hand/)]
  [[Code](https://github.com/lmb-freiburg/contra-hand)] \
  *Christian Zimmermann, Max Argus, Thomas Brox*

+ [H.RG.8] Hand Image Understanding via Deep Multi-Task Learning. ICCV21.
  [[PDF](https://arxiv.org/pdf/2107.11646.pdf)] \
  *Xiong Zhang, Hongsheng Huang, Jianchao Tan, Hongmin Xu, Cheng Yang, Guozhu Peng, Lei Wang, Ji Liu*

+ [H.RG.7] DeepHandMesh: Weakly-supervised Deep Encoder-Decoder Framework for High-fidelity Hand Mesh Modeling from a Single RGB Image. ECCV20.
  [[PDF](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470426.pdf)]
  [[Project](https://mks0601.github.io/DeepHandMesh/)]
  *Gyeongsik Moon, Takaaki Shiratori, Kyoung Mu Lee* 

+ [H.RG.6] Knowledge as Priors: Cross-Modal Knowledge Generalization for Datasets without Superior Knowledge. CVPR20. 
  [[PDF](https://arxiv.org/pdf/2004.00176.pdf)] \
  Long Zhao, Xi Peng, Yuxiao Chen, Mubbasir Kapadia, Dimitris N. Metaxas

+ [H.RG.5] FreiHAND: A Dataset for Markerless Capture of Hand Pose and Shape from Single RGB Images. ICCV19.
  [[PDF](https://arxiv.org/pdf/1909.04349.pdf)]
  [[Project](https://lmb.informatik.uni-freiburg.de/projects/freihand/)][Code](https://github.com/lmb-freiburg/freihand)] \
  *Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, Thomas Brox*

+ [H.RG.4] End-to-end Hand Mesh Recovery from a Monocular RGB Image. ICCV19.
  [[PDF](https://arxiv.org/pdf/1902.09305.pdf)]
  [[Code](https://github.com/MandyMo/HAMR)] \
  Xiong Zhang*, Qiang Li*, Wenbo Zhang, Wen Zheng

+ [H.RG.3] 3D Hand Shape and Pose from Images in the Wild. CVPR19.
  [[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Boukhayma_3D_Hand_Shape_and_Pose_From_Images_in_the_Wild_CVPR_2019_paper.pdf)]
  [[code](https://github.com/boukhayma/3dhand)] \
  *Adnane Boukhayma, Rodrigo de Bem, Philip H.S. Torr*

+ [H.RG.2] Pushing the Envelope for RGB-based Dense 3D Hand Pose Estimation via Neural Rendering. CVPR19.
  [[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Baek_Pushing_the_Envelope_for_RGB-Based_Dense_3D_Hand_Pose_Estimation_CVPR_2019_paper.pdf)] \
  *Seungryul Baek, Kwang In Kim, Tae-Kyun Kim*
  
+ [H.RG.1] GANerated Hands for Real-Time 3D Hand Tracking from Monocular RGB. CVPR18.
  [[PDF](http://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/content/GANeratedHands_CVPR2018.pdf)]
  [[Supp](http://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/content/GANeratedHands_CVPR2018_Supp.pdf)]
  [[Project](http://handtracker.mpi-inf.mpg.de/projects/GANeratedHands/)] \
  *Franziska Mueller, Florian Bernard, Oleksandr Sotnychenko, Dushyant Mehta, Srinath Sridhar, Dan Casas, Christian Theobalt*

#### Inverse Kinematics

+ [H.IK.4] HandTailor: Towards High-Precision Monocular 3D Hand Recovery. arXiv21.
  [[PDF](https://arxiv.org/pdf/2102.09244.pdf)]
  [Code](https://github.com/LyuJ1998/HandTailor) \
  *Jun Lv, Wenqiang Xu, Lixin Yang, Sucheng Qian, Chongzhao Mao, Cewu Lu*
  
+ [H.IK.3] HybrIK: A Hybrid Analytical-Neural Inverse Kinematics Solution for 3D Human Pose and Shape Estimation. CVPR21.
  [[PDF](https://arxiv.org/pdf/2011.14672.pdf)]
  [[Code](https://github.com/Jeff-sjtu/HybrIK)] \
  *Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin Yang, Cewu Lu*

+ [H.IK.2] BiHand: Recovering Hand Mesh with Multi-stage Bisected Hourglass Networks. BMVC20.
  [[PDF](https://arxiv.org/pdf/2008.05079.pdf)]
  [[Code](https://github.com/lixiny/bihand)]
  *Lixin Yang, Jiasen Li, Wenqiang Xu, Yiqun Diao, Cewu Lu*

+ [H.IK.1] Monocular Real-time Hand Shape and Motion Capture using Multi-modal Data. CVPR20.
  [[PDF](https://arxiv.org/pdf/2003.09572.pdf)]
  [[Project](https://calciferzh.github.io/publications/zhou2020monocular)]
  [[Code](https://github.com/CalciferZh/minimal-hand)]
  *Yuxiao Zhou, Marc Habermann, Weipeng Xu, Ikhsanul Habibie, Christian Theobalt, Feng Xu*

#### Graph

+ [H.GH.8] Point2Skeleton: Learning Skeletal Representations from Point Clouds. CVPR21.
  [[PDF](https://arxiv.org/pdf/2012.00230.pdf)]
  [[Code](https://github.com/clinplayer/Point2Skeleton?utm_source=catalyzex.com)] \
  *Cheng Lin, Changjian Li, Yuan Liu, Nenglun Chen, Yi-King Choi, Wenping Wang*

+ [H.GH.7] DC-GNet: Deep Mesh Relation Capturing Graph Convolution Network for 3D Human Shape Reconstruction
  [[PDF](https://arxiv.org/pdf/2108.12384.pdf)] \
  *Shihao Zhou, Mengxi Jiang, Shanshan Cai, Yunqi Lei*

+ [H.GH.6] MVHM: A Large-Scale Multi-View Hand Mesh Benchmark for Accurate 3D Hand Pose Estimation. WACV21.
  [[PDF](https://arxiv.org/abs/2012.03206)] \
  *Liangjian Chen, Shih-Yao Lin, Yusheng Xie, Yen-Yu Lin, Xiaohui Xie*

+ [H.GH.5] Camera-Space Hand Mesh Recovery via Semantic Aggregation and Adaptive 2D-1D Registration. CVPR21. 
  [[PDF](https://arxiv.org/pdf/2103.02845.pdf)]
  [[Code]](https://github.com/SeanChenxy/HandMesh) \
  *Xingyu Chen, Yufeng Liu, Chongyang Ma, Jianlong Chang, Huayan Wang, Tian Chen, Xiaoyan Guo, Pengfei Wan, Wen Zheng*
 
+ [H.GH.4] Pose2Mesh: Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose. ECCV20.
  [[PDF](https://arxiv.org/pdf/2008.09047.pdf)]
  [[Code](https://github.com/hongsukchoi/Pose2Mesh_RELEASE?utm_source=catalyzex.com)] \
  *Hongsuk Choi, Gyeongsik Moon, Kyoung Mu Lee*

+ [H.GH.3] Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild. CVPR20.
  [[PDF](https://arxiv.org/pdf/2004.01946.pdf)]
  [[Project](https://www.arielai.com/mesh_hands/)] \
  *Dominik Kulon, Riza Alp Güler, Iasonas Kokkinos, Michael Bronstein, Stefanos Zafeiriou*

+ [H.GH.2] Exploiting Spatial-temporal Relationships for 3D Pose Estimation via Graph Convolutional Networks. ICCV19.
  [[PDF](https://cse.buffalo.edu/~jsyuan/papers/2019/Exploiting_Spatial-temporal_Relationships_for_3D_Pose_Estimation_via_Graph_Convolutional_Networks.pdf)] \
  *Yujun Cai, Liuhao Ge, Jun Liu, Jianfei Cai, Tat-Jen Cham, Junsong Yuan, and Nadia Magnenat Thalmann*

+ [H.GH.1] 3D Hand Shape and Pose Estimation from a Single RGB Image. CVPR19.
  [[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_3D_Hand_Shape_and_Pose_Estimation_From_a_Single_RGB_CVPR_2019_paper.pdf)]
  [[Project](https://sites.google.com/site/geliuhaontu/home/cvpr2019)]
  [[Code](https://github.com/3d-hand-shape/hand-graph-cnn)]
  *Liuhao Ge, Zhou Ren, Yuncheng Li, Zehao Xue, Yingying Wang, Jianfei Cai, Junsong Yuan*

#### Transformer

+ [H.TR.4] Mesh Graphormer. ICCV21.
  [[PDF](https://arxiv.org/pdf/2104.00272.pdf)] \
  *Kevin Lin, Lijuan Wang, Zicheng Liu*

+ [H.TR.3] End-to-End Human Pose and Mesh Reconstruction with Transformers. CVPR21.
  [[PDF](https://arxiv.org/pdf/2012.09760.pdf)]
  [[Code](https://github.com/microsoft/MeshTransformer)] \
  *Kevin Lin, Lijuan Wang, Zicheng Liu*

+ [H.TR.2] Hand-Transformer: Non-Autoregressive Structured Modeling for 3D Hand Pose Estimation. ECCV20.
  [[PDF](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700018.pdf)] \
  *Lin Huang, Jianchao Tan, Ji Liu, and Junsong Yuan*

+ [H.TR.1] Epipolar Transformers. CVPR20.
  [[PDF](https://arxiv.org/pdf/2005.04551.pdf)]
  [[Code](https://github.com/yihui-he/epipolar-transformers)] \
  *Yihui He*, Rui Yan*, Shoou-I Yu, Katerina Fragkiadaki*

#### 2.5D

+ [H.VX.5] HandVoxNet++: 3D Hand Shape and Pose Estimation using Voxel-Based Neural Networks.
  [[PDF](https://arxiv.org/pdf/2107.01205.pdf)] \
  *Jameel Malik, Soshi Shimada, Ahmed Elhayek, Sk Aziz Ali, Christian Theobalt, Vladislav Golyanik, Didier Stricker*

+ [H.VX.4] Exploiting Learnable Joint Groups for Hand Pose Estimation. AAAI21.
  [[PDF](https://arxiv.org/pdf/2012.09496.pdf)]
  [[Code](https://github.com/moranli-aca/LearnableGroups-Hand)]
  *Moran Li, Yuan Gao, Nong Sang*

+ [H.VX.3] I2L-MeshNet: Image-to-Lixel Prediction Network for Accurate 3D Human Pose and Mesh Estimation from a Single RGB Image. ECCV20.
  [[PDF](https://arxiv.org/abs/2008.03713)]
  [[Code](https://github.com/mks0601/I2L-MeshNet_RELEASE)] \
  *Gyeongsik Moon, Kyoung Mu Lee*

+ [H.VX.2] HandVoxNet: Deep Voxel-Based Network for 3D Hand Shape and Pose Estimation from a Single Depth Map. CVPR20.
  [[PDF](https://arxiv.org/pdf/2004.01588.pdf)] \
  *Jameel Malik, Ibrahim Abdelaziz, Ahmed Elhayek, Soshi Shimada, Sk Aziz Ali, Vladislav Golyanik, Christian Theobalt, Didier Stricker*

+ [H.VX.1] Hand Pose Estimation via Latent 2.5D Heatmap Regression. ECCV18.
  [[PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Umar_Iqbal_Hand_Pose_Estimation_ECCV_2018_paper.pdf)] \
  *Umar Iqbal, Pavlo Molchanov, Thomas Breuel, Juergen Gall, Jan Kautz*

#### UV

+ [H.UV.2] I2UV-HandNet: Image-to-UV Prediction Network for Accurate and High-fidelity 3D Hand Mesh Modeling. ICCV21.
  [[PDF](https://arxiv.org/pdf/2102.03725.pdf)] \
  *Ping Chen, Yujin Chen, Dong Yang, Fangyin Wu, Qin Li, Qingpei Xia, Yong Tan*

+ [H.UV.1] HTML: A Parametric Hand Texture Model for 3D Hand Reconstruction and Personalizationm. ECCV20.
  [[PDF](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560052.pdf)]
  [[Project](https://handtracker.mpi-inf.mpg.de/projects/HandTextureModel/)]
  *Neng Qian, Jiayi Wang, Franziska Mueller, Florian Bernard, Vladislav Golyanik, Christian Theobalt*

#### Weak Supervision

+ [H.WS.7] Self-Supervised 3D Hand Pose Estimation from Monocular RGB via Contrastive Learning. arXiv21.
  [[PDF](https://arxiv.org/pdf/2106.05953.pdf)] \
  Adrian Spurr, Aneesh Dahiya, Xucong Zhang, Xi Wang, Otmar Hilliges

+ [H.WS.6] Adversarial Motion Modelling helps Semi-supervised Hand Pose Estimation. arXiv21.
  [[PDF](https://arxiv.org/pdf/2106.05954.pdf)] \
  *Adrian Spurr, Pavlo Molchanov, Umar Iqbal, Jan Kautz, Otmar Hilliges*

+ [H.WS.5] SemiHand: Semi-supervised Hand Pose Estimation with Consistency. ICCV21. \
  *Linlin Yang, Shicheng Chen, Angela Yao*

+ [H.WS.4] Model-based 3D Hand Reconstruction via Self-Supervised Learning. CVPR21.
  [[PDF](https://arxiv.org/pdf/2103.11703.pdf)] \
  *Yujin Chen, Zhigang Tu, Di Kang, Linchao Bao, Ying Zhang, Xuefei Zhe, Ruizhi Chen, Junsong Yuan*

+ [H.WS.3] Weakly-Supervised 3D Hand Pose Estimation via Biomechanical Constraints. ECCV20.
  [[PDF](https://arxiv.org/pdf/2003.09282.pdf)] \
  *Adrian Spurr, Umar Iqbal, Pavlo Molchanov, Otmar Hilliges, Jan Kautz*

+ [H.WS.2] Dual Grid Net: hand mesh vertex regression from single depth maps. ECCV20.
  [[PDF](https://arxiv.org/pdf/1907.10695.pdf)] \
  *Chengde Wan, Thomas Probst, Luc Van Gool, Angela Yao*

+ [H.WS.1] Adaptive Wasserstein Hourglass for Weakly Supervised Hand Pose Estimation from Monocular RGB. MM20.
  [[PDF](https://arxiv.org/pdf/1909.05666.pdf)] \
  Yumeng Zhang, Li Chen, Yufeng Liu, Junhai Yong, Wen Zheng

#### Temporal

+ [H.TP.5] TravelNet: Self-supervised Physically Plausible Hand Motion Learning from Monocular Color Images. ICCV21.
  [[PDF](https://www.yangangwang.com/papers/ZHAO-TRAVEL-2021-08.pdf)] \
  *Zimeng Zhao, Xi Zhao and Yangang Wang*

+ [H.TP.4] Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video. CVPR21
  [[PDF](https://arxiv.org/pdf/2011.08627.pdf)]
  [[Code](https://github.com/hongsukchoi/TCMR_RELEASE)] \
  *Hongsuk Choi, Gyeongsik Moon, Ju Yong Chang, Kyoung Mu Lee*
 
+ [H.TP.3] Temporal-Aware Self-Supervised Learning for 3D Hand Pose and Mesh Estimation in Videos. WACV21.
  [[PDF](https://arxiv.org/pdf/2012.03205.pdf)] \
  Liangjian Chen, Shih-Yao Lin, Yusheng Xie, Yen-Yu Lin, Xiaohui Xie

+ [H.TP.2] Adaptive Computationally Efficient Network for Monocular 3D Hand Pose Estimation. ECCV20.
  [[PDF](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490120.pdf)] \
  *Zhipeng Fan, Jun Liu, Yao Wang*

+ [H.TP.1] SeqHAND: RGB-Sequence-Based 3D Hand Pose and Shape Estimation. ECCV20.
  [[PDF](https://arxiv.org/pdf/2007.05168.pdf)] \
  *John Yang, Hyung Jin Chang, Seungeui Lee, Nojun Kwak*

#### SDF

+ [H.SDF.1] PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization. CVPR19
  [[PDF](https://arxiv.org/pdf/1905.05172.pdf)]
  [[Code]](https://github.com/shunsukesaito/PIFu) \
  *Shunsuke Saito, Zeng Huang, Ryota Natsume, Shigeo Morishima, Angjoo Kanazawa, Hao Li*

+ [H.SDF.2] PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization. CVPR20.
  [[PDF](https://arxiv.org/pdf/2004.00452.pdf)]
  [[Code]](https://github.com/facebookresearch/pifuhd?utm_source=catalyzex.com) \
  *Shunsuke Saito, Tomas Simon, Jason Saragih, Hanbyul Joo*

+ [H.SDF.3] Geo-PIFu: Geometry and Pixel Aligned Implicit Functions for Single-view Human Reconstruction. NeurIPS20.
  [[PDF](https://arxiv.org/pdf/2006.08072.pdf)]
  [[Code](https://github.com/simpleig/Geo-PIFu)] \
  *Tong He, John Collomosse, Hailin Jin, Stefano Soatto*

### Hand-Object Interaction

#### Regression

+ [HO.RG.5] HO-3D_v3: Improving the Accuracy of Hand-Object Annotations of the HO-3D Dataset. arXiv21.
  [[PDF](https://arxiv.org/pdf/2107.00887.pdf)] \
  Shreyas Hampali, Sayan Deb Sarkar, Vincent Lepetit

+ [HO.RG.4] Unsupervised Domain Adaptation with Temporal-Consistent Self-Training for 3D Hand-Object Joint Reconstruction. arXiv21.
  [[PDF](https://arxiv.org/pdf/2012.11260.pdf)] \
  *Mengshi Qi, Edoardo Remelli, Mathieu Salzmann, Pascal Fua*

+ [HO.RG.3] DexYCB: A Benchmark for Capturing Hand Grasping of Objects. CVPR21.
  [[PDF](https://dex-ycb.github.io/assets/chao_cvpr2021.pdf)]
  [[Project](https://dex-ycb.github.io/)]
  [[Code](https://github.com/NVlabs/dex-ycb-toolkit)] \
  *Yu-Wei Chao, Wei Yang, Yu Xiang, Pavlo Molchanov, Ankur Handa, Jonathan Tremblay, Yashraj S. Narang, Karl Van Wyk, Umar Iqbal, Stan Birchfield, Jan Kautz, Dieter Fox*

+ [HO.RG.2] HOnnotate: A method for 3D Annotation of Hand and Objects Poses. CVPR20.
  [[PDF](https://arxiv.org/pdf/1907.01481.pdf)]
  [[Project](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/)]
  [[Code](https://github.com/shreyashampali/ho3d)] \
  *Shreyas Hampali, Mahdi Rad, Markus Oberweger, Vincent Lepetit*

+ [HO.RG.1] Learning joint reconstruction of hands and manipulated objects. CVPR19.
  [[PDF](https://arxiv.org/pdf/1904.05767.pdf)]
  [[Project](https://hassony2.github.io/obman)]
  [[Code](https://github.com/hassony2/obman_train)] \
  *Yana Hasson, Gül Varol, Dimitris Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and Cordelia Schmid*

#### GAN

+ [HO.GAN.1] GanHand: Predicting Human Grasp Affordances in Multi-Object Scenes. CVPR20.
  [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Corona_GanHand_Predicting_Human_Grasp_Affordances_in_Multi-Object_Scenes_CVPR_2020_paper.pdf)]
  [[Code](https://github.com/enriccorona/GanHand)] \
  Enric Corona, Albert Pumarola, Guillem Alenya, Francesc Moreno-Noguer, Gregory Rogez

#### SDF

+ [HO.SDF.1] Grasping Field: Learning Implicit Representations for Human Grasps.
  [[PDF](https://arxiv.org/pdf/2008.04451.pdf)]
  [[Code](https://github.com/korrawe/grasping_field)] \
  *Korrawe Karunratanakul, Jinlong Yang, Yan Zhang, Michael Black, Krikamol Muandet, Siyu Tang*


#### Weak Supervision

+ [HO.WS.3] Reconstructing Hand-Object Interactions in the Wild. ICCV21.
  [[PDF](https://arxiv.org/pdf/2012.09856.pdf)]
  [[Code]](https://people.eecs.berkeley.edu/~zhecao/rhoi/) \
  *Zhe Cao*, Ilija Radosavovic*, Angjoo Kanazawa, Jitendra Malik*

+ [HO.WS.2] Semi-Supervised 3D Hand-Object Poses Estimation with Interactions in Time. CVPR21.
  [[PDF](https://arxiv.org/pdf/2106.05266.pdf)]
  [[Project](https://stevenlsw.github.io/Semi-Hand-Object/)]
  [[Code](https://github.com/stevenlsw/Semi-Hand-Object)] \
  *Shaowei Liu*, Hanwen Jiang*, Jiarui Xu, Sifei Liu, Xiaolong Wang*

+ [HO.WS.1] Weakly-supervised Domain Adaptation via GAN and Mesh Model for Estimating 3D Hand Poses Interacting Objects. CVPR20.
  [[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Baek_Weakly-Supervised_Domain_Adaptation_via_GAN_and_Mesh_Model_for_Estimating_CVPR_2020_paper.pdf)]
  [[Code](https://github.com/bsrvision/weak_da_hands)] \
  Seungryul Baek, Kwang In Kim, Tae-Kyun Kim

#### Temporal

+ [HO.TP.2] Towards unconstrained joint hand-object reconstruction from RGB videos. arXiv21.
  [[PDF](https://arxiv.org/pdf/2108.07044.pdf)]
  [[Project](https://hassony2.github.io/homan.html)]
  [[Code](https://github.com/hassony2/homan)] \
  *Yana Hasson, Gül Varol, Ivan Laptev, Cordelia Schmid*

+ [HO.TP.1] Leveraging Photometric Consistency over Time for Sparsely Supervised Hand-Object Reconstruction. CVPR20.
  [[PDF](https://arxiv.org/pdf/2004.13449.pdf)]
  [[Project](https://hassony2.github.io/handobjectconsist.html)]
  [[Code](https://github.com/hassony2/handobjectconsist)] \
  *Yana Hasson, Bugra Tekin, Federica Bogo, Ivan Laptev, Marc Pollefeys, Cordelia Schmid*

#### Joint Optimzation

+ [HO.OP.3] Perceiving 3D Human-Object Spatial Arrangements from a Single Image in the Wild. ECCV20.
  [[PDF](https://arxiv.org/pdf/2007.15649.pdf)]
  [[Code](https://github.com/facebookresearch/phosa?utm_source=catalyzex.com)] \
  *Jason Y. Zhang, Sam Pepose, Hanbyul Joo, Deva Ramanan, Jitendra Malik, Angjoo Kanazawa*

+ [HO.OP.2] Hand-Object Contact Consistency Reasoning for Human Grasps Generation. ICCV21.
  [[PDF](https://arxiv.org/pdf/2104.03304.pdf)]
  [[Project](https://hwjiang1510.github.io/GraspTTA/)] \
  *Hanwen Jiang, Shaowei Liu, Jiashun Wang, Xiaolong Wang*

+ [HO.OP.1] CPF: Learning a Contact Potential Field to Model the Hand-object Interaction. ICCV21.
  [[PDF](https://arxiv.org/pdf/2012.00924.pdf)]
  [[Code](https://github.com/lixiny/CPF)] \
  *Lixin Yang, Xinyu Zhan, Kailin Li, Wenqiang Xu, Jiefeng Li, Cewu Lu*

### Hand-Hand Interaction

#### Regression

+ [HH.RG.1] RGB2Hands: Real-Time Tracking of 3D Hand Interactions from Monocular RGB Video. SIGGRAPHAsia20
  [[PDF](https://handtracker.mpi-inf.mpg.de/projects/RGB2Hands/content/RGB2Hands_author_version.pdf)]
  [[Project](https://handtracker.mpi-inf.mpg.de/projects/RGB2Hands/)] \
  *Jiayi Wang, Franziska Mueller, Florian Bernard, Suzanne Sorli, Oleksandr Sotnychenko, Neng Qian, Miguel A. Otaduy, Dan Casas, and Christian Theobalt*

#### 2.5D

+ [HH.VX.3] Learning to Disambiguate Strongly Interacting Hands via Probabilistic Per-pixel Part Segmentation. arXiv21.
  [[PDF](https://arxiv.org/pdf/2107.00434.pdf)] \
  *Zicong Fan, Adrian Spurr, Muhammed Kocabas, Siyu Tang, Michael J. Black, Otmar Hilliges*

+ [HH.VX.2] Interacting Two-Hand 3D Pose and Shape Reconstruction from Single Color Image. ICCV21.
  [[PDF](https://www.yangangwang.com/papers/ZHANG-ITH-2021-08.pdf)]
  [[Project](https://baowenz.github.io/Intershape/)]
  [[Code](https://github.com/BaowenZ/Intershape)]
  *Baowen Zhang, Yangang Wang, Xiaoming Deng, Yinda Zhang, Ping Tan, Cuixia Ma and Hongan Wang*


+ [HH.VX.1] InterHand2.6M: A New Large-scale Dataset and Baseline for 3D Single and Interacting Hand Pose Estimation from a Single RGB Image. ECCV20.
  [[PDF](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650545.pdf)]
  [[Project](https://mks0601.github.io/InterHand2.6M/)]
  [[Code](https://github.com/facebookresearch/InterHand2.6M)] \
  *Gyeongsik Moon, Shoou-i Yu, He Wen, Takaaki Shiratori, Kyoung Mu Lee*

### Two-Hand-Object Interaction

#### Graph

+ [THO.GH.1] H2O: Two Hands Manipulating Objects for First Person Interaction Recognition. ICCV21.
  [[PDF](https://arxiv.org/pdf/2104.11181.pdf)]
  [[Project](https://www.taeinkwon.com/projects/h2o)]
  [[Code](https://github.com/taeinkwon/h2odataset)] \
  *Taein Kwon, Bugra Tekin, Jan Stuhmer, Federica Bogo, Marc Pollefeys*

#### Transformer

+ HandsFormer: Keypoint Transformer for Monocular 3D Pose Estimation ofHands and Object in Interaction. arXiv21.
  [[PDF](https://arxiv.org/pdf/2104.14639.pdf)] \
  *Shreyas Hampali, Sayan Deb Sarkar, Mahdi Rad, Vincent Lepetit*

#### GAN

+ [THO.GAN.2] H2O: A Benchmark for Visual Human-human Object Handover Analysis. arXiv21.
  [[PDF](https://arxiv.org/pdf/2104.11466.pdf)] \
  *Ruolin Ye, Wenqiang Xu, Zhendong Xue, Tutian Tang, Yanfeng Wang, Cewu Lu*

+ [THO.GAN.1] Body2Hands: Learning to Infer 3D Hands from Conversational Gesture Body Dynamics. CVPR21.
  [[PDF](https://arxiv.org/pdf/2007.12287.pdf)]
  [[Project](http://people.eecs.berkeley.edu/~evonne_ng/projects/body2hands/)] 
  [[Code](https://github.com/facebookresearch/body2hands)] \
  *Evonne Ng, Hanbyul Joo, Shiry Ginosar, Trevor Darrell*

### Full Body Reconstruction

#### Regression

+ [FBR.RG.3] FrankMocap: A Monocular 3D Whole-Body Pose Estimation System via Regression and Integration. arXiv21.
   [[PDF](https://arxiv.org/pdf/2108.06428.pdf)]
   [[Code](https://github.com/facebookresearch/frankmocap)] \
   *Yu Rong, Takaaki Shiratori, Hanbyul Joo*

+ [FBR.RG.2] Monocular Expressive Body Regression through Body-Driven Attention. ECCV20.
  [[PDF](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550018.pdf)]
  [[Project](https://expose.is.tue.mpg.de/en)]
  [[Code](https://github.com/vchoutas/expose)] \
  *Vasileios Choutas, Georgios Pavlakos, Timo Bolkart, Dimitrios Tzionas , Michael J. Black*

+ [FBR.RG.1] GRAB: A Dataset of Whole-Body Human Grasping of Objects. ECCV20
  [[PDF](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490562.pdf)]
  [[Project](https://grab.is.tue.mpg.de/)]
  [[Code](https://github.com/otaheri/GrabNet)] \
  *Omid Taheri, Nima Ghorbani, Michael J. Black, Dimitrios Tzionas*

#### Inverse Kinematics

+ [FBR.IK.2] Pose2Pose: 3D Positional Pose-Guided 3D Rotational Pose Prediction for Expressive 3D Human Pose and Mesh Estimation. arXiv20.
  [[PDF](https://arxiv.org/pdf/2011.11534.pdf)] \
  *Gyeongsik Moon, Kyoung Mu Lee*

+ [FBR.IK.1] Monocular Real-time Full Body Capture with Inter-part Correlations. CVPR21.
  [[PDF](https://arxiv.org/pdf/2012.06087.pdf)] \
  *Yuxiao Zhou, Marc Habermann, Ikhsanul Habibie, Ayush Tewari, Christian Theobalt, Feng Xu*

#### Optimization

+ [FBR.OP.2] Monocular Total Capture: Posing Face, Body, and Hands in the Wild. CVPR19.
  [[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiang_Monocular_Total_Capture_Posing_Face_Body_and_Hands_in_the_CVPR_2019_paper.pdf)]
  [[Project](http://domedb.perception.cs.cmu.edu/monototalcapture.html)]
  [[Code](https://github.com/CMU-Perceptual-Computing-Lab/MonocularTotalCapture)] \
  *Donglai Xiang, Hanbyul Joo, Yaser Sheikh*

+ [FBR.OP.1] Expressive Body Capture: 3D Hands, Face, and Body from a Single Image. CVPR19.
  [[PDF](https://arxiv.org/pdf/1904.05866)]
  [[Project](https://smpl-x.is.tue.mpg.de/)]
  [[Code](https://github.com/vchoutas/smplify-x)] \
  *Georgios Pavlakos*, Vasileios Choutas*, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, Michael J. Black*

### Tools

#### Model

+ [T.MD.3] [manotorch](https://github.com/lixiny/manotorch)
+ [T.MD.2] [manopth](https://github.com/hassony2/manopth)
+ [T.MD.1] [spheremesh](https://github.com/anastasia-tkach/hmodel-cpp-public)

#### MoCap

+ [T.MC.3] [frankmocap](https://github.com/facebookresearch/frankmocap)
+ [T.MC.2] [HandMotionViewer](https://github.com/dimtziwnas/HandObjectInteractionIJCV16_HandMotionViewer?utm_source=catalyzex.com)
+ [T.MC.1] [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
