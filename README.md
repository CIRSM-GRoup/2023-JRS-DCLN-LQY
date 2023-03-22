# 2023-JRS-DCLN-LQY
高光谱遥感影像分类
 # Deep Contrastive Learning Network for Small-Sample Hyperspectral Image Classification
 The source of our paper: 
[https://spj.science.org/doi/epdf/10.34133/remotesensing.0025]


## Description

The DCLN is method for small-sample HSI classification. It can realize effective spatial–spectral feature extraction, pseudo-label learning, and classification in the case of limited training samples.


##Model

<img src="figs/Model.png"/>


## Prerequisites

- [Anaconda 3]
- [Pytorch 1.7]
- [CUDA 10.1]
- [sklearn 0.23.2]

## dataset

You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./data` folder.

An example dataset folder has the following structure:
```
data
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
├── salinas
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
├── hou
│   ├── houston.mat
│   └── houston_gt.mat
└── paviaU
    ├── paviaU_gt.mat
    └── paviaU.mat
```

## Usage:

Take DCLN method on the UP dataset as an example: 
1. Download the required data set and move to folder`./data`.
2. Install the requirements : conda env create -f environment.yml.
3. Taking 5 labeled samples per class as an example, run `train.py` to train the model.
4.  run `test.py` and get the results.


##Citation

lf you use DCLN code in your research, we would appreciate a citation to the original paper:

“Liu Q, Peng J, Zhang G, Sun W, Du Q. Deep Contrastive Learning Network for Small-Sample Hyperspectral Image 
Classification. J. Remote Sens. 2023;3:Article 0025. https://doi.org/10.34133/remotesensing.0025”


##Contact
Quanyong Liu, 584298639@qq.com
