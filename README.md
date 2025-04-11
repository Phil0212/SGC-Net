# SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection.
This is an official implementation for CVPR 2025 paper ["SGC-Net:Stratified Granular Comparison Network for Open-Vocabulary HOI Detection"](https://arxiv.org/pdf/2503.00414).
## Overview
**Abstract:** Recent open-vocabulary human-object interaction (OV-HOI) detection methods primarily rely on large language model (LLM) for generating auxiliary descriptions and leverage knowledge distilled from CLIP to detect unseen interaction categories. Despite their effectiveness, these methods face two challenges: (1) feature granularity deficiency, due to reliance on last layer visual features for text alignment, leading to the neglect of crucial object-level details from intermediate layers; (2) semantic similarity confusion, resulting from CLIP's inherent biases toward certain classes, while LLM-generated descriptions based solely on labels fail to adequately capture inter-class similarities. To address these challenges, we propose a stratified granular comparison network. First, we introduce a granularity sensing alignment module that aggregates global semantic features with local details, refining interaction representations and ensuring robust alignment between intermediate visual features and text embeddings. Second, we develop a hierarchical group comparison module that recursively compares and groups classes using LLMs, generating fine-grained and discriminative descriptions for each interaction category. Experimental results on two widely-used benchmark datasets, SWIG-HOI and HICO-DET, demonstrate that our method achieves state-of-the-art results in OV-HOI detection. 

![GitHub Logo](data/framework.png)

## Preparation

### Installation

Our codebase is built upon CLIP and requires the installation of PyTorch, torchvision, and a few additional dependencies.


```bash
conda create -n SGC-Net python==3.10
conda activate SGC-Net
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

### Dataset

The experiments are primarily conducted on the **HICO-DET** and **SWIG-HOI** datasets. We follow the instructions provided in this repository to prepare the [HICO-DET](https://github.com/YueLiao/PPDM) dataset, and this repository for the preparation of the [SWIG-HOI](https://github.com/scwangdyd/large_vocabulary_hoi_detection) dataset.

#### HICO-DET

The HICO-DET dataset can be downloaded from [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After downloading, extract the tarball `hico_20160224_det.tar.gz` into the `data` directory. We utilize the annotation files provided by the authors of [PPDM](https://github.com/YueLiao/PPDM) and reorganize these annotations by adding supplementary metadata, such as image width and height. These enhanced annotation files can be downloaded from [here](https://drive.google.com/open?id=1lqmevkw8fjDuTqsOOgzg07Kf6lXhK2rg). Please ensure that the downloaded files are placed according to the following structure. Alternatively, you can specify custom paths by modifying the default paths in [datasets/hico.py](./datasets/hico.py).



``` plain
 |─ data
 │   └─ hico_20160224_det
 |       |- images
 |       |   |─ test2015
 |       |   |─ train2015
 |       |─ annotations
 |       |   |─ trainval_hico_ann.json
 |       |   |─ test_hico_ann.json
 :       :
```

#### SWIG-DET

The SWIG-DET dataset can be downloaded from [here](https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip). After downloading, extract the `images_512.zip` file into the `data` directory. Annotation files can be downloaded from [here](https://drive.google.com/open?id=1GxNP99J0KP6Pwfekij_M1Z0moHziX8QN). Please ensure that the downloaded files are placed according to the following directory structure. Alternatively, you can modify the default paths to your custom locations in [datasets/swig.py](./datasets/swig.py)


``` plain
 |─ data
 │   └─ swig_hoi
 |       |- images_512
 |       |─ annotations
 |       |   |─ swig_train_1000.json
 |       |   |- swig_val_1000.json
 |       |   |─ swig_trainval_1000.json
 |       |   |- swig_test_1000.json
 :       :
```
## Build Discriminative Descriptions

Run this command to generate fine-grained descriptions for HICO-DET/SWIG-HOI dataset or download from [here]().

``` bash
python tools/main.py
```


## Training

Run this command to train the model in HICO-DET dataset

``` bash
bash ./build_tree/train_hico.sh --dataset_file [hico/swig]
```

Run this command to train the model in SWIG-HOI dataset

``` bash
bash ./tools/train_swig.sh
```


## Inference

Run this command to evaluate the model on HICO-DET dataset

``` bash
bash ./tools/test_hico.sh
```

Run this command to evaluate the model on SWIG-HOI dataset

``` bash
bash ./tools/test_swig.sh
```

## Models

| Dataset  | Unseen | Seen  | Full  | Ckeckpoint |
|:----------:|:--------:|:-------:|:-------:|:-------------:|
| HICO-Det | 23.27  | 28.34 | 27.22 |  [CKPT](https://pan.baidu.com/s/1E6F_aG_VTk-s93_DtRW57Q?pwd=guzd) |


| Dataset  | Non-Rare | Rare  | Unseen | Full  | Ckeckpoint |
|:--------:|:--------:|:-----:|:------:|:-----:|:-----------:|
| SWIG-HOI | 23.67    | 16.55 | 12.46  | 17.20 |[CKPT]()   |

## Citing
Please consider citing our paper if it helps your research.
```bash
@article{lin2025sgc,
  title={SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection},
  author={Lin, Xin and Shi, Chong and Yang, Zuopeng and Tang, Haojin and Zhou, Zhili},
  journal={arXiv preprint arXiv:2503.00414},
  year={2025}
} 
```

## Acknowledgments 
We gratefully acknowledge the authors of the following repositories, from which portions of our code are adapted.
+ [Lei's repository](https://github.com/ltttpku/CMD-SE-release)
+ [Wang's repository](https://github.com/scwangdyd/promting_hoi) 

