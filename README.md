# Constraint-Loss-AAAI-2020

The source code and dataset for our paper "[Integrating Relation Constraints with Neural Relation Extractors](https://arxiv.org/abs/1911.11493)" which is publicated at AAAI 2020.

### Methods:

In this paper, we propose a unified framework to effectively integrate discrete relation constraints with neural networks for relation extraction. Specifically, we develop two approaches to evaluate how well *NRE* predictions satisfy our relation constraints in a batch-wise, from both general and precise perspectives. We explore our approach on English and Chinese dataset, and the experimental results show that our approach can help the base *NRE* models to effectively learn from the discrete relation constraints, and outperform popular *NRE* models as well as their *ILP* enhanced versions. Our study reveals that learning with the constraints can better utilize the constraints from a different perspective compared to the *ILP* post-processing method.

And the framework and experimental results are shown as follows:

![Model Framework](https://github.com/PKUYeYuan/Constraint-Loss-AAAI-2020/blob/master/FrameworkAndExpFigures/FrameworkFigure.jpg)

Fig1: Model Framework 

![Experimental Results](https://github.com/PKUYeYuan/Constraint-Loss-AAAI-2020/blob/master/FrameworkAndExpFigures/ExperimentResult.jpg)

Fig2: Experimental Results

### Requirements:

Python version >= 3.5 (recommended)

Tensorflow version == 1.13.1

### Usage:

- ACNN model
  - Configuration instruction:
    - "**-mode**" argument is used to specify "Train" or "Test"
    - "**-train_mode**" argument is used to specify the training mode ("Base" or "Sem" or "Coh")
    - "**-dataset**" argument is used to decide the dataset to use (English or Chinese) 
    - And other configurations can be find in our source code.
  - Some running examples:
    - *Train* a model on *English dataset* in *Sem* mode: 
      - First you need go into the ACNN/src directory, and then run the following command:
      - python main.py -mode=Train -dataset=English -train_mode=Sem
    - *Test* a model on *English dataset* using pre-trained models (You need to download the pre-trained models from [here](https://drive.google.com/drive/folders/1-Fs1fI6j_ZRzaBN8mnvD5Bv0XrWqryrZ?usp=sharing) and put them in ACNN/SavedModel)
      - First you need go into the ACNN/src directory, and then run the following command:
      - python main.py -mode=Test -dataset=English
- APCNN model
  - Configuration instruction:
    - "**-mode**" argument is used to specify "Train" or "Test"
    - "**-train_mode**" argument is used to specify the training mode ("Base" or "Sem" or "Coh")
    - "**-dataset**" argument is used to decide the dataset to use (English or Chinese) 
    - And other configurations can be find in our source code.
  - Some running examples:
    - *Train* a model on *English dataset* in *Sem* mode: 
      - First you need go into the APCNN/src directory, and then run the following command:
      - python main.py -mode=Train -dataset=English -train_mode=Sem
    - *Test* a model on *English dataset* using pre-trained models (You need to download the pre-trained models from [here](https://drive.google.com/drive/folders/1-Fs1fI6j_ZRzaBN8mnvD5Bv0XrWqryrZ?usp=sharing) and put them in APCNN/SavedModel)
      - First you need go into the APCNN/src directory, and then run the following command:
      - python main.py -mode=Test -dataset=English

### Datasets and Pre-trained Models:

Note that here we only exhibit some demo data to make others can run the code successfully. And we put the whole dataset and pre-trained models on [google cloud](https://drive.google.com/drive/folders/1-Fs1fI6j_ZRzaBN8mnvD5Bv0XrWqryrZ?usp=sharing), you can download them and put them in the corresponding directories to reproduce the experimental results presented in our paper.

### Citation:

If you use the code, please cite the following paper: **"[Integrating Relation Constraints with Neural Relation Extractors](https://arxiv.org/abs/1911.11493)"**

```reStructuredText
@article{ye2019integrating,
  title={Integrating Relation Constraints with Neural Relation Extractors},
  author={Ye, Yuan and Feng, Yansong and Luo, Bingfeng and Lai, Yuxuan and Zhao, Dongyan},
  journal={arXiv preprint arXiv:1911.11493},
  year={2019}
}
```

