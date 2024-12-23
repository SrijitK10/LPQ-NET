### Using Local Phase Quantization (LPQ) for Detecting Fake Faces Online

This repository contains the code for the paper "Using Local Phase Quantization (LPQ) for Detecting Fake Faces Online" by Srijit Kundu, Tanusree Ghosh and Ruchira Naskar. The paper has been accepted at the 2024 IEEE Region 10 Conference (TENCON 2024). 

### Abstract

<div style="text-align: justify;">
The rapid advancement of Generative AI, especially
Generative Adversarial Networks (GANs), has increased the issue of fake news on Online Social Networks (OSNs) by generating deceptive face images for social media profiles. Although existing detection methods are accurate, their effectiveness decreases when images are post-processed, which is common on OSNs. In this paper, we present LPQ-Net, a model combining Local Phase Quantization (LPQ) for feature extraction with a CNN-based classifier. We explore two variants: one sets a new benchmark in detecting StyleGAN2-generated images, and the other excels in identifying images shared on Facebook, WhatsApp, and Instagram. LPQ-Net also operates with minimal parameters, outperforming state-of-the-art methods and making it ideal for resource-constraint applications. Furthermore, our solution demonstrates its effectiveness by performing exceptionally well in detecting images generated by various Diffusion models.

We further show that incorporating LPQ features into fine-tuned classifiers like ResNet50, ResNet101, InceptionV3, and DenseNet121 significantly improves performance.
</div>

### Dataset

The dataset used in this paper is the FFHQ and styleGAN2 dataset, which can be downloaded from the following link: 
[FFHQ Dataset](https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL)
[styleGAN2](https://drive.google.com/drive/folders/1-5oQoEdAecNTFr8zLk5sUUvrEUN4WHXa)

We have also used Wang et al.'s dataset, which can be downloaded from the following link:
[CNNDetection CVPR2020](https://github.com/peterwang512/CNNDetection)

### File Structure

The file structure of the repository is as follows:
```
LPQ-NET
│   README.md
│   requirements.txt
│   train.py
│   test.py
│   model.py
│
└───data
│   └───train
│   │   │   real
│   │   │   fake
│   │
│   └───validation
│   │   │   real
│   │   │   fake
│   │    
│   └───test
│   │   │   real
│   │   │   fake
│
└───models
│   │   LPQ_Net_Gray.keras
│   │   LPQ_Net_Color.keras
│
└───preprocessing
│   │   lpq.py
│
└───utils
│   │   filepaths_train.py
│   │   filepaths_test.py
│   │   filepaths_validation.py
│
└───csv
│   │   train.csv
│   │   test.csv
│   │   validation.csv
```
### Setup

1. Clone the repository
```bash
git clone https://github.com/SrijitK10/LPQ-NET.git
```
2. Create a virtual environment using the following command:
```bash
conda create -n lpqnet python=3.9
```
3. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
4. Run the following command to create the CSV files:
```bash
cd utils
python3 filepaths_train.py
python3 filepaths_test.py
python3 filepaths_validation.py
```

This should arrange the file paths in the CSV files as follows:


| Path                          | Truth |
|--------------------------------|-------|
| ./datasets/train/fake/000248.png| 0     |
| ./datasets/train/fake/002054.png| 0     |
| ./datasets/train/real/01426.png | 1     |
| ./datasets/train/real/01582.png | 1     |
| ./datasets/train/real/02444.png | 1     |
| ./datasets/train/real/02673.png | 1     |
| ./datasets/train/real/01015.png | 1     |
| ./datasets/train/fake/002518.png| 0     |


Similarly, the test and validation CSV files will be created.

### Usage

1. To train the model, run the following command:
```bash
python3 train.py
```
2. To test the model, run the following command:
```bash
python3 test.py
```



