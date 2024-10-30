# UNeXt: Paper Implementation for EN3160 - Image Processing and Machine Vision

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Results](#results)
- [References](#references)

---

## Introduction

The **UNeXt** is lightweight, more efficient segmentation architecture that maintains state of the art accuracy. The UNeXt paper proposes a hybrid model that combines the strengths of convolutional layers with MLP layers to enhance feature extraction, all while reducing computational costs. This project aims to implement and replicate  the results of the the UNeXt model as a part of EN3160 Image Processing and Machine Vision module.


## Installation

To set up the environment, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/nadunnr/UNeXt.git
   cd UNeXt-EN3160
   ```

2. Install the required packages using `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the `train.py` file for model training:
   ```bash
   python3 train.py
   ```
4. Run the `val.py` file for model inferencing:
   ```bash
   python3 val.py
   ```


## Dataset

For this project, we will use the BUSI dataset, which provides annotated ultrasound images. The dataset should be organized in the following structure. Note that multiple annotation masks can be provided under nested folders inside `masks` folder. Rename them with class indeces.:

```plaintext
data/
├── images/
│   ├── 001.png
│   ├── 002.png
├── masks/
│   ├── 0/
│   |   ├── 001.png
|   |   ├── 002.png
```

Refer to the dataset provider’s terms and conditions for use. Ensure data preprocessing is consistent with the format expected by UNeXt.


## Results

TODO


## References

- [UNeXt: MLP-based Rapid Medical Image Segmentation Network](https://arxiv.org/abs/2203.04967)
- Official UNeXt Implementation: [GitHub Repository](hhttps://github.com/jeya-maria-jose/UNeXt-pytorch)