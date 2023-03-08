# RSNABreast7thPlace
## RSNA Screening Mammography Breast Cancer Detection
The kaggle competition overview is [here](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/overview). This repository is for our 7th solution (Team: luddite&MT) writeup. Short solution summary is here([solution summary](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/391125)).

## Preparation
1. Please make sure to put `train.csv` downloaded from kaggle in `data/input`
2. Please refer to [here](https://www.kaggle.com/code/masato114/rsna-generate-train-images/notebook) to prepare train images with H1520xW912 and put them into `data/input/train_images` directory.
3. [Option] If you would like to use sigmoid-windowing applied images, please refer to [here](https://www.kaggle.com/code/masato114/rsna-get-windowing-stat/notebook) to get windowing information in advance. If the given information are added on the dataframe from `train.csv` as new columns, `train.csv` of step1 can be replaced by this.

## Train

## Inference
- To complete inference faster, we compiled the pytorch models with [Torch-tensorRT](https://pytorch.org/TensorRT/) in advance. I noticed that the compilation did not work as usual for th 'tf' type of EfficientNet due to its dynamic padding function, so I edited the source code and used it. See [here](https://www.kaggle.com/code/masato114/rsna-tf-efficientnetv2s-tensorrt/notebook).
