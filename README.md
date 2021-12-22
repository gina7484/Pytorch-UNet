# U-Net: Semantic segmentation with PyTorch
<a href="#"><img src="https://img.shields.io/github/workflow/status/milesial/PyTorch-UNet/Publish%20Docker%20image?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.9.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)



- [Getting Started](#getting-started)
- [Prepare Dataset](#prepare-dataset)
- [Train](#train)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)
- [Data](#data)

## Getting Started

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch](https://pytorch.org/get-started/locally/)

3. Clone the directory:
```bash
git clone https://github.com/gina7484/Pytorch-UNet.git
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Prepare Dataset
```bash
Pytorch-UNet
|_ data
    |_  imgs: directory with train input images
    |_  masks: directory with train masks (labels)
    |_  aug_imgs: directory with augmented train input images
    |_  aug_masks: directory with augmented train masks (labels)
```

## Train
Move to the branch according to which model you would like to train.

### For Basic U-Net 
This is the master branch. No need to checkout.

command for training:
```bash
python train.py --epochs 30 --batch-size 16 --learning-rate 0.0001 --amp --scale 0.5 --validation 15.0
```

If you want to apply data augmentation, use this command:
```bash
python train_aug.py --epochs 30 --batch-size 32 --learning-rate 0.0001 --amp --scale 0.5 --validation 15.0
```

After training, ```/run-2021XXXX_XXXXXX/``` folder is created under ```./checkpoints/```.
In this folder, weights of each epoch is stored by .pth file format.

### U-Net with Residual layers and Summation-based skip connection

checkout ```fusion``` branch:
```bash
git checkout fusion
```

command for training:
```bash
python train.py --epochs 30 --batch-size 16 --learning-rate 0.0001 --amp --scale 0.5 --validation 15.0
```

If you want to apply data augmentation, use this command:
```bash
python train_aug.py --epochs 30 --batch-size 32 --learning-rate 0.0001 --amp --scale 0.5 --validation 15.0
```

After training, ```/run-2021XXXX_XXXXXX/``` folder is created under ```./checkpoints/```.
In this folder, weights of each epoch is stored by .pth file format.

### Adjusting Training Options

| 기능            | 개발자    | 
| -------------- | :-----: | ------------------------ | :------: | :------: | :------: | :------: | :------: |
 **시작 화면**| 선영 |Splash 화면 띄우기/font 적용         |       |          |*          |          |          |
| **사진 촬영 화면**| 선영 |CameraX를 이용해 카메라 기능 구현         |*         |          |          |          |          |
|                |   선영      |MLkit object detection 반영| *      |         |          |          |          |
|                |      현수   | Detect된 object 정보 이용해서 사진 크기 조정 |     | *        |          |          |          |
|                |      현수   | Detect된 object의 이름으로 사진 저장 |     | *        |          |          |          |
| **단어 학습 화면**| 종선   |글자 쓰기 기능                |*         |          |          |          |          |
|                |         |사진촬영구현되면 사진/단어 정보 가져오기|      |          |*          |          |          |          
|                |         |잘못된 정보 수정 기능|       |          |          |*         |          |
|                |선영      |획수 표시 기능|       |          |          |*         |*         |
| **테스트 화면**   | 종선     |테스트하기 기능 (글자 인식)     |          |           |*         |          |          |    
| **채점 화면**    | 종선      |글자 인식 기능               |*         |          |          |          |          |
|                |         |사진촬영구현되면 사진/단어 정보 가져오기|       |          |*         |          |          |          
|                |         |잘못된 정보 수정 기능|       |          |          |*         |          |
| **단어장 화면**   | 재형     |사진 띄우기 기능              |*         |           |          |          |          |
|                |         | 사진 뒤집기 기능             | *         |          |          |          |          |
|                |         | 사진촬영구현되면 사진/단어 정보 가져오기|      |          | *        |          |          |          
|                |         | 사진 단어 수정 제거|      |          |          |  *       |          |  
| **최적화**       |모두      |디자인, 성능, 정확도, 사용감 개선|          |           |          | *        |*         | 



## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable.


## Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v2.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)
```
The training was done with a 50% scale and bilinear upsampling.

## Data
The Carvana data is available on the [Kaggle website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

You can also download it using the helper script:

```
bash scripts/download_data.sh
```

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). For Carvana, images are RGB and masks are black and white.

You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
