# U-Net: Semantic segmentation with PyTorch
<a href="#"><img src="https://img.shields.io/github/workflow/status/milesial/PyTorch-UNet/Publish%20Docker%20image?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.9.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)



- [Getting Started](#getting-started)
- [Prepare Dataset](#prepare-dataset)
- [Train](#train)
- [Inference](#inference)

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

| Arguments            | Descriptions    | 
| -------------- | :-----: | 
-h| Show this help message and exit |
| --epochs E, -e E| The number of epochs  |
| --batch-size B, -b B        |Batch size|
|--learning-rate LR, -l LR|Learning rate|
|--load LOAD, -f LOAD|Load model from a .pth file|
|--scale SCALE, -s SCALE|Downscaling factor of the images (default scale is 0.5. If you want better result, use 1. But that uses more memory.)|
|--validation VAL, -v VAL|Percent of the data that is used as validation (0-100)|
|--amp|Use mixed precision (It uses less memory and make GPU speed up.)|

## Inference

Move to the branch according to which model you would like to use.

### For basic U-Net 
This is the master branch. No need to checkout.

command for inference:
```
python predict.py --model ./checkpoints/checkpoint_epoch4.pth --input_dir "../2D/testing/test_lung/" --output_dir "../2D_result/" --scale 0.5
```

After inference, ```/output-2021XXXX_XXXXXX//``` folder is created in selected path.
In this folder, the results of inference is stored.

### U-Net with Residual layers and Summation-based skip connection

move to ```fusion``` branch:
```
git checkout fusion
```

command for inference:
```
python predict.py --model ./checkpoints/checkpoint_epoch4.pth --input_dir "../2D/testing/test_lung/" --output_dir "../2D_result/" --scale 0.5
```

After inference, ```/output-2021XXXX_XXXXXX//``` folder is created in selected path.
In this folder, the results of inference is stored.

### Adjusting Inference Options
| Arguments            | Descriptions    | 
| -------------- | :-----: | 
-h, --help| Show this help message and exit |
|--model FILE, -m FILE|Specify the file in which the model is stored|
|--input_dir PATH, -i PATH|[Required] Path to directory with input images|
|--output_dir PATH, -o PATH|[Required] Path to directory to store output images|
|--viz, -v|Visualize the images as they are processed|
|--no-save, -n|Do not save the output masks|
|--mask_threshold THRESHOLD, -t THRESHOLD|Minimum probability value to consider a mask pixel white|
|--scale SCALE, -s SCALE|Scale factor for the input images|

---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
