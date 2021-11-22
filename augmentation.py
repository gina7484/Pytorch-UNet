import albumentations as A
import os
import cv2
import imageio
import shutil
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

transform = A.Compose([
    A.ElasticTransform(p=0.5),
    A.GridDistortion(p=0.5),
    A.HorizontalFlip(p=0.5),
    #A.OpticalDistortion(p=1),
])

image_dir = '../2D/training/image/'
mask_dir = '../2D/training/label/'
#image_dir = '../2D/training/test_image/'
#mask_dir = '../2D/training/test_label/'

aug_image_dir = Path('../2D_aug/aug_image/')
aug_mask_dir = Path('../2D_aug/aug_mask/')
#aug_image_dir = Path('../2D_aug/test_aug_image/')
#aug_mask_dir = Path('../2D_aug/test_aug_mask/')

for image_file, mask_file in zip(sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir))):
    print(image_file, mask_file)

    # load image and mask
    image = imageio.imread(image_dir + image_file)
    #print(image.shape)
    mask = imageio.imread(mask_dir + mask_file)
    #print(mask.shape)

    # transform image and mask
    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    print(transformed_image.shape)
    print(transformed_mask.shape)

    # store augmented image and mask
    aug_img = Image.fromarray(transformed_image)
    aug_img.save('aug_' + image_file)
    shutil.move('aug_' + image_file, aug_image_dir)
    aug_mask = Image.fromarray(transformed_mask)
    aug_mask.save('aug_' + mask_file)
    shutil.move('aug_' + mask_file, aug_mask_dir)