import albumentations as A
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

transform = A.Compose([
    A.ElasticTransform(p=0.3),
    A.GridDistortion(p=0.3),
    A.HorizontalFlip(p=0.3),
    #A.OpticalDistortion(p=1),
])

image_dir = '../2D/training/image/'
mask_dir = '../2D/training/label/'
#image_dir = 'CT-ORG/Training_jpg/test5/'
#mask_dir = 'CT-ORG/Training_jpg/mask5/'

transformed_images = []
transformed_masks = []

for image_file, mask_file in zip(os.listdir(image_dir), os.listdir(mask_dir)):
    #print(image_file, mask_file)

    image = cv2.imread(image_dir + image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_dir + mask_file)

    transformed = transform(image=image, mask=mask)
    transformed_image = transformed['image']
    transformed_mask = transformed['mask']
    transformed_images.append(transformed_image)
    transformed_masks.append(transformed_mask)

    '''
    if image is not None:
            plt.figure()
            plt.imshow(image.astype(np.uint8))
            plt.axis('off')
            plt.show()

    if transformed_image is not None:
            plt.figure()
            plt.imshow(transformed_image.astype(np.uint8))
            plt.axis('off')
            plt.show()

    if mask is not None:
            plt.figure()
            plt.imshow(mask.astype(np.uint8))
            plt.axis('off')
            plt.show()

    if transformed_mask is not None:
            plt.figure()
            plt.imshow(transformed_mask.astype(np.uint8))
            plt.axis('off')
            plt.show()
    '''

for t in transformed_images:
    if t is not None:
            plt.figure()
            plt.imshow(t.astype(np.uint8))
            plt.axis('off')
            plt.show()