import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
import json

import shutil
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

inputdir = "../2D/testing/image/"
outputdir = "../2D_result/"

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    #parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    #parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    parameter_dict={}
    args = get_args()
    #in_files = args.input
    #out_files = get_output_filenames(args)

    net = UNet(n_channels=1, n_classes=2)
    parameter_dict['model']='UNet'
    # net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True)

    parameter_dict['scale']=args.scale
    parameter_dict['mask_threshold']=args.mask-threshold

    device = torch.device('cpu')
    print("Loading model and using device")
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    parameter_dict['checkpoint_path']=args.model

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    now=datetime.now()
    now_string = now.strftime("output-%Y%m%d_%H%M%S/")
    dir_output = Path(outputdir+now_string)
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    parameter_dict['date']=now.strftime("%Y%m%d_%H%M%S/")
    parameter_dict['output_dir']=outputdir+now_string

    # save parameters as json file
    with open(outputdir+now_string+"log.json","w") as fp:
        json.dump(parameter_dict,fp)

    for filename in os.listdir(inputdir):
        if not filename.endswith('.jpg'):
            continue
        
        f = os.path.join(inputdir,filename)
        logging.info(f'\nPredicting image {filename} ...')
        img = Image.open(f)

        mask = predict_img(net=net,
                        full_img=img,
                        scale_factor=args.scale,
                        out_threshold=args.mask_threshold,
                        device=device)

        if not args.no_save:
            out_filename = 'out' + filename[3:]
            result = mask_to_image(mask)
            result.save(out_filename)
            shutil.move(out_filename, dir_output)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
