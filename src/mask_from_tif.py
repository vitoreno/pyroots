import argparse
import numpy as np
import cv2
import os
from progressbar import ProgressBar
from glob import glob
from mask_defs import *
import matplotlib.pyplot as plt
from pdb import set_trace as _breakpoint



def process(input_tif, thr, qualitative):
    img_inference = cv2.imread(input_tif, cv2.IMREAD_ANYDEPTH)
    if qualitative:
        mask = mask_inference(img_inference)
        out_fn = input_tif.split(os.path.sep)
        out_fn[-2] = 'qualitative_mask'
        out_fn[-1] = f"{out_fn[-1].replace('inference', f'mask').replace('tif','png')}"
        save_qualitative_image(mask, os.path.join(*out_fn))
        
    else:
        mask = binarize(img_inference, thr)
        out_fn = f"{input_tif.split(os.path.sep)[-1].replace('inference', f'mask_{thr}').replace('tif','png')}"
        cv2.imwrite(out_fn, mask)


def mask_inference(img_inference):
    mask = np.dstack([img_inference*0, img_inference*0, img_inference*0])
    mask = np.uint8(mask)
    mask[img_inference <= 0.3,:] = colors["black"]
    mask[np.logical_and(img_inference > 0.3, img_inference < 0.6), :] = colors["yellow"]
    mask[img_inference >= 0.6, :] = colors["orange"]
    mask[img_inference >= 0.8, :] = colors["green"]
    mask[img_inference >= 0.95, :] = colors["super_green"]
    return mask


def save_qualitative_image(mask, name):    
    plt.figure(figsize=(8,12))
    plt.imshow(mask)
    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=[black_patch, yellow_patch, orange_patch, green_patch, super_green_patch], bbox_to_anchor=(1, 1))
    plt.savefig(name, format="png", bbox_inches='tight')


def binarize(img_inference, thr):
    binary_mask = img_inference * 0
    binary_mask[img_inference < thr] = 0
    binary_mask[img_inference >= thr] = 255
    return binary_mask


def main(args):
    if(args.input_dir is None):
        print(f'Processing image {args.input_tif}')
        process(args.input_tif, args.thr)
    else:
        print(f'Processing directory {args.input_dir}')
        files = glob(os.path.join(args.input_dir, f'*{args.filter}*.tif'))
        pbar = ProgressBar()
        for fn in pbar(files):
            process(fn, args.thr, args.qualitative)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_tif")
    parser.add_argument("-d", "--input_dir")
    parser.add_argument("-f", "--filter", default='')
    parser.add_argument('-q', "--qualitative", type=bool, action=argparse.BooleanOptionalAction, default=True, help='View image during inference')
    parser.add_argument("-t", "--thr", type=float)
    
    args = parser.parse_args()
    main(args)
