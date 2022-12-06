import argparse
import torch
import numpy as np
import torchvision.transforms as T
import cv2
from NetworkModels import Net, BinaryNet
from matplotlib import pyplot as plt
import os
from progressbar import ProgressBar


def view_mask(mask_fn):
    img_mask = cv2.imread(mask_fn, cv2.IMREAD_ANYDEPTH)
    import copy
    tmp = copy.deepcopy(img_mask)

    root_percentage = 0.99
    tmp[tmp<root_percentage] = 0
    m = np.min(tmp[tmp!=0])
    M = np.max(tmp)
    im = ((tmp - m) / (M-m))
    im[img_mask < root_percentage] = 0
    plt.imshow(im)
    plt.show()


def binarize_mask(mask_gpu, thr):
    binary_mask = mask_gpu.cpu().numpy()
    binary_mask[binary_mask < thr] = 0
    binary_mask[binary_mask >= thr] = 255
    return binary_mask


def main(args):
    transform = T.Compose([T.ToTensor(),
                            T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))])

    T_norm = T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))

    img_fn = args.img_fn #'C:\\Users\\vitor\\Desktop\\Roots20210208_Subset\\compositeImage\\CIC09C15A001_2019-10-15_cV_iS_compositeImage.png'
    model_fn = args.model_fn #'.\\models\\20220923_171621.pth'
    HALF_WINDOW_SIZE = int((args.img_width_height-1)/2)
    batch_size = args.batch_size #128

    img = cv2.imread(img_fn)
    height, width, _ = img.shape
    mask = np.zeros((height, width), dtype=np.float32)
    device = torch.device("cuda")
    model = BinaryNet(args.img_width_height)
    model.load_state_dict(torch.load(model_fn))
    model.to(device)

    if args.bView:
        cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)
        #cv2.namedWindow("Mask", cv2.WINDOW_FREERATIO)
        cv2.imshow("Image", img)
        #cv2.imshow("Mask", mask)
        cv2.waitKey(0)

    height_range = torch.tensor(range(HALF_WINDOW_SIZE, height-HALF_WINDOW_SIZE))
    width_range = torch.tensor(range(HALF_WINDOW_SIZE, width-HALF_WINDOW_SIZE))
    coords = torch.cartesian_prod(height_range, width_range)

    GPUcoords = coords.to(device)
    imgT = T.ToTensor()(img)
    imgT = imgT.to(device)

    mask = torch.zeros((height, width))
    mask = mask.to(device)

    N = int(len(GPUcoords)/batch_size)
    root_percentage = 0.99
    print(f'Processing file {args.img_fn.split(os.path.sep)[-1]}')
    with torch.no_grad():
        pbar = ProgressBar()
        for i in pbar(range(N)): #range(int(N/2), N):
            idx = GPUcoords[i*batch_size:i*batch_size+batch_size]
            tensor_list = []
            for r, c in idx:
                patch = imgT[:, r-HALF_WINDOW_SIZE:r+HALF_WINDOW_SIZE+1, c-HALF_WINDOW_SIZE:c+HALF_WINDOW_SIZE+1]
                tensor_list.append(T_norm(patch))
            tensor_batch = torch.stack(tensor_list)
            output = model(tensor_batch)
            predicted = torch.sigmoid(output)
            for j, t in enumerate(idx):
                mask[t[0], t[1]] = predicted[j]
            if args.bView:
                mask_show = mask.cpu().numpy()
                #mask_show[mask_show < root_percentage] = 0
                #mask_show[mask_show >= root_percentage] = 255
                img[mask_show > 0.99, :] = [0, 204, 0] # Verde, radice > 99%
                img[np.logical_and(mask_show < 0.99, mask_show > 0.9), :] = [0, 102, 0] # Verde scuro, radice oltre 90% ma sotto 99%
                img[np.logical_and(mask_show < 0.9, mask_show > 0.8), :] = [0, 255, 255] # Giallo, radice tra 90% e 90%
                img[np.logical_and(mask_show < 0.8, mask_show > 0.75), :] = [0, 128, 255] # Arancio
                img[np.logical_and(mask_show < 0.3, mask_show > 0), 0] = 128 # Radice entro il 30%
                cv2.imshow("Image", img)
                key_pressed = cv2.waitKey(1)
    
    img_id = img_fn.split(os.path.sep)[-1].split('.')[0]
    mod_id = model_fn.split(os.path.sep)[-1].split('.')[0]
    cv2.imwrite(f'{img_id}_{mod_id}_{args.img_width_height}_inference.tif', mask.cpu().numpy())
    binary_mask = binarize_mask(mask, args.thr)
    cv2.imwrite(f'{img_id}_{mod_id}_{args.img_width_height}_mask.png', binary_mask)
    
    if args.bView:        
        plt.imshow(mask.cpu().numpy())
        plt.show()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RootInference')
    parser.add_argument('--view', dest='bView', type=bool, action=argparse.BooleanOptionalAction, default=True, help='View image during inference')
    parser.add_argument('--img_fn', dest='img_fn', default='', help='Image filename')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_fn', dest='model_fn', default='model.pth', help='Model filename')
    parser.add_argument('--img_width_height', dest='img_width_height', type=int, default=65, help='Size of width/height of the squared patch')
    parser.add_argument('--thr', dest='thr', type=float, default=0.5, help='Threshold to produce the binary mask')

    args = parser.parse_args()
    main(args)