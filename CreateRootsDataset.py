import cv2
import numpy as np
from glob import glob
import os


'''
Lavora su un dataset la cui alberatura è
\Roots20210208
    \compositeImage --> Immagini a colori
    \GroundTruth --> Immagini B/N con radici evidenziate

i nomi dei file hanno la medesima radice e differiscono per la parte finale
[ID]_compositeImage.png
[ID]_groundTruth.png
'''

def buildCompositeFn(fn, DATASET_PATH):
    tmp = fn.split(os.path.sep)[-1]
    tmp = tmp.replace('groundTruth', 'compositeImage')
    return os.path.join(DATASET_PATH, 'compositeImage', tmp)

def buildOutFn(fn, classname, id, OUTPUT_PATH):
    tmp = fn.split(os.path.sep)[-1].replace('groundTruth', id)
    return os.path.join(OUTPUT_PATH, classname, tmp)

DATASET_PATH = 'C:\\Users\\vitor\\Desktop\\Roots20210208_Subset'

OUTPUT_PATH = 'C:\\Users\\vitor\\Desktop\\output'
HALF_WINDOW_SIZE = 32
PERC_ZEROS_THR = 0.75
N_pixels = (2*HALF_WINDOW_SIZE+1) ** 2

fns = glob(os.path.join(DATASET_PATH, 'GroundTruth', '*.png'))

cnt = 1
for fn in fns:
    print(f'{cnt}/{len(fns)} {fn}')
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    compositeImg = cv2.imread(buildCompositeFn(fn, DATASET_PATH))
    height, width = img.shape

    N_root_patches = 0
    coords = np.argwhere(img != 0)
    for r, c in coords:
        if r > HALF_WINDOW_SIZE and r < (height - HALF_WINDOW_SIZE) and c > HALF_WINDOW_SIZE and c < (width - HALF_WINDOW_SIZE):
            patch = compositeImg[r-HALF_WINDOW_SIZE:r+HALF_WINDOW_SIZE+1, c-HALF_WINDOW_SIZE:c+HALF_WINDOW_SIZE+1, :]
            cv2.imwrite(buildOutFn(fn, 'root', f'{r}-{c}', OUTPUT_PATH), patch)
            N_root_patches += 1
    
    N_other_patches = 0
    coords = np.argwhere(img == 0)
    for r, c in coords:
        if r > HALF_WINDOW_SIZE and r < (height - HALF_WINDOW_SIZE) and c > HALF_WINDOW_SIZE and c < (width - HALF_WINDOW_SIZE):
            GT_patch = img[r-HALF_WINDOW_SIZE:r+HALF_WINDOW_SIZE+1, c-HALF_WINDOW_SIZE:c+HALF_WINDOW_SIZE+1]
            perc_zeros = 1 - (np.count_nonzero(GT_patch) / N_pixels)
            if perc_zeros > PERC_ZEROS_THR:
                patch = compositeImg[r-HALF_WINDOW_SIZE:r+HALF_WINDOW_SIZE+1, c-HALF_WINDOW_SIZE:c+HALF_WINDOW_SIZE+1, :]
                cv2.imwrite(buildOutFn(fn, 'other', f'{r}-{c}', OUTPUT_PATH), patch)
                N_other_patches += 1
                if N_other_patches == N_root_patches:
                    break
    cnt += 1
