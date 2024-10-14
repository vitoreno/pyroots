import os
from glob import glob

import cv2
import numpy as np
from pathlib import Path
from pdb import set_trace
from typing import Any

"""
Lavora su un dataset la cui alberatura è
\Roots20210208
    \compositeImage --> Immagini a colori
    \GroundTruth --> Immagini B/N con radici evidenziate

i nomi dei file hanno la medesima radice e differiscono per la parte finale
GRANDE CAZZATA
[ID]_compositeImage.png
[ID]_groundTruth.png
"""

def append_suffix(fn: Path, sfx: str = '_gt') -> Path:
    """ Appends a suffix to a file path.

    Args:
        fn: the path of the file to modify.
        sfx: The suffix.
        
    """
    tmp = str(fn.name).replace(sfx, '')
    return Path(fn.parent, tmp)


def buildCompositeFn(fn, DATASET_PATH):
    try:
        fn = str(fn)
        tmp = fn.split(os.path.sep)[-1]
        tmp = tmp.replace("_gt", "")
        return os.path.join(DATASET_PATH, "", tmp)
    except Exception as e:
        print(e)


def buildOutFn(fn, classname, id, OUTPUT_PATH):
    try:
        fn = str(fn)
        tmp = fn.split(os.path.sep)[-1].replace("_gt", id)
        return os.path.join(OUTPUT_PATH, classname, tmp)
    except Exception as e:
        print(e)


DATASET_PATH = Path('data', 'raw', 'cracks', 'images', 'default', 'db_cracks')

OUTPUT_PATH = Path('data', 'processed', 'cracks_257')
HALF_WINDOW_SIZE = 128
PERC_ZEROS_THR = 0.75  # percentuale di sfondo in una patch di tipo other
N_MAX_PATCH_PER_IMG = 1024
N_pixels = (2 * HALF_WINDOW_SIZE + 1) ** 2

fns = [image for image in DATASET_PATH.rglob('*_gt.JPG')]
# set_trace()

cnt = 1
for fn in fns:
    print(f"{cnt}/{len(fns)} {fn}")
    # set_trace()
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    # compositeImg = cv2.imread(buildCompositeFn(str(fn), DATASET_PATH))
    compositeImg = cv2.imread(str(fn))
    height, width = img.shape

    N_root_patches = 0
    coords = np.argwhere(img != 0)
    np.random.shuffle(coords)  # hey, hey, hey with the barley shuffle
    for r, c in coords:
        if (
            r > HALF_WINDOW_SIZE
            and r < (height - HALF_WINDOW_SIZE)
            and c > HALF_WINDOW_SIZE
            and c < (width - HALF_WINDOW_SIZE)
        ):
            try:
                patch = compositeImg[
                    r - HALF_WINDOW_SIZE : r + HALF_WINDOW_SIZE + 1,
                    c - HALF_WINDOW_SIZE : c + HALF_WINDOW_SIZE + 1,
                    :,
                ]
                cv2.imwrite(buildOutFn(str(fn), "crack", f"{r}-{c}", OUTPUT_PATH), patch)
            except Exception as e:
                break
            N_root_patches += 1
            if N_root_patches == N_MAX_PATCH_PER_IMG:
                break

    N_other_patches = 0
    coords = np.argwhere(img == 0)
    np.random.shuffle(
        coords
    )  # per evitare di prendere le patch sempre e solo a partire dall'alto
    for r, c in coords:
        if (
            r > HALF_WINDOW_SIZE
            and r < (height - HALF_WINDOW_SIZE)
            and c > HALF_WINDOW_SIZE
            and c < (width - HALF_WINDOW_SIZE)
        ):
            GT_patch = img[
                r - HALF_WINDOW_SIZE : r + HALF_WINDOW_SIZE + 1,
                c - HALF_WINDOW_SIZE : c + HALF_WINDOW_SIZE + 1,
            ]
            perc_zeros = 1 - (np.count_nonzero(GT_patch) / N_pixels)
            if perc_zeros > PERC_ZEROS_THR:
                try:
                    patch = compositeImg[
                        r - HALF_WINDOW_SIZE : r + HALF_WINDOW_SIZE + 1,
                        c - HALF_WINDOW_SIZE : c + HALF_WINDOW_SIZE + 1,
                        :,
                    ]
                    cv2.imwrite(buildOutFn(fn, "other", f"{r}-{c}", OUTPUT_PATH), patch)
                except Exception as e:
                    break
                N_other_patches += 1
                if N_other_patches == N_MAX_PATCH_PER_IMG:
                    break
    cnt += 1
