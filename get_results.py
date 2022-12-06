from pathlib import Path
import os
from typing import Union

from src.exceptions.exceptions import PyRootsError
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import cv2
import numpy as np
import pandas as pd
import pygame


def play_sound(soundpath):
    pygame.mixer.init()
    pygame.mixer.music.load(soundpath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue


def get_metrics(
    gt_path: Union[str, Path],
    pred_path: Union[str, Path]) -> dict:
    """ Get metrics from comparing two images.

    This function returns true positives, true negatives,
    false positives, false negatives, as well as precision,
    recall, accuracy and F1-Score by comparing the ground truth
    with the predicted mask. 

    Args:
        gt_path: path of the ground truth.
        pred_path: path of the predicted mask.
    Returns:
        A dictionary containing required metrics.
    """
    if isinstance(gt_path, Path):
        gt_path = str(gt_path) 
    if isinstance(pred_path, Path):
        pred_path = str(pred_path)
    or_img = cv2.imread(gt_path)
    or_img = or_img[:, :, 0]
    _, or_img = cv2.threshold(or_img, 127, 255, cv2.THRESH_BINARY)
    or_img[or_img == 255] = 1
    pred_img = cv2.imread(pred_path)
    pred_img = pred_img[:, :, 0]
    _, pred_img = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)
    pred_img[pred_img == 255] = 1

    fn = np.count_nonzero(np.logical_and(or_img == 1, pred_img == 0))
    fp = np.count_nonzero(np.logical_and(or_img == 0, pred_img == 1))
    tn = np.count_nonzero(np.logical_and(or_img == 0, pred_img == 0))
    tp = np.count_nonzero(np.logical_and(or_img == 1, pred_img == 1))

    p = tp / (tp + fp)
    r = tp / (tp + fn)
    a = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * (p * r) / p + r

    return {
        'img_name': pred_path.split('/')[-1],
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'p': p,
        'r': r,
        'a': a,
        'f1': f1
    }

def to_csv(
    gt: list[pd.DataFrame],
    preds: list[pd.DataFrame],
    fpath: str,
    idx_col: str = 'img_name',
    return_df: bool = True,
    funny: bool = False):
    """ Utility function to process and save results.

    This function process and saves the provided lists
    of dataframes, optionally returning results.

    Args:
        gt: list of ground truth dataframes.
        preds: list of prediction dataframes.
        fpath: path of the output file.
        idx_col: index column.
        return_df: states whether the dataframe should be returned.
        funny: enables fun mode.
    """
    try:
        df = pd.DataFrame([get_metrics(str(gt[i]), str(preds[i])) for i in range(len(gt))])
        df.set_index(idx_col, inplace=True)
        df.to_csv(fpath)
    except Exception as e:
        print(e)
        if funny:
            play_sound('../sounds/smb_gameover.wav')
    return df if return_df else None


ground_truths = [img for img in Path('data', 'ground_truth').glob('*.png')]
preds_segroot_or = [img for img in Path('data', 'preds_or_weights').glob('*.jpg')]
preds_segroot_retrain = [img for img in Path('data', 'preds_segroot_retrain').glob('*.jpg')]
preds_pyroots_257 = [img for img in Path('data', 'preds_pyroots').glob('*_257_mask_0.75.png')]
preds_pyroots_129 = [img for img in Path('data', 'preds_pyroots').glob('*_129_mask_0.75.png')]
preds_pyroots_65 = [img for img in Path('data', 'preds_pyroots').glob('*_65_mask_0.75.png')]
# preds_pyroots_65 = [img for img in Path('..', 'data', 'preds_pyroots').glob('*_64_mask_0.75.png')]

# check lists
if any(len(lst) != len(ground_truths) for lst in [preds_pyroots_257, preds_pyroots_129, preds_pyroots_65, preds_segroot_retrain, preds_segroot_or]):
    print('Error!')
    # play_sound('../sounds/smb_gameover.wav')
    raise PyRootsError
    # raise ValueError('Check images list lengths')
    

# TODO: if save
df_pso = to_csv(ground_truths, preds_segroot_or, 'res_pso.csv')
df_psr = to_csv(ground_truths, preds_segroot_retrain, 'res_psr.csv')
df_pp_257 = to_csv(ground_truths, preds_pyroots_257, 'res_pp_257.csv')
df_pp_129 = to_csv(ground_truths, preds_pyroots_129, 'res_pp_129.csv')
df_pp_65 = to_csv(ground_truths, preds_pyroots_65, 'res_pp_65.csv')

with open('res.txt', 'w') as f:
    f.write('Metrics for SegRoot - original weights\n')
    f.write(f"Precision: {round(df_pso['p'].mean() * 100, 2)}\t\tRecall: {round(df_pso['r'].mean() * 100, 2)}\t\tAccuracy: {round(df_pso['a'].mean() * 100, 2)}\t\tF1 Score: {round(df_pso['f1'].mean() * 100, 2)}\n")
    f.write('Metrics for SegRoot - retrained\n')
    f.write(f"Precision: {round(df_psr['p'].mean() * 100, 2)}\t\tRecall: {round(df_psr['r'].mean() * 100, 2)}\t\tAccuracy: {round(df_psr['a'].mean() * 100, 2)}\t\tF1 Score: {round(df_psr['f1'].mean() * 100, 2)}\n")
    f.write('Metrics for PyRoots 257\n')
    f.write(f"Precision: {round(df_pp_257['p'].mean() * 100, 2)}\t\tRecall: {round(df_pp_257['r'].mean() * 100, 2)}\t\tAccuracy: {round(df_pp_257['a'].mean() * 100, 2)}\t\tF1 Score: {round(df_pp_257['f1'].mean() * 100, 2)}\n")
    f.write('Metrics for PyRoots 129\n')
    f.write(f"Precision: {round(df_pp_129['p'].mean() * 100, 2)}\t\tRecall: {round(df_pp_129['r'].mean() * 100, 2)}\t\tAccuracy: {round(df_pp_129['a'].mean() * 100, 2)}\t\tF1 Score: {round(df_pp_129['f1'].mean() * 100, 2)}\n")
    f.write('Metrics for PyRoots 65\n')
    f.write(f"Precision: {round(df_pp_65['p'].mean() * 100, 2)}\t\tRecall: {round(df_pp_65['r'].mean() * 100, 2)}\t\tAccuracy: {round(df_pp_65['a'].mean() * 100, 2)}\t\tF1 Score: {round(df_pp_65['f1'].mean() * 100, 2)}\n")


print('complete!')
play_sound('sounds/smb_stage_clear.wav')
