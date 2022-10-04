import cv2
import numpy as np
import argparse


def get_metrics(args):
    or_img = cv2.imread(args.or_img_path)
    or_img = or_img[:, :, 0]
    _, or_img = cv2.threshold(or_img, 127, 255, cv2.THRESH_BINARY)
    or_img[or_img == 255] = 1
    pred_img = cv2.imread(args.pred_img_path)
    pred_img = pred_img[:, :, 0]
    _, pred_img = cv2.threshold(pred_img, 127, 255, cv2.THRESH_BINARY)
    pred_img[pred_img == 255] = 1

    fn = np.count_nonzero(np.logical_and(or_img == 1, pred_img == 0))
    fp = np.count_nonzero(np.logical_and(or_img == 0, pred_img == 1))
    tn = np.count_nonzero(np.logical_and(or_img == 0, pred_img == 0))
    tp = np.count_nonzero(np.logical_and(or_img == 1, pred_img == 1))

    print(f'true positive {tp}')
    print(f'false positive {fp}')
    print(f'false negative {fn}')
    print(f'true negative {tn}')
    p = tp / (tp + fp)
    print(f'precision {round(p*100, 2)}')
    r = tp / (tp + fn)
    print(f'recall {round(r*100, 2)}')
    a = (tp + tn) / (tp + tn + fp + fn)
    print(f'accuracy {round(a*100, 2)}')
    f1 = 2 * (p * r) / (p + r)
    print(f'f1 score {round(f1*100, 2)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--or_img_path')
    parser.add_argument('-p', '--pred_img_path')
    args = parser.parse_args()
    get_metrics(args)
