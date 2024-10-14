# python datumaro_to_gt.py -f .\data\raw\cracks\annotations\default.json -p .\data\raw\cracks\images\default\

import argparse
import json
import cv2
from pathlib import Path
import copy
from pdb import set_trace


def main(args):
    with open(args.filename, 'r') as f:
        data = json.load(f)

    cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('GT Mask', cv2.WINDOW_FREERATIO)

    processed_img = 0
    # set_trace()
    for item in data['items']:
        if len(item['annotations']) == 0:
            continue
        fn = Path(args.base_path, f"{item['id']}.{args.extension}")
        img = cv2.imread(str(fn))
        try:
            gt_mask = copy.deepcopy(img[:,:,0])
            gt_mask *= 0
            for ann in item['annotations']:
                if ann['type'] == 'polyline':
                    p_list = ann['points']
                    pts = list(zip(p_list[0::2], p_list[1::2]))
                    for idx in range(1, len(pts)):
                        x0, y0 = int(pts[idx-1][0]), int(pts[idx-1][1])
                        x1, y1 = int(pts[idx][0]), int(pts[idx][1])
                        cv2.line(gt_mask, (x0, y0), (x1, y1), 255, args.line_thickness)
                        f"{item['id'].split('/')[1:]}.{args.extension}".replace('compositeImage', 'GroundTruth')
            out_path = Path(fn.parent, f"{fn.name.split('.')[0]}_gt.{fn.name.split('.')[-1]}")
            cv2.imwrite(str(out_path), gt_mask)
        except Exception as e:
            print(e)    
        processed_img += 1

        if processed_img % 100 == 99:
            print(f'Processed {processed_img} imgs')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--filename',
        help='Name of the file where the annotations are stored',
        required=True)
    parser.add_argument(
        '-l',
        '--line_thickness',
        help='Thickness of the line',
        type=int,
        default=3)
    parser.add_argument(
        '-p',
        '--base_path',
        help='Path where results should be stored',
        default='')
    parser.add_argument(
        '-e',
        '--extension',
        default='JPG'
    )
    args = parser.parse_args()
    main(args)
