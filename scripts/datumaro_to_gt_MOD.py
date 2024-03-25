##DATUMARO TO GT: Process JSON file

from concurrent.futures import process
import json
import cv2
import os
import copy
import argparse
from pathlib import Path


#BASE_PATH = '\\\\srv03.ba.stiima.cnr.it\\data01\\datasets\\Roots\\barley'
#LINE_THICKNESS = 3


# def Datumato_to_gt(
#         file_name: Path,
#         BASE_PATH: Path,
#         LINE_THICKNESS: int):
    


def main(args):

    with open(args.file_name, 'r') as read_file:
        data = json.load(read_file)
    
    cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
    cv2.namedWindow('GT Mask', cv2.WINDOW_FREERATIO)
    
   
    processed_img = 0
    for item in data['items']:
        if len(item['annotations']) == 0:
            continue
    
    fn = os.path.join(args.base_path, item['id']) + '.png'
    img = cv2.imread(fn)

   
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
       
    cv2.imwrite(f"{os.path.join(args.base_path, args.subset_path, *item['id'].split('/')[1:])}.png", img)
    cv2.imwrite(f"{os.path.join(args.base_path, args.subset_path, 'GroundTruth', item['id'].split('/')[-1].replace('compositeImage', 'GroundTruth'))}.png", gt_mask)
    processed_img += 1
    if processed_img % 100 == 99:
        print(f'Processed {processed_img} imgs')
        
    cv2.destroyAllWindows()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON file")
    parser.add_argument(
        "--base_path", type= str, default='.', help="Directory cartella contenente il database", required=True
    )
    parser.add_argument(
        "--subset_path", type= str, default='.', help="Directory e nome cartella che conterrà il nuovo database creato", required=True
    )
    parser.add_argument(
        "--file_name", type= str, help="Directory e nome del file JSON", required=True
    )
    parser.add_argument(
        "--line_thickness", type=int, default=3, help="Valore dello spessore della linea GT", required=True
    )

    args = parser.parse_args()
    main(args)


