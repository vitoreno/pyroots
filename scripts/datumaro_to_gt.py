from concurrent.futures import process
import json
import cv2
import os
import copy


BASE_PATH = '\\\\srv03.ba.stiima.cnr.it\\data01\\datasets\\Roots\\barley'
LINE_THICKNESS = 3

with open('20220923_barley_roots.json', 'r') as read_file:
    data = json.load(read_file)


cv2.namedWindow('Image', cv2.WINDOW_FREERATIO)
cv2.namedWindow('GT Mask', cv2.WINDOW_FREERATIO)

#item = data['items'][-1]
processed_img = 0
for item in data['items']:
    if len(item['annotations']) == 0:
        continue
    
    fn = os.path.join(BASE_PATH, item['id']) + '.png'
    img = cv2.imread(fn)

    #cv2.imshow("Image", img)
    #key = cv2.waitKey()

    gt_mask = copy.deepcopy(img[:,:,0])
    gt_mask *= 0
    
    for ann in item['annotations']:
        if ann['type'] == 'polyline':
            p_list = ann['points']
            pts = list(zip(p_list[0::2], p_list[1::2]))
            for idx in range(1, len(pts)):
                x0, y0 = int(pts[idx-1][0]), int(pts[idx-1][1])
                x1, y1 = int(pts[idx][0]), int(pts[idx][1])
                cv2.line(gt_mask, (x0, y0), (x1, y1), 255, LINE_THICKNESS)
                #cv2.line(img, (x0, y0), (x1, y1), (0,0,255), LINE_THICKNESS)
            
            #for x, y in pts:
            #    cv2.drawMarker(img, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 5, 2)

    #cv2.imshow("Image", img)
    #cv2.imshow("GT Mask", gt_mask)
    #key = cv2.waitKey()

    cv2.imwrite(f"{os.path.join(BASE_PATH, 'Roots20220923_Subset', *item['id'].split('/')[1:])}.png", img)
    cv2.imwrite(f"{os.path.join(BASE_PATH, 'Roots20220923_Subset', 'GroundTruth', item['id'].split('/')[-1].replace('compositeImage', 'GroundTruth'))}.png", gt_mask)
    processed_img += 1
    if processed_img % 100 == 99:
        print(f'Processed {processed_img} imgs')

cv2.destroyAllWindows()