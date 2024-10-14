import torch
import numpy as np
import torchvision.transforms as T
import cv2
from src.NetworkModels import Net, BinaryNet
from pathlib import Path
from pdb import set_trace
import tqdm
from matplotlib import pyplot as plt


transform = T.Compose([T.ToTensor(),
                        T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))])

T_norm = T.Normalize((0.5, 0.5, 0.5), (0.1, 0.1, 0.1))

# img_name = '16'
image_paths = [img for img in Path('data', 'inference', 'patches', 'bayolo').rglob('*.jpg')]

# img_fn = str(Path('data', 'inference', 'patches', 'bayolo', f'{img_name}.jpg'))
for image_path in image_paths:
    img_fn = str(image_path)
    print(f'Processing image {img_fn}...')
    model_fn = Path('cracks.pth')
    HALF_WINDOW_SIZE = 32
    batch_size = 128

    img = cv2.imread(img_fn)
    height, width, _ = img.shape
    mask = np.zeros((height, width), dtype=np.float32)
    device = torch.device("cuda")
    model = BinaryNet()
    model.load_state_dict(torch.load(model_fn))
    model.to(device)
    '''
    for r in range(HALF_WINDOW_SIZE, height-HALF_WINDOW_SIZE):
        for c in range(HALF_WINDOW_SIZE, width-HALF_WINDOW_SIZE):
            patch = img[r-HALF_WINDOW_SIZE:r+HALF_WINDOW_SIZE+1, c-HALF_WINDOW_SIZE:c+HALF_WINDOW_SIZE+1, :]
            tensor_patch = transform(patch)
            tensor_batch = torch.stack([tensor_patch for _ in range(batch_size)])
            input = tensor_batch.to(device)
            output = model(input)
            if torch.argmax(output):
                mask[r,c] = 255
    '''
    # cv2.namedWindow("Image", cv2.WINDOW_FREERATIO)
    # #cv2.namedWindow("Mask", cv2.WINDOW_FREERATIO)
    # cv2.imshow("Image", img)
    # #cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    # set_trace()
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
    bViewProgress = False
    # set_trace()
    with torch.no_grad():
        # set_trace()
        for i in tqdm.trange(N): #range(int(N/2), N):
            # set_trace()
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
            mask_show = mask.cpu().numpy()
            # set_trace()
            img[mask_show < 0.45, :] = [255, 255, 255]
            img[mask_show > 0.45, :] = [0, 0, 0]

                # set_trace()
            # if bViewProgress:
            #     mask_show = mask.cpu().numpy()
            #     #mask_show[mask_show < root_percentage] = 0
            #     #mask_show[mask_show >= root_percentage] = 255
            #     img[mask_show > 0.99, :] = [0, 204, 0] # Verde, radice > 99%
            #     img[np.logical_and(mask_show < 0.99, mask_show > 0.9), :] = [0, 102, 0] # Verde scuro, radice oltre 90% ma sotto 99%
            #     img[np.logical_and(mask_show < 0.9, mask_show > 0.8), :] = [0, 255, 255] # Giallo, radice tra 90% e 90%
            #     img[np.logical_and(mask_show < 0.8, mask_show > 0.75), :] = [0, 128, 255] # Arancio
            #     img[np.logical_and(mask_show < 0.3, mask_show > 0), 0] = 128 # Radice entro il 30%
            #     # cv2.imshow("Image", img)
            #     # key_pressed = cv2.waitKey(1)

    outpath = Path('results', f'{image_path.parent}')
    outpath.mkdir(exist_ok=True, parents=True)
    img_name = image_path.name.split('.')[0]
    out_file = Path(outpath,  f'{img_name}.png')
    cv2.imwrite(str(out_file), img)
