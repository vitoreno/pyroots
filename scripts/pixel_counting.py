from skimage.io import imread
from pathlib import Path
from pdb import set_trace
import matplotlib.pyplot as plt
import numpy as np


i = imread(Path('results', 'data', 'inference', 'patches', 'bayolo', '16.png'))

o = imread(Path('data', 'inference', 'patches', 'bayolo', '16.jpg'))

img = i[32:-32, 32:-32, :]

ratio = np.count_nonzero(img) / ((img.shape[0] + 64) * (img.shape[1] + 64) )
