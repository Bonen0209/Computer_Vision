import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image
from scipy.ndimage import median_filter
from tqdm import tqdm

DEBUG = True

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape

    Il = cv2.bilateralFilter(Il, 3, 21, 21)
    Ir = cv2.bilateralFilter(Ir, 3, 21, 21)
    Il = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY).astype(np.float32)
    Ir = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY).astype(np.float32)
    labels = np.zeros((h, w), dtype=np.float32)
    
    border_size = 4
    Il = cv2.copyMakeBorder(Il,
                            border_size,
                            border_size,
                            border_size,
                            border_size,
                            borderType=cv2.BORDER_REFLECT).astype(np.float32)

    Ir = cv2.copyMakeBorder(Ir,
                            border_size,
                            border_size,
                            border_size+max_disp,
                            border_size,
                            borderType=cv2.BORDER_REFLECT).astype(np.float32)

    abs_cost = np.ones((h, w, max_disp), dtype=np.float32) * np.inf
    census_cost = np.ones((h, w, max_disp), dtype=np.float32) * np.inf

    # >>> Cost computation
    p_w = w + 2 * border_size
    for d in tqdm(range(max_disp)):
        Id = Il - Ir[:, max_disp-d:max_disp-d+p_w]
        Id = np.absolute(Id)

        abs_cost[:, :, d] = Id[border_size:-border_size, border_size:-border_size]

        census_x = 4
        census_y = 3
        for y in range(border_size, border_size + h):
            for x in range(border_size, border_size + w):
                tmp_l = Il[y-census_y:y+census_y+1, x-census_x:x+census_x+1] < Il[y, x]
                tmp_r = Ir[:, max_disp-d:max_disp-d+p_w][y-census_y:y+census_y+1, x-census_x:x+census_x+1] \
                        < Ir[:, max_disp-d:max_disp-d+p_w][y, x]

                hamming_cost = (tmp_l != tmp_r).sum()
                census_cost[y-border_size, x-border_size, d] = hamming_cost

    total_cost = 2 - np.exp(-abs_cost/10) - np.exp(-census_cost/10)

    # >>> Cost aggregation
    total_cost = cv2.ximgproc.guidedFilter(guide=Il[border_size:-border_size, border_size:-border_size], \
                                           src=total_cost, radius=8, eps=100, dDepth=-1)
    total_cost = cv2.boxFilter(total_cost, -1, (5, 5))

    # >>> Disparity optimization
    total_cost = np.argmin(total_cost, axis=2)

    # >>> Disparity refinement
    for d in range(max_disp):
        total_cost[:, d] = total_cost[:, max_disp+5]

    total_cost = median_filter(total_cost, 3)
    
    labels = total_cost

    return labels.astype(np.uint8)