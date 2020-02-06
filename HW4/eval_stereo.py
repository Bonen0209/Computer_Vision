import numpy as np
import cv2
import time
import sys
from computeDisp import computeDisp

def evaluate(input_path, gt_path, scale_factor, threshold=1.0):
    disp_gt = cv2.imread(gt_path, -1)
    disp_gt = np.int32(disp_gt/scale_factor)
    disp_input = cv2.imread(input_path, -1)
    disp_input = np.int32(disp_input/scale_factor)

    nr_pixel = 0
    nr_error = 0
    h, w = disp_gt.shape
    for y in range(0, h):
        for x in range(0, w):
            if disp_gt[y, x] > 0:
                nr_pixel += 1
                if np.abs(disp_gt[y, x] - disp_input[y, x]) > threshold:
                    nr_error += 1

    return float(nr_error)/nr_pixel

def main():
    print('Tsukuba')
    tic = time.time()
    img_left = cv2.imread('./HW4/testdata/tsukuba/im3.png')
    img_right = cv2.imread('./HW4/testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('HW4/output/tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    img_left = cv2.imread('./HW4/testdata/venus/im2.png')
    img_right = cv2.imread('./HW4/testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('HW4/output/venus.png', np.uint8(labels * scale_factor))
    
    print('Teddy')
    img_left = cv2.imread('./HW4/testdata/teddy/im2.png')
    img_right = cv2.imread('./HW4/testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('HW4/output/teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./HW4/testdata/cones/im2.png')
    img_right = cv2.imread('./HW4/testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('HW4/output/cones.png', np.uint8(labels * scale_factor))
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    print('-----evaluate-----')
    print('[Bad Pixel Ratio]')
    avg = 0

    res = evaluate('./HW4/tsukuba.png', './HW4/testdata/tsukuba/disp3.pgm', scale_factor=16)
    avg += res
    print('Tsukuba: %.2f%%' % (res*100))

    res = evaluate('./HW4/venus.png', './HW4/testdata/venus/disp2.pgm', scale_factor=8)
    avg += res
    print('Venus: %.2f%%' % (res*100))

    res = evaluate('./HW4/teddy.png', './HW4/testdata/teddy/disp2.png', scale_factor=4)
    avg += res
    print('Teddy: %.2f%%' % (res*100))

    res = evaluate('./HW4/cones.png', './HW4/testdata/cones/disp2.png', scale_factor=4)
    avg += res
    print('Cones: %.2f%%' % (res*100))

    print('Average: %.2f%%' % (avg*100/4))

if __name__ == '__main__':
    main()