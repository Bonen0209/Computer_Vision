# Server only
# -*- coding: future_fstrings -*- 

from glob import glob
from os.path import join
import argparse
import cv2
import numpy as np

class Rgb2Gray(object):

    def __init__(self, weight_r = 0.299, weight_g = 0.587, weight_b = 0.114):
        self.weight_r = weight_r
        self.weight_g = weight_g
        self.weight_b = weight_b

    def rgb2gray(self, img, r = 0, g = 1, b = 2):
        img_r = img[:, :, r]
        img_g = img[:, :, g]
        img_b = img[:, :, b]

        img_y = self.weight_r * img_r + self.weight_g * img_g + self.weight_b * img_b
        img_y = img_y.astype(np.uint8)
        return img_y

def main():
    parser = argparse.ArgumentParser(description='RGB to GRAY')
    parser.add_argument('--input_folder', default='./HW1/testdata/', help='path of input folder')
    parser.add_argument('--output_folder', default='./HW1/conventional_rgb2gray/', help='path of output folder')

    args = parser.parse_args()

    filenames = [filename.split('/')[-1] for filename in glob(join(args.input_folder, '1*.png'))]

    Conventional = Rgb2Gray()

    for filename in filenames:
        img = cv2.imread(args.input_folder + filename)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = Conventional.rgb2gray(img_rgb)
        cv2.imwrite(args.output_folder + filename.split('.')[0] + '_gray.png', img_gray)

if __name__ == '__main__':
    main()