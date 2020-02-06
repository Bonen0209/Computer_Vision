# Server only
# -*- coding: future_fstrings -*- 

from os import makedirs
from os.path import exists
from six.moves import cPickle as pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import argparse
import cv2
import numpy as np
from rgb2gray import Rgb2Gray
from joint_bilateral_filter import Joint_bilateral_filter

class Adv_Rgb2Gray(object):

    def __init__(self, img, sigma_s, sigma_r, border_type, image_name, output_folder):
        self.JBF = Joint_bilateral_filter(sigma_s, sigma_r, border_type)
        self.RGB2GRAY = Rgb2Gray()
        self.image_name = image_name
        self.img = img  # RGB
        self.img_base = self.JBF.joint_bilateral_filter(self.img, self.img)
        self.weights = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]
        self.loss = {}
        self.output_folder = f'{output_folder}s_{sigma_s}_r_{sigma_r}/'
        if not exists(self.output_folder):
            makedirs(self.output_folder)

    def cal_diff(self, img_base, img_diff):
        return np.sum(np.abs(img_diff.astype(np.float64) - img_base.astype(np.float64)))

    def cal_loss(self, weight):
        img_diff = cv2.imread(f'{self.output_folder}{self.image_name}_{weight[0]}_{weight[1]}_{weight[2]}.png')
        img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2RGB)

        self.loss[weight] = self.cal_diff(self.img_base, img_diff)

    def process_image(self, weight):
        self.RGB2GRAY.weight_r = weight[0]
        self.RGB2GRAY.weight_g = weight[1]
        self.RGB2GRAY.weight_b = weight[2]

        img_gray = self.RGB2GRAY.rgb2gray(self.img)
        img_diff = self.JBF.joint_bilateral_filter(self.img, img_gray)

        img_diff = cv2.cvtColor(img_diff, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{self.output_folder}{self.image_name}_{weight[0]}_{weight[1]}_{weight[2]}.png', img_diff)

    def adv_rgb2gray(self):
        with ProcessPoolExecutor() as executor:
            executor.map(self.process_image, self.weights)
        
        with ThreadPoolExecutor() as executor:
            executor.map(self.cal_loss, self.weights)
    
def main():
    parser = argparse.ArgumentParser(description='Advanced RGB to GRAY')
    parser.add_argument('--input_path', default='./HW1/testdata/1a.png', help='path of input image')
    parser.add_argument('--output_folder', default='./HW1/advanced_rgb2gray/1a/', help='path of output folder')
    parser.add_argument('--sigma_s', default=3, type=int, help='sigma of spatial kernel')
    parser.add_argument('--sigma_r', default=0.2, type=float, help='sigma of range kernel')

    args = parser.parse_args()

    img = cv2.imread(args.input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    ADV = Adv_Rgb2Gray(img_rgb, args.sigma_s, args.sigma_r, 'reflect', args.input_path.split('/')[-1].split('.')[0],  args.output_folder)
    ADV.adv_rgb2gray()

    with open(f'{args.output_folder}s_{args.sigma_s}_r_{args.sigma_r}/s_{args.sigma_s}_r_{args.sigma_r}.pkl', 'wb') as f:
        pickle.dump(ADV.loss, f)

if __name__ == '__main__':
    main()