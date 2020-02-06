# Server only
# -*- coding: future_fstrings -*- 

from os import listdir
from six.moves import cPickle as pickle
import argparse
import cv2
import numpy as np
from math import sqrt
from joint_bilateral_filter import Joint_bilateral_filter
from rgb2gray import Rgb2Gray
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

class Local_min(object):

    def __init__(self, input_folder):
        self.sigma_s = [1, 2, 3]
        self.sigma_r = [0.05, 0.1, 0.2]
        self.filenames = [f'{input_folder}s_{s}_r_{r}/s_{s}_r_{r}.pkl' for s in self.sigma_s for r in self.sigma_r]
        self.votes = {}
    
    def plot3Dscatter(self, data, output_folder):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        dots = [(key, data[key]) for key in data]

        x = [d_x[0][0] for d_x in dots]
        y = [d_y[0][1] for d_y in dots]
        z = [d_z[0][2] for d_z in dots]
        c = [d_v[1] for d_v in dots]

        img = ax.scatter(x, y, z, c=c, cmap=plt.cool())
        fig.colorbar(img)
        ax.view_init(30, 30)
        plt.savefig(f'{output_folder}/Scatter.png')

    def distance(self, a, b):
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def local_min(self):
        for filename in self.filenames:
            with open(filename, 'rb') as f:
                loss_table = pickle.load(f)

                for key in loss_table:
                    nbrs = [nbr for nbr in loss_table if self.distance(key, nbr) <= 0.1 * sqrt(2) and key != nbr]

                    losses = [loss_table[nbr] for nbr in nbrs]

                    if loss_table[key] <= min(losses):
                        if key not in self.votes:
                            self.votes[key] = 1
                        else:
                            self.votes[key] += 1
                    
                self.plot3Dscatter(loss_table, filename.rsplit('/', 1)[0])
        
        return self.votes

def main():
    parser = argparse.ArgumentParser(description='Local minimum')
    parser.add_argument('--input_path', default='./HW1/testdata/1a.png', help='path of input image')
    parser.add_argument('--input_folder', default='./HW1/advanced_rgb2gray/1a/', help='path of input image folder')
    
    args = parser.parse_args()

    LMIN = Local_min(args.input_folder)
    votes = LMIN.local_min()
    most_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)[:3]

    img = cv2.imread(args.input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    RGB2GRAY = Rgb2Gray()
    for i, (weight, vote) in enumerate(most_votes):
        RGB2GRAY.weight_r = weight[0]
        RGB2GRAY.weight_g = weight[1]
        RGB2GRAY.weight_b = weight[2]

        img_gray = RGB2GRAY.rgb2gray(img_rgb)
        cv2.imwrite(f"{args.input_folder}{args.input_path.rsplit('/')[-1].split('.')[0]}_y{i}.png", img_gray)

        print(weight, vote)

if __name__ == '__main__':
    main()