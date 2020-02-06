from os import makedirs
from os.path import exists
import sys
from pprint import pprint
import cv2
import numpy as np
from math import sqrt
from pca import Principal_Components_Analysis
from helper import plot_tsne_gif

class K_means(object):

    def __init__(self, input_folder='HW2/hw2_data/hw2-3_data/', output_folder='HW2/hw2_result/hw2-3_result/'):
        self.input_folder = input_folder
        self.output_folder = output_folder

        if not exists(f'{self.output_folder}/k_means/'):
            makedirs(f'{self.output_folder}/k_means/')  

        self.PCA = Principal_Components_Analysis(self.input_folder, self.output_folder, kmeans=True)
        
        self.data, self.label = self.init_data(10)

        self.centroid = np.random.uniform(np.min(self.data) / 2, np.max(self.data) / 2, 100).reshape(10, 10)

    def init_data(self, dim=10):
        train_data = np.concatenate((self.PCA.train_data, self.PCA.test_data), axis=0)
        label = np.concatenate((self.PCA.train_label, self.PCA.test_label), axis=0)

        mean_face, eigen_faces = self.PCA.pca(train_data, save=False)
        data = self.PCA.reduce_dim(train_data, mean_face, eigen_faces, n=dim)
        
        return data, label

    def w_eucli_dist(self, vec_base, vec_diff):
        ws = np.array([0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        diff = vec_diff - vec_base

        return sqrt(np.dot(np.square(diff), ws))

    def k_means(self, k=10):
        n, dim = list(self.data.shape)

        count = 1
        while True:
            plot_tsne_gif(self.data[:, :2], self.label, self.centroid[:, :2], f'{self.output_folder}/k_means/', str(count))
            temp_dict = {i:[] for i in range(k)}
            for i in range(n):
                temp_dist = np.zeros(10)

                for j in range(k):
                    dist = self.w_eucli_dist(self.data[i], self.centroid[j])
                    temp_dist[j] = dist

                temp_dict[np.argmin(temp_dist)].append(self.data[i])

            temp_centroids = []
            for key in temp_dict:
                if len(temp_dict[key]) > 0:
                    temp = np.array(temp_dict[key])
                    centroid = np.mean(temp, axis=0)
                    temp_centroids.append(centroid)
                else:
                    temp_centroids.append(self.centroid[key])

            temp_centroid = np.array(temp_centroids)

            if (self.centroid == temp_centroid).all():
                break

            self.centroid = temp_centroid

            count += 1

def main():
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if not exists(output_folder):
        makedirs(output_folder)

    Kmeans = K_means(input_folder, output_folder)
    Kmeans.k_means()

if __name__ == '__main__':
    main()