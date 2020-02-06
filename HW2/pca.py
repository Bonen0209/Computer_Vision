from os import listdir, makedirs
from os.path import join, exists
import sys
import cv2
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import StratifiedKFold
from helper import read_img2flatten, read_img, save_flatten2img
from helper import mean_square_error, normalize_img
from helper import plot_tsne

class Principal_Components_Analysis(object):

    def __init__(self, input_folder='HW2/hw2_data/hw2-3_data/', output_folder='HW2/hw2_result/hw2-3_result/', kmeans=False):
        self.input_folder = input_folder
        self.output_folder = output_folder

        self.kmeans = kmeans

        self.train_data, \
        self.train_label, \
        self.test_data, \
        self.test_label = self.get_data()

    def get_data(self):
        train_data = []
        train_label = []

        test_data = []
        test_label = []

        for img_path in listdir(self.input_folder):
            img_path = join(self.input_folder, img_path)

            img, label, index = read_img2flatten(img_path)

            if self.kmeans:
                if label <= 10:
                    if index <= 7:
                        train_data.append(img)
                        train_label.append(label)
                    else:
                        test_data.append(img)
                        test_label.append(label)
            else:
                if index <= 7:
                    train_data.append(img)
                    train_label.append(label)
                else:
                    test_data.append(img)
                    test_label.append(label)
                        

        train_data = np.array(train_data)
        train_label = np.array(train_label)

        test_data = np.array(test_data)
        test_label = np.array(test_label)
        
        # print('Image loaded')

        return train_data, train_label, test_data, test_label
    
    def pca(self, data, k=280, save=True, gram_matrix_trick=False):
        X = np.copy(data)
        n, dim = list(X.shape)
        
        M = np.mean(X, axis=0).astype(np.uint8).reshape(dim, 1)
        X = X.T - M

        if gram_matrix_trick:
            G = np.dot(X.T, X)
            U, s, V = np.linalg.svd(G, full_matrices=False)
            U = preprocessing.normalize(np.dot(X, U).T, norm='l2').T
        else:
            U, s, V = np.linalg.svd(X, full_matrices=False)
        
        E = U[:, :k]
        
        if save:
            save_flatten2img(f'{self.output_folder}mean_face.png', M)
            for i in range(5):
                if not exists(f'{self.output_folder}/eigen_faces/'):
                    makedirs(f'{self.output_folder}/eigen_faces/')
                save_flatten2img(f'{self.output_folder}/eigen_faces/eigen_face_{i}.png', normalize_img(U[:, i]))

        # print('PCA finished')

        return M, E
    
    def reconstruct_img(self, img_face, mean_face, eigen_faces, k=5, h=56, w=46):
        face = img_face.flatten() - mean_face.flatten()

        weights = np.dot(face, eigen_faces[:, :k])
        
        diff_face = np.dot(weights, eigen_faces.T[:k]).reshape(h * w).astype(np.float64)
        re_face = (diff_face + mean_face.flatten()).reshape(h, w)
        re_face.astype(np.uint8)

        # print('Reconstruct finished')
        
        return re_face
    
    def reduce_dim(self, data, mean_face, eigen_faces, n=100):
        X = np.copy(data)

        X = X - mean_face.T
        V = np.dot(X, eigen_faces[:, :n])

        # print('Reduce finished')

        return V
    
    def knn(self, train_data, train_label, test_data, test_label, k):
        knc = KNC(n_neighbors=k)
        knn_model = knc.fit(train_data, train_label)
        result = knn_model.score(test_data, test_label)

        # print('KNN finished')

        return result

def main():
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    if not exists(output_folder):
        makedirs(output_folder)

    PCA = Principal_Components_Analysis(input_folder, output_folder)
    mean_face, eigen_faces = PCA.pca(PCA.train_data)

    input_img = read_img(f'{input_folder}8_6.png')

    for i in [5, 50, 150, eigen_faces.shape[1]]:
        recover_face = PCA.reconstruct_img(input_img , mean_face, eigen_faces, k=i)
        
        if not exists(f'{output_folder}/recover_faces/'):
            makedirs(f'{output_folder}/recover_faces/')        

        cv2.imwrite(f'{output_folder}/recover_faces/recover_{i}.png', recover_face)
        print(f'MSE:\t{mean_square_error(input_img, recover_face):.3f}')
    
    plot_tsne(PCA.reduce_dim(PCA.test_data ,mean_face, eigen_faces), PCA.test_label, output_folder, 'PCA_t-SNE')

    print()

    for k in [1, 3, 5]:
        for n in [3, 10, 39]:
            avg_score = 0

            PCA = Principal_Components_Analysis(input_folder, output_folder)

            skf = StratifiedKFold(n_splits=3)            
            for train_idx, valid_idx in skf.split(PCA.train_data, PCA.train_label):
                train_data = PCA.train_data[train_idx]
                train_label = PCA.train_label[train_idx]

                valid_data = PCA.train_data[valid_idx]
                valid_label = PCA.train_label[valid_idx]

                mean_face, eigen_faces = PCA.pca(train_data)

                train_vec = PCA.reduce_dim(train_data, mean_face, eigen_faces, n=n)
                valid_vec = PCA.reduce_dim(valid_data, mean_face, eigen_faces, n=n)

                score = PCA.knn(train_vec, train_label, valid_vec, valid_label, k=k)
                avg_score += score

            avg_score /= 3

            print(f'k = {k}\tn = {n}\tAccuracy: {avg_score:.3f}')
    
    mean_face, eigen_faces = PCA.pca(PCA.train_data)

    train_vec = PCA.reduce_dim(PCA.train_data, mean_face, eigen_faces, n=39)
    test_vec = PCA.reduce_dim(PCA.test_data, mean_face, eigen_faces, n=39)

    score = PCA.knn(train_vec, PCA.train_label, test_vec, PCA.test_label, k=1)
    print(f'Testing Accuracy: {score:.3f}')

if __name__ == '__main__':
    main()