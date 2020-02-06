import cv2
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

def read_img(img_path, gray=True):
    if gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float64)
    else:
        img = cv2.imread(img_path).astype(np.float64)

    return img
        
def read_img2flatten(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).flatten()
    img = img.astype(np.float64)

    label = img_path.split('/')[-1]
    label, index = map(int, label.split('.')[0].split('_'))

    return img, label, index

def mean_square_error(img_base, img_diff):
    img_base = img_base.astype(np.float64)
    img_diff = img_diff.astype(np.float64)

    mse = (np.square(img_diff - img_base)).mean()

    return mse

def save_flatten2img(img_path, img_flatten, h=56, w=46):
    img_flatten = img_flatten.astype(np.uint8)
    img_flatten = img_flatten.reshape(h, w)

    cv2.imwrite(img_path, img_flatten)

def normalize_img(img):
    return ((img - img.min()) / (img.max() - img.min()) * 255).astype('uint8')

def plot_tsne(datas, labels, img_path, img_name):
    tsne = TSNE(n_components=2, random_state=0)
    data_2ds = tsne.fit_transform(datas)

    X, Y = data_2ds[:, 0], data_2ds[:, 1]
    L = labels

    fig = plt.figure()
    ax = plt.scatter(X, Y, c=L, cmap=plt.cool())
    plt.colorbar(ax)
    plt.title('t-SNE')
    plt.savefig(f'{img_path}/{img_name}.png')

def plot_tsne_gif(datas, labels, centroids, img_path, img_name):
    X, Y = datas[:, 0], datas[:, 1]
    L = labels
    C_X, C_Y = centroids[:, 0], centroids[:, 1]

    fig = plt.figure()
    ax = plt.scatter(X, Y, c=L, cmap=plt.cool())
    cx = plt.scatter(C_X, C_Y, c='black', s=100)
    plt.colorbar(ax)
    plt.title('K means')
    plt.savefig(f'{img_path}/{img_name}.png')

def plot_loss_accuracy(X, Y, folder_path, name):
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.8625)
    par = host.twinx()

    p1, = host.plot(X, Y[0], color='red', label='Accuracy')
    p2, = par.plot(X, Y[1], color='blue', label='Loss')
    
    host.set_xlabel('Epoch')
    host.set_ylabel('Accuracy')
    par.set_ylabel('Loss')

    lines = [p1, p2]
    host.legend(lines, [l.get_label() for l in lines], loc='upper left')

    plt.savefig(f'{folder_path}{name}_accuracy_loss.png')


def main():
    input_img = read_img('HW2/hw2_data/hw2-3_data/1_2.png')

    print(mean_square_error(input_img, input_img))

if __name__ == '__main__':
    main()