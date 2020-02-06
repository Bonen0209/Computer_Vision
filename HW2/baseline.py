###################################################################################
## Problem 4(b):                                                                 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import os, sys
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import alexnet
from data import get_dataloader_vgg

def feature_alexnet(imgs):
    extractor = alexnet(pretrained=True).features
    extractor.eval()

    feat = extractor(imgs).view(imgs.size(0), 256, -1)
    feat = torch.mean(feat, 2)
    feat = feat.cpu().detach().numpy()

    return feat

def main():
    # Specifiy data folder path
    folder = sys.argv[1]
    
    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader_vgg(folder, batch_size=32)

    feat_loader = [[], []]
    for x, label in train_loader:
        feat = feature_alexnet(x)

        label = label.cpu().detach().numpy()

        feat_loader[0].append(feat)
        feat_loader[1].append(label)
    
    feat_loader[0] = np.vstack(feat_loader[0])
    feat_loader[1] = np.concatenate(feat_loader[1], axis=0)

    knc = KNC(n_neighbors=100)
    knn_model = knc.fit(feat_loader[0], feat_loader[1])

    val_feat_loader = [[], []]
    for x, label in val_loader:
        feat = feature_alexnet(x)
        label = label.cpu().detach().numpy()

        val_feat_loader[0].append(feat)
        val_feat_loader[1].append(label)

    val_feat_loader[0] = np.vstack(val_feat_loader[0])
    val_feat_loader[1] = np.concatenate(val_feat_loader[1], axis=0)

    result = knn_model.score(val_feat_loader[0], val_feat_loader[1])

    print(f'Accuracy: {result}')

if __name__ == "__main__":
    main()