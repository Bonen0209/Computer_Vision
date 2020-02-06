import numpy as np
import torch
import torch.nn as nn
from torchvision.models import alexnet
from model import ResNet
from helper import plot_tsne
from data import get_dataloader_vgg
from tqdm import tqdm

folder = 'HW2/hw2_data/hw2-4_data/problem2/'
result_folder = 'HW2/hw2_result/hw2-4_result/problem2/'

def feature_alexnet(imgs):
    extractor = alexnet(pretrained=True).features
    extractor.eval()

    feat = extractor(imgs).view(imgs.size(0), 256, -1)
    feat = torch.mean(feat, 2)
    feat = feat.cpu().detach().numpy()

    return feat

def feature_resnet(imgs):
    extractor = ResNet([2, 2, 2, 2])
    extractor.load_state_dict(torch.load('./HW2/checkpoint/%s.pth' % extractor.name(), map_location=torch.device('cpu')))

    layers = list(extractor.children())[:-1]
    extractor = nn.Sequential(*layers)
    extractor.eval()

    feat = extractor(imgs).view(imgs.size(0), 512)
    feat = feat.cpu().detach().numpy()
    
    return feat

def main():
    train_loader, val_loader = get_dataloader_vgg(folder, batch_size=32, feature=True)

    feat_loader = [[], []]
    for x, label in tqdm(train_loader):
        feat = feature_alexnet(x)

        label = label.cpu().detach().numpy()

        feat_loader[0].append(feat)
        feat_loader[1].append(label)
    
    feat_loader[0] = np.vstack(feat_loader[0])[:762, :]
    feat_loader[1] = np.concatenate(feat_loader[1], axis=0)[:762]

    plot_tsne(feat_loader[0], feat_loader[1], result_folder, 'AlexNet_TSNE_train')

    feat_loader = [[], []]
    for x, label in tqdm(train_loader):
        feat = feature_resnet(x)

        label = label.cpu().detach().numpy()

        feat_loader[0].append(feat)
        feat_loader[1].append(label)
    
    feat_loader[0] = np.vstack(feat_loader[0])[:762, :]
    feat_loader[1] = np.concatenate(feat_loader[1], axis=0)[:762]

    plot_tsne(feat_loader[0], feat_loader[1], result_folder, 'ResNet_TSNE_train')

    feat_loader = [[], []]
    for x, label in tqdm(val_loader):
        feat = feature_alexnet(x)

        label = label.cpu().detach().numpy()

        feat_loader[0].append(feat)
        feat_loader[1].append(label)
    
    feat_loader[0] = np.vstack(feat_loader[0])[:161, :]
    feat_loader[1] = np.concatenate(feat_loader[1], axis=0)[:161]

    plot_tsne(feat_loader[0], feat_loader[1], result_folder, 'AlexNet_TSNE_valid')

    feat_loader = [[], []]
    for x, label in val_loader:
        feat = feature_resnet(x)

        label = label.cpu().detach().numpy()

        feat_loader[0].append(feat)
        feat_loader[1].append(label)
    
    feat_loader[0] = np.vstack(feat_loader[0])[:161, :]
    feat_loader[1] = np.concatenate(feat_loader[1], axis=0)[:161]

    plot_tsne(feat_loader[0], feat_loader[1], result_folder, 'ResNet_TSNE_valid')

if __name__ == "__main__":
    main()
