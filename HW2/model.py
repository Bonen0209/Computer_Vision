import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv_1 = nn.Conv2d(1, 6, 5)
        self.max_pool_1 = nn.MaxPool2d(2)

        self.conv_2 = nn.Conv2d(6, 16, 5)
        self.max_pool_2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.max_pool_1(x)

        x = F.relu(self.conv_2(x))
        x = self.max_pool_2(x)

        x = x.view(-1, 16*4*4)

        x = F.relu(self.fc1(x))
        x = F.selu(self.fc2(x))
        
        out = F.log_softmax(self.fc3(x), dim=1)

        return out

    def name(self):
        return 'ConvNet'

class Fully(nn.Module):
    def __init__(self):
        super(Fully, self).__init__()

        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.selu(self.fc4(x))

        out = F.log_softmax(self.fc5(x), dim=1)
        
        return out

    def name(self):
        return 'Fully'

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)

        out = F.relu(out)

        return out
    
    def name(self):
        return 'Residual Block'

class ResNet(nn.Module):
    def __init__(self, layers, out_dim=100):
        super(ResNet, self).__init__()

        self.inplanes = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.Dropout(0.1)
        )

    def _make_layer(self, block, planes, blocks, stride):
        layers = []

        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        out = F.log_softmax(self.fc(x), dim=1)
        
        return out

    def name(self):
        return 'ResNet'

def main():
    model = ResNet([2, 2, 2, 2])
    print(sum(p.numel() for p in model.parameters()))
    print(model)

if __name__ == "__main__":
    main()