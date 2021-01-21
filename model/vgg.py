import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import models

class VGG19(nn.Module):
    def __init__(self, out_features):
        super(VGG19, self).__init__()
        self.vgg1 = models.vgg19(pretrained=True)
        self.vgg2 = models.vgg19(pretrained=False)

        self.classifier = nn.Linear(2000, out_features)
        self.Softmax = nn.Softmax()

    def forward(self, x):
        f1 = self.vgg1(x)
        f2 = self.vgg2(x)

        features = torch.cat((f1,f2), dim=1)

        out = self.classifier(features)
        out = self.Softmax(out)
        
        return out

class VGG19_feature(nn.Module):
    def __init__(self, out_features):
        super(VGG19_feature, self).__init__()
        self.vgg1 = models.vgg19(pretrained=True)
        self.vgg2 = models.vgg19(pretrained=False)

        self.classifier = nn.Linear(2000, out_features)
        self.Softmax = nn.Softmax()
    def forward(self, x):
        f1 = self.vgg1(x)
        f2 = self.vgg2(x)

        features = torch.cat((f1,f2), dim=1)

        out = self.classifier(features)
        out = self.Softmax(out)
        return features

class VGG19_db_train(nn.Module):
    def __init__(self, out_features):
        super(VGG19_db_train, self).__init__()
        self.classifier = nn.Linear(2000, out_features)
        self.Softmax = nn.Softmax()
    def forward(self, feature):
        out = self.classifier(feature)
        out = self.Softmax(out)
        return out