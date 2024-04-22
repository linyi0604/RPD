import torch
from torch import nn
import torchvision.models as models



class Diagnostic_model(nn.Module):
    def __init__(self, class_num=3, pretrained=True):
        super(Diagnostic_model, self).__init__()

        base_model = models.vgg16(pretrained=pretrained)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.extractor = base_model.classifier[:6]
        self.classifier = nn.Linear(4096, class_num)

    def forward(self, x, distillation=False):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.extractor(x)
        x = self.classifier(features)
        if distillation:
            return x, features
        return x