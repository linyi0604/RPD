import torch
from torch import nn
import torchvision.models as models


class Regional_Interaction_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_i = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.k_i = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)
        self.v_i = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)

        self.q_r = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.k_r = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
        self.v_r = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)

        self.fusion = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1)


    def forward(self, img, mask):
        q_i = self.q_i(img)
        k_i = self.k_i(img)
        v_i = self.v_i(img)

        q_r = self.q_r(mask)
        k_r = self.k_r(mask)
        v_r = self.v_r(mask)
        
        f_i = torch.sigmoid(q_i * k_i) * v_i + torch.sigmoid(q_i * k_r) * v_r
        f_r = torch.sigmoid(q_r * k_r) * v_r + torch.sigmoid(q_r * k_i) * v_i

        f_f = self.fusion(torch.cat([f_i, f_r], dim=1))
        return f_f



class Prior_fused_model(nn.Module):
    def __init__(self, class_num=3, pretrained=True):
        super(Prior_fused_model, self).__init__()

        base_model = models.vgg16(pretrained=pretrained)
        self.rim = Regional_Interaction_Module()
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.extractor = base_model.classifier[:6]
        self.classifier = nn.Linear(4096, class_num)

    def forward(self, x, mask, distillation=False):
        x = self.rim(x, mask)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        features = self.extractor(x)
        x = self.classifier(features)
        if distillation:
            return features
        return x