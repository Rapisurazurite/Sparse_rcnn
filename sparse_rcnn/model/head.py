import torch
from torch import nn
from torchvision.ops import MultiScaleRoIAlign


class DynamicHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, features):
        dict_features = dict([(f"f{i+2}", features[i]) for i in range(len(features))])

        pass
