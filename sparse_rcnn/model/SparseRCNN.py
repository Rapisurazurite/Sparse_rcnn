import timm
import torch
from torch import nn

_available_backbones = {
    "resnet18": timm.create_model("resnet18", features_only=True, out_indices=(1, 2, 3, 4), pretrained=True,
                                  num_classes=0,
                                  global_pool="")
}


class FPN(nn.Module):
    def __init__(self, c2, c3, c4, c5, inner_channel=256, bias=False):
        super(FPN, self).__init__()
        self.c2_to_f2 = nn.Conv2d(c2, inner_channel, 1, 1, 0, bias=bias)
        self.c3_to_f3 = nn.Conv2d(c3, inner_channel, 1, 1, 0, bias=bias)
        self.c4_to_f4 = nn.Conv2d(c4, inner_channel, 1, 1, 0, bias=bias)
        self.c5_to_f5 = nn.Conv2d(c5, inner_channel, 1, 1, 0, bias=bias)

        self.p2_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p3_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p4_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)
        self.p5_out = nn.Conv2d(inner_channel, inner_channel, 3, 1, 1, bias=bias)

    def forward(self, c2, c3, c4, c5):
        latent_2 = self.c2_to_f2(c2)
        latent_3 = self.c3_to_f3(c3)
        latent_4 = self.c4_to_f4(c4)
        latent_5 = self.c5_to_f5(c5)

        f4 = latent_4 + nn.UpsamplingBilinear2d(size=(latent_4.shape[2:]))(latent_5)
        f3 = latent_3 + nn.UpsamplingBilinear2d(size=(latent_3.shape[2:]))(f4)
        f2 = latent_2 + nn.UpsamplingBilinear2d(size=(latent_2.shape[2:]))(f3)
        p2 = self.p2_out(f2)
        p3 = self.p3_out(f3)
        p4 = self.p4_out(f4)
        p5 = self.p5_out(latent_5)
        return p2, p3, p4, p5


class SparseRCNN(torch.nn.Module):
    def __init__(self, cfg, num_classes, backbone, neck, head):
        super(SparseRCNN, self).__init__()
        assert backbone in _available_backbones, f"{backbone} is not available"
        self.cfg = cfg
        self.in_channels = 256
        # model components
        self.backbone: nn.Module = _available_backbones[backbone]
        self.fpn = FPN(*self.backbone.feature_info.channels())

        # embedding parameters
        self.init_proposal_features = nn.Embedding(self.cfg.MODEL.NUM_PROPOSALS, self.in_channels)
        self.init_proposal_boxes = nn.Embedding(self.cfg.MODEL.NUM_PROPOSALS, 4)  # cx, cy, w, h
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)  # center
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)  # size

    def forward(self, x):
        batch_size, *image_wh = x.shape
        features = self.backbone(x)
        features = self.fpn(*features)

        print(f"batch_size: {batch_size}")
        print(f"image_wh: {image_wh}")

        return features
