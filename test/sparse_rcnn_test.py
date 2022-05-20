import env
import os
import sys

import argparse
import torch
import torch.nn as nn

from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg
from sparse_rcnn.dataset import CocoDataset
from sparse_rcnn.model import SparseRCNN, DynamicHead

parser = argparse.ArgumentParser(description="Test coco dataset")
coco_config = "../sparse_rcnn/configs/coco.yaml"
model_config = "../sparse_rcnn/configs/sparse_rcnn.yaml"
cfg_from_yaml_file(coco_config, cfg)
cfg_from_yaml_file(model_config, cfg)

model = SparseRCNN(cfg, num_classes=81, backbone='resnet18', neck=None, head=None)
# head = DynamicHead(cfg, roi_input_shape=)
input = torch.randn(*[2, 3, 800, 1216])
img_whwh = torch.tensor([[721, 480, 721, 480],
                         [800, 1216, 800, 1216]])
model.eval()
output = model(input, img_whwh)

for out in output:
    print(out.shape)
