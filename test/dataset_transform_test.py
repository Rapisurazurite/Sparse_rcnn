import argparse

from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg
from sparse_rcnn.dataloader.dataset import CocoDataset
import albumentations as A

parser = argparse.ArgumentParser(description="Test coco dataset")
coco_config = "../sparse_rcnn/configs/coco.yaml"
cfg_from_yaml_file(coco_config, cfg)

transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2)])

dataset = CocoDataset(cfg, 'val')
print(dataset.__len__())

for i in range(10):
    print(f"img shape[{i}]: {dataset[i][0].shape}")
sample = dataset.__getitem__(0)
print(sample)

sample[0].transp