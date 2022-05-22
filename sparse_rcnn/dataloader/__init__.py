from torch.utils.data import DataLoader

from .sampler import RandomSampler, AspectRatioBasedSampler, DistributedGroupSampler
from .dataset import CocoDataset
from .collate import Collate

__all__ = {
    "CocoDataset": CocoDataset
}


def build_dataloader(dataset_cfg, transforms, batch_size, dist, workers=4,
                     mode="train"):
    if dist:
        raise NotImplementedError

    if dataset_cfg.DATASET == "CocoDataset":
        dataset = __all__[dataset_cfg.DATASET](dataset_cfg, mode, transforms)
        if mode == "train":
            sampler = AspectRatioBasedSampler(
                dataset, batch_size, drop_last=True)
            dataloader = DataLoader(dataset,
                                    num_workers=workers,
                                    pin_memory=False,
                                    collate_fn=Collate(dataset_cfg),
                                    batch_sampler=sampler)
        else:  # val
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=workers,
                                    pin_memory=False,
                                    collate_fn=Collate(dataset_cfg),
                                    drop_last=False)
        return dataloader

    else:
        raise NotImplementedError
