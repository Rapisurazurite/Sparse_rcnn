from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .sampler import RandomSampler, AspectRatioBasedSampler, DistributedGroupSampler
from .dataset import CocoDataset
from .collate import Collate

__all__ = {
    "CocoDataset": CocoDataset
}


def build_dataloader(dataset_cfg, transforms, batch_size, dist, workers=4,
                     mode="train"):

    if dataset_cfg.DATASET not in __all__.keys():
        raise ValueError("Dataset {} not supported".format(dataset_cfg.DATASET))

    dataset = __all__[dataset_cfg.DATASET](dataset_cfg, mode, transforms)

    if dist:
        sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
    else:
        sampler = None
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=workers,
        shuffle=False,
        collate_fn=Collate(dataset_cfg),
        drop_last=False,
        sampler=sampler
    )

    return dataloader
