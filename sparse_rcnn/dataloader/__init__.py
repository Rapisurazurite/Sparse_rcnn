from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from .sampler import RandomSampler, AspectRatioBasedSampler, DistributedGroupSampler
from .dataset import CocoDataset
from .collate import Collate

__all__ = {
    "CocoDataset": CocoDataset
}


def build_dataloader(dataset_cfg, transforms, batch_size, dist, workers=4, pin_memory=True,
                     mode="train"):
    if dist and mode == "val":
        raise ValueError("DDP currently does not support validation")

    if dataset_cfg.DATASET not in __all__.keys():
        raise ValueError("Dataset {} not supported".format(dataset_cfg.DATASET))

    dataset = __all__[dataset_cfg.DATASET](dataset_cfg, mode, transforms)

    if dist:
        # sampler = DistributedSampler(dataset, shuffle=(mode == "train"))
        sampler = DistributedGroupSampler(dataset, samples_per_gpu=batch_size)
    else:
        # Note: Use this sampler to reduce memory usage.
        sampler = AspectRatioBasedSampler(dataset, batch_size=batch_size, drop_last=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=workers,
        shuffle=(mode == "train"),
        collate_fn=Collate(dataset_cfg),
        drop_last=False,
        sampler=sampler if mode == "train" else None,
    )

    return dataloader
