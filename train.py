import argparse
import datetime
import os
import time
from typing import Dict, Any

import torch
import tqdm

from sparse_rcnn.utils.config import cfg_from_yaml_file, cfg, cfg_from_list, log_config_to_file
from sparse_rcnn.utils import common_utils
from sparse_rcnn.dataloader import build_dataloader
from sparse_rcnn.dataloader.dataset import build_coco_transforms
from sparse_rcnn.model import SparseRCNN
from sparse_rcnn.loss import SparseRcnnLoss
from sparse_rcnn.solver.build_optimizer import build_optimizer, build_lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train sparse rcnn")
    parser.add_argument("--dataset", type=str, default="sparse_rcnn/configs/coco.yaml")
    parser.add_argument("--model", type=str, default="sparse_rcnn/configs/sparse_rcnn.yaml")
    parser.add_argument("--extra_tag", type=str, default="default")
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    cfg_from_yaml_file(args.dataset, cfg)
    cfg_from_yaml_file(args.model, cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def train_model(model, criterion, optimizer, train_loader, scheduler, start_epoch, total_epochs, device, logger, ckpt_save_dir,
                solver_cfg):
    model.train()
    with tqdm.trange(start_epoch, total_epochs, desc="epochs", dynamic_ncols=True) as tbar:
        for cur_epoch in tbar:
            train_one_epoch(model, optimizer, criterion, train_loader, scheduler, cur_epoch, device, logger)
            model_state = checkpoint_state(model=model, optimizer=optimizer, epoch=cur_epoch)
            save_checkpoint(model_state, os.path.join(ckpt_save_dir, "checkpoint_epoch_%d" % cur_epoch + 1))
            logger.info("Saving checkpoint to %s", ckpt_save_dir)
            eval(model, cur_epoch=cur_epoch, logger=logger)

        pass


def eval(model, cur_epoch, logger):
    logger.info("Evaluating checkpoint at epoch %d", cur_epoch + 1)
    pass


def train_one_epoch(model, optimizer, criterion, train_loader, scheduler, cur_epoch, device, logger):
    model.train()

    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)

    data_time = common_utils.AverageMeter()
    batch_time = common_utils.AverageMeter()
    forward_time = common_utils.AverageMeter()

    pbar = tqdm.trange(total_it_each_epoch, desc="train", dynamic_ncols=True)
    for cur_iter in range(total_it_each_epoch):
        end = time.time()
        batch = next(dataloader_iter)
        img, img_whwh, label = batch
        img, img_whwh = img.to(device), img_whwh.to(device)
        for t in label:
            for k in t.keys():
                if k in ['gt_boxes', 'gt_classes', 'image_size_xyxy', 'image_size_xyxy_tgt']:
                    t[k] = t[k].to(device)

        data_time = time.time()

        scheduler.step(cur_epoch * total_it_each_epoch + cur_iter)
        optimizer.zero_grad()

        output = model(img, img_whwh)
        loss = criterion(output, label)
        weighted_loss = loss["weighted_loss"]
        forward_timer = time.time()
        cur_forward_time = forward_timer - data_time

        weighted_loss.backward()
        optimizer.step()
        cur_batch_time = time.time() - end

        disp_dict = {
            "loss": weighted_loss.item(),
            "lr": float(scheduler.get_lr()[0]),
            "data": data_time,
            "batch": cur_batch_time,
            "forward": cur_forward_time,
        }
        pbar.set_postfix(disp_dict)
        pbar.update()


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def save_checkpoint(state: Dict[str, Any], filename="checkpoint"):
    filename = f"{filename}.pth"
    torch.save(state, filename)


def main():
    args, cfg = parse_args()
    output_dir = os.path.join("./output", args.extra_tag, "results")
    ckpt_dir = os.path.join("./output", args.extra_tag, "ckpt")
    log_file = os.path.join(output_dir, "log_train_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = common_utils.create_logger(log_file=log_file)
    device = torch.device(cfg.DEVICE)
    logger.info('**********************Start logging**********************')
    log_config_to_file(cfg, logger=logger)
    # ------------ Create dataloader ------------
    train_dataloader = build_dataloader(cfg,
                                        transforms=build_coco_transforms(cfg, mode="val"),
                                        batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                        dist=False,
                                        workers=0,
                                        mode="val")
    model = SparseRCNN(
        cfg,
        num_classes=cfg.MODEL.SparseRCNN.NUM_CLASSES,
        backbone="resnet18"
    )

    # start_epoch, model = load_model()
    start_epoch = 0

    model.to(device)

    criterion = SparseRcnnLoss(cfg)
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    train_model(model,
                criterion,
                optimizer,
                train_loader=train_dataloader,
                scheduler=lr_scheduler,
                start_epoch=start_epoch,
                total_epochs=cfg.SOLVER.NUM_EPOCHS,
                device=device,
                logger=logger,
                ckpt_save_dir=ckpt_dir,
                solver_cfg=cfg.SOLVER)


if __name__ == "__main__":
    main()
