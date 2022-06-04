import numpy as np
import torch
from fvcore.nn import sigmoid_focal_loss_jit
from torch import nn
from torch.nn import functional as F


class OtaMatcher(nn.Module):
    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 use_focal: bool = False):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        if self.use_focal:
            self.focal_loss_alpha = cfg.MODEL.LOSS.FOCAL_LOSS_ALPHA
            self.focal_loss_gamma = cfg.MODEL.LOSS.FOCAL_LOSS_GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        for index in range(bs):
            tgt_ids = targets[index]["gt_classes"]
            tgt_boxes = targets[index]["gt_boxes"]
            num_gt = len(tgt_ids)
            out_prob = outputs["pred_logits"][index]
            out_boxes = outputs["pred_boxes"][index]

            # Compute the focal loss for each image
            if self.use_focal:
                alpha = self.focal_loss_alpha
                gamma = self.focal_loss_gamma
                tgt_ids_onehot = F.one_hot(tgt_ids, num_classes=self.num_classes).float()
                loss_cls = sigmoid_focal_loss_jit(
                    out_prob.unsqueeze(0).expand(num_gt, num_queries, -1),
                    tgt_ids_onehot.unsqueeze(1).expand(num_gt, num_queries, -1),
                    alpha=alpha,
                    gamma=gamma,
                    reduction="none"
                ).sum(dim=-1)  # [num_gt, num_queries]
                loss_bg = sigmoid_focal_loss_jit(
                    out_prob,
                    torch.zeros_like(out_prob),
                    alpha=alpha,
                    gamma=gamma,
                    reduction="none"
                )  # [num_queries]
            else:
                raise NotImplementedError

            # Compute the L1 loss for each image
            image_size_out = targets["image_size_xyxy"][index]
            out_boxes_ = out_boxes/image_size_out
            image_size_tgt = targets["image_size_xyxy_tgt"][index]
            tgt_boxes_ = tgt_boxes/image_size_tgt
            cost_bbox = torch.cdist()



