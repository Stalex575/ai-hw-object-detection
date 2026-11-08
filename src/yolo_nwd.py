"""YOLO26 loss helpers; NWD C is measured in resized input-image pixels.

NWD: Wang et al., https://arxiv.org/abs/2110.13389
"""

import torch
from ultralytics.utils import loss as losses
from ultralytics.utils.tal import TaskAlignedAssigner

_Assign = TaskAlignedAssigner.forward

_BboxLoss = losses.BboxLoss


def nwd_similarity(first, second, constant=12.8):
    first_center = (first[..., :2] + first[..., 2:]) / 2
    second_center = (second[..., :2] + second[..., 2:]) / 2
    delta = torch.cat((first_center - second_center,
                       ((first[..., 2:] - first[..., :2]) -
                        (second[..., 2:] - second[..., :2])) / 2), dim=-1)
    return torch.exp(-torch.linalg.vector_norm(delta, dim=-1, keepdim=True) / constant)


class NWDBoxLoss(_BboxLoss):
    alpha = 0.5
    constant = 12.8

    def __init__(self, reg_max=16):
        super().__init__(reg_max)
        self.alpha, self.constant = type(self).alpha, type(self).constant

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        iou_loss, auxiliary_loss = super().forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride)
        positive_stride = stride.expand(*fg_mask.shape, 1)[fg_mask]
        similarity = nwd_similarity(pred_bboxes[fg_mask] * positive_stride,
                                    target_bboxes[fg_mask] * positive_stride, self.constant)
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        nwd_loss = ((1 - similarity) * weight).sum() / target_scores_sum
        return (1 - self.alpha) * iou_loss + self.alpha * nwd_loss, auxiliary_loss


def configure(alpha=None, constant=12.8):
    if alpha is None:
        losses.BboxLoss = _BboxLoss
        return
    if not 0 <= alpha <= 1 or constant <= 0:
        raise ValueError("Invalid NWD blend or normalization constant")
    NWDBoxLoss.alpha, NWDBoxLoss.constant = alpha, constant
    losses.BboxLoss = NWDBoxLoss


@torch.no_grad()
def _assign_per_image(self, scores, boxes, anchors, labels, targets, mask):
    # Assignment is independent across images; slicing avoids the dense batch-sized workspace.
    if scores.device.type != 'mps':
        return _Assign(self, scores, boxes, anchors, labels, targets, mask)
    result = [_Assign(self, scores[i:i+1], boxes[i:i+1], anchors,
                      labels[i:i+1], targets[i:i+1], mask[i:i+1])
              for i in range(scores.shape[0])]
    return tuple(torch.cat(parts) for parts in zip(*result))


def configure_mps_assignment(enabled=False):
    TaskAlignedAssigner.forward = _assign_per_image if enabled else _Assign
