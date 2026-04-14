# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
import math
from .metrics import bbox_iou, probiou
from .tal import bbox2dist

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_beta_loss(pred, target, beta=0.11, reduction="mean"):
    beta_t = torch.as_tensor(beta, device=pred.device, dtype=pred.dtype)
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta_t, 0.5 * diff * diff / beta_t, diff - 0.5 * beta_t)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")

def obb_to_polar(gt_bboxes, lambda_r=12.44):
    """
    gt_bboxes: (..., 5) => [cx, cy, w, h, theta]
    theta in [-pi/2, pi/2)
    return (..., 4) => [Mr, a1, a2, theta_m]
    """
    dtype = gt_bboxes.dtype
    device = gt_bboxes.device

    pi_t = torch.tensor(math.pi, device=device, dtype=dtype)
    half_pi_t = pi_t / 2.0
    lambda_r_t = torch.tensor(lambda_r, device=device, dtype=dtype)
    eps_t = torch.tensor(1e-6, device=device, dtype=dtype)
    tiny_t = torch.tensor(1e-9, device=device, dtype=dtype)

    w = gt_bboxes[..., 2].clamp_min(eps_t)
    h = gt_bboxes[..., 3].clamp_min(eps_t)
    theta = gt_bboxes[..., 4]

    r = torch.sqrt((w / 2.0) ** 2 + (h / 2.0) ** 2)
    Mr = torch.log(r / lambda_r_t + tiny_t)

    theta_p = theta + half_pi_t
    delta = torch.atan(h / w)

    a1 = theta_p - delta
    a2 = theta_p + delta
    theta_m = theta_p

    a1 = a1.clamp(0.0, pi_t)
    a2 = a2.clamp(0.0, pi_t)
    theta_m = theta_m.clamp(0.0, pi_t)

    a1_sorted = torch.minimum(a1, a2)
    a2_sorted = torch.maximum(a1, a2)

    return torch.stack([Mr, a1_sorted, a2_sorted, theta_m], dim=-1)

def build_rotated_gaussian(h, w, cx, cy, bw, bh, theta, lambda_s=0.001, device=None, dtype=torch.float32):
    device = device if device is not None else cx.device

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij"
    )

    lambda_s_t = torch.tensor(lambda_s, device=device, dtype=dtype)
    eps_t = torch.tensor(1e-6, device=device, dtype=dtype)
    two_t = torch.tensor(2.0, device=device, dtype=dtype)
    half_t = torch.tensor(0.5, device=device, dtype=dtype)

    sigma11 = (bw / two_t) / torch.sqrt(-two_t * torch.log(lambda_s_t))
    sigma22 = (bh / two_t) / torch.sqrt(-two_t * torch.log(lambda_s_t))
    sigma11 = sigma11.clamp_min(eps_t)
    sigma22 = sigma22.clamp_min(eps_t)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    dx = xx - cx
    dy = yy - cy

    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy

    g = torch.exp(-half_t * ((xr / sigma11) ** 2 + (yr / sigma22) ** 2))
    return g

def assign_by_heatmap_threshold(heatmap, thr=0.5):
    return heatmap > thr

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses with polar parameterization."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                pred_polar=None, target_polar=None):
        """IoU loss with polar parameter loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        # Polar parameter loss (if provided)
        if pred_polar is not None and target_polar is not None:
            # pred_polar 可能是 (bs, 4, anchors) 或 (bs, anchors, 4)
            # 先统一转换为 (bs, anchors, 4)
            if pred_polar.dim() == 3:
                if pred_polar.shape[1] == 4:  # 形状为 (bs, 4, anchors)
                    pred_polar = pred_polar.permute(0, 2, 1).contiguous()  # 转为 (bs, anchors, 4)
                    target_polar = target_polar.permute(0, 2, 1).contiguous()  # 同样转换target



            # 展平张量进行索引
            batch_size, num_anchors, _ = pred_polar.shape
            fg_mask_flat = fg_mask.view(-1)
            pred_polar_flat = pred_polar.view(-1, 4)
            target_polar_flat = target_polar.view(-1, 4)

            pred_r = pred_polar_flat[fg_mask_flat, 0:1]  # radius
            pred_a1 = pred_polar_flat[fg_mask_flat, 1:2]  # angle 1
            pred_a2 = pred_polar_flat[fg_mask_flat, 2:3]  # angle 2

            target_r = target_polar_flat[fg_mask_flat, 0:1]
            target_a1 = target_polar_flat[fg_mask_flat, 1:2]
            target_a2 = target_polar_flat[fg_mask_flat, 2:3]

            # r loss: Smooth L1
            loss_r = F.smooth_l1_loss(pred_r, target_r, reduction='none') * weight
            loss_r = loss_r.sum() / target_scores_sum

            # Angle loss
            # 使用 Cosine Similarity 是一种很好的做法，保持不变
            loss_a1 = (1.0 - torch.cos(pred_a1 - target_a1)) * weight
            loss_a2 = (1.0 - torch.cos(pred_a2 - target_a2)) * weight
            loss_angle = (loss_a1.sum() + loss_a2.sum()) / target_scores_sum

            # 【核心修改】调整 Loss 组合权重
            # 1. 降低 loss_r 的权重 (因为它只是辅助，不决定框的大小)
            # 2. 提高 loss_angle 的权重 (角度决定生死)
            # 3. 这里的 2.0 是为了平衡 IoU loss 的量级
            loss_iou = loss_iou + 0.05 * loss_r + 0.5 * loss_angle

            # ------------------ 修改结束 ------------------

        return loss_iou, loss_dfl

class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss:
    """
    SDR-Net style OBB loss:
        L_total = L_heat + lambda_off * L_off + lambda_polar * L_polar
    """

    def __init__(self, model):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]

        self.device = device
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.lambda_r = float(getattr(h, "lambda_r", 12.44))
        self.lambda_s = float(getattr(h, "lambda_s", 0.001))
        self.lambda_th = float(getattr(h, "lambda_th", 0.5))
        self.lambda_o = float(getattr(h, "lambda_o", 2.0))
        self.lambda_off = float(getattr(h, "lambda_off", 1.0))
        self.lambda_polar = float(getattr(h, "lambda_polar", 0.5))
        self.beta = float(getattr(h, "beta_smooth_l1", 0.11))

    def _build_targets_per_level(self, batch, bs, feat_h, feat_w, stride, dtype):
        batch_idx = batch["batch_idx"].view(-1).to(self.device)
        gt_cls = batch["cls"].view(-1).to(self.device).long()
        gt_box = batch["bboxes"].view(-1, 5).to(self.device, dtype=dtype)

        target_hm = torch.zeros(bs, self.nc, feat_h, feat_w, device=self.device, dtype=dtype)
        target_off = torch.zeros(bs, 2, feat_h, feat_w, device=self.device, dtype=dtype)
        target_pol = torch.zeros(bs, 4, feat_h, feat_w, device=self.device, dtype=dtype)
        pos_mask = torch.zeros(bs, 1, feat_h, feat_w, device=self.device, dtype=torch.bool)

        if gt_box.numel() == 0:
            return target_hm, target_off, target_pol, pos_mask

        scale_vec = torch.tensor(
            [feat_w * stride, feat_h * stride, feat_w * stride, feat_h * stride, 1.0],
            device=self.device,
            dtype=dtype
        )
        stride_t = torch.tensor(stride, device=self.device, dtype=dtype)
        lambda_o_t = torch.tensor(self.lambda_o, device=self.device, dtype=dtype)

        for i in range(gt_box.shape[0]):
            b = int(batch_idx[i].item())
            if b < 0 or b >= bs:
                continue

            cls_id = int(gt_cls[i].item())
            if cls_id < 0 or cls_id >= self.nc:
                continue

            box_lvl = gt_box[i] * scale_vec
            cx, cy, bw, bh, theta = box_lvl

            cx_f = cx / stride_t
            cy_f = cy / stride_t
            bw_f = bw / stride_t
            bh_f = bh / stride_t

            g = build_rotated_gaussian(
                feat_h, feat_w,
                cx_f, cy_f, bw_f, bh_f, theta,
                lambda_s=self.lambda_s,
                device=self.device,
                dtype=dtype
            )

            target_hm[b, cls_id] = torch.maximum(target_hm[b, cls_id], g)

            pm = g > self.lambda_th
            if pm.any():
                ys, xs = torch.where(pm)
                ys_d = ys.to(device=self.device, dtype=dtype)
                xs_d = xs.to(device=self.device, dtype=dtype)

                target_off[b, 0, ys, xs] = (cx_f - xs_d) / lambda_o_t
                target_off[b, 1, ys, xs] = (cy_f - ys_d) / lambda_o_t

                polar = obb_to_polar(box_lvl.unsqueeze(0).to(dtype), lambda_r=self.lambda_r)[0].to(dtype)

                target_pol[b, 0, ys, xs] = polar[0]
                target_pol[b, 1, ys, xs] = polar[1]
                target_pol[b, 2, ys, xs] = polar[2]
                target_pol[b, 3, ys, xs] = polar[3]

                pos_mask[b, 0, ys, xs] = True

        return target_hm, target_off, target_pol, pos_mask

    def __call__(self, preds, batch):
        # val/infer stage may pass (decoded, raw)
        if isinstance(preds, tuple):
            preds = preds[1]

        if not isinstance(preds, dict):
            raise TypeError(f"v8OBBLoss expects dict or (decoded, dict), but got {type(preds)}")

        pred_hm = preds["heatmap"]
        pred_off = preds["offset"]
        pred_pol = preds["polar"]

        bs = pred_hm[0].shape[0]
        dtype = torch.float32
        pi_t = torch.tensor(math.pi, device=self.device, dtype=dtype)

        loss_heat = torch.zeros(1, device=self.device, dtype=dtype)
        loss_off = torch.zeros(1, device=self.device, dtype=dtype)
        loss_polar = torch.zeros(1, device=self.device, dtype=dtype)

        for i in range(len(pred_hm)):
            pred_hm_i = pred_hm[i].float()
            pred_off_i = pred_off[i].float()
            pred_pol_i = pred_pol[i].float()

            _, _, h, w = pred_hm_i.shape
            stride = self.stride[i].item() if self.stride[i] > 0 else (2 ** (i + 3))

            target_hm, target_off, target_pol, pos_mask = self._build_targets_per_level(
                batch, bs, h, w, stride, dtype
            )

            pos_expand_hm = pos_mask.expand(-1, self.nc, -1, -1)
            pred_hm_sig = pred_hm_i.sigmoid()

            if pos_expand_hm.any():
                heat_loss_map = smooth_l1_beta_loss(
                    pred_hm_sig[pos_expand_hm],
                    target_hm[pos_expand_hm],
                    beta=self.beta,
                    reduction="none"
                )
                if heat_loss_map.numel():
                    loss_heat += heat_loss_map.mean()

            pos_expand_off = pos_mask.expand(-1, 2, -1, -1)
            if pos_expand_off.any():
                off_loss_map = smooth_l1_beta_loss(
                    pred_off_i[pos_expand_off],
                    target_off[pos_expand_off],
                    beta=self.beta,
                    reduction="none"
                )
                if off_loss_map.numel():
                    loss_off += off_loss_map.mean()

            pos_expand_pol = pos_mask.expand(-1, 4, -1, -1)
            if pos_expand_pol.any():
                pred_pol_pos = pred_pol_i[pos_expand_pol].view(-1, 4)
                tgt_pol_pos = target_pol[pos_expand_pol].view(-1, 4)

                l_radius = smooth_l1_beta_loss(
                    pred_pol_pos[:, 0],
                    tgt_pol_pos[:, 0],
                    beta=self.beta,
                    reduction="mean"
                )

                pred_a1 = pred_pol_pos[:, 1].sigmoid() * pi_t
                pred_a2 = pred_pol_pos[:, 2].sigmoid() * pi_t
                pred_tm = pred_pol_pos[:, 3].sigmoid() * pi_t

                tgt_a1 = tgt_pol_pos[:, 1]
                tgt_a2 = tgt_pol_pos[:, 2]
                tgt_tm = tgt_pol_pos[:, 3]

                l_angle = (1.0 - torch.cos(pred_a1 - tgt_a1)).mean()
                l_angle += (1.0 - torch.cos(pred_a2 - tgt_a2)).mean()
                l_mid = (1.0 - torch.cos(pred_tm - tgt_tm)).mean()

                loss_polar += l_radius + l_angle + l_mid

        nlv = max(len(pred_hm), 1)
        loss_heat = loss_heat / nlv
        loss_off = loss_off / nlv
        loss_polar = loss_polar / nlv

        loss = loss_heat + self.lambda_off * loss_off + self.lambda_polar * loss_polar
        loss_items = torch.cat([loss_heat.detach(), loss_off.detach(), loss_polar.detach()])

        return loss * bs, loss_items


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
