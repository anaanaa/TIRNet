import numpy as np
import pandas as pd
import torch
from sklearn.metrics import jaccard_score
import cv2
from torch import nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        assert pred.shape == target.shape

        loss = F.binary_cross_entropy(
            pred,
            target,
            reduction="mean"
        )
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, smooth=1e-5):
        batch_size = pred.shape[0]

        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        assert pred.shape == target.shape

        intersection = (pred * target).sum(dim=1)

        denominator = (
            (pred * pred).sum(dim=1)
            + (target * target).sum(dim=1)
        )

        dice = (
            2.0 * intersection + smooth
        ) / (
            denominator + smooth
        )

        loss = 1.0 - dice

        return loss.mean()


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        self.dice_loss = DiceLoss()
        self.bce_loss = BCELoss()

    def _show_dice(self, inputs, targets):
        pred = (inputs >= 0.5).float()
        target = (targets > 0).float()
        hard_dice_coeff = 1.0 - self.dice_loss(pred, target)
        return hard_dice_coeff

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)

        loss = (
            self.dice_weight * dice
            + self.bce_weight * bce
        )

        return loss


def iou_on_batch(masks, pred):
    ious = []
    for i in range(pred.shape[0]):
        pred_tmp = pred[i][0].cpu().detach().numpy()
        mask_tmp = masks[i].cpu().detach().numpy()
        pred_tmp[pred_tmp >= 0.5] = 1
        pred_tmp[pred_tmp < 0.5] = 0
        mask_tmp[mask_tmp > 0] = 1
        mask_tmp[mask_tmp <= 0] = 0
        ious.append(jaccard_score(mask_tmp.reshape(-1), pred_tmp.reshape(-1)))
    return np.mean(ious)


def save_on_batch(images1, masks, pred, names, vis_path):
    for i in range(len(images1)):
        image = images1[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy()
        pred_ = pred[i][0].cpu().detach().numpy()
        pred_[pred_ >= 0.5] = 255
        pred_[pred_ < 0.5] = 0
        mask = np.uint8(mask * 255)
        cv2.imwrite(vis_path + str(names[i]) + '_mask.jpg', mask)
        cv2.imwrite(vis_path + str(names[i]) + '_pred.jpg', pred_)


def read_text(filename):
    df = pd.read_excel(filename)
    text = {}
    for i in df.index.values:
        count = len(df.Description[i].split())
        if count < 9:
            df.Description[i] = df.Description[i] + ' EOF XXX' * (9 - count)
        text[df.Image[i]] = df.Description[i]
    return text


def count_parameters_m(model: nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters())) / 1e6


def _register_flops_hooks(model: nn.Module, flops_accumulator: dict):
    hooks = []

    def add_flops(name: str, flops: int):
        flops_accumulator[name] = flops_accumulator.get(name, 0) + int(flops)

    def conv2d_hook(module: nn.Conv2d, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(output):
            return
        batch = int(x.shape[0])
        out_h = int(output.shape[-2])
        out_w = int(output.shape[-1])
        out_c = int(output.shape[1])
        kernel_h, kernel_w = module.kernel_size
        in_c = int(module.in_channels)
        groups = int(module.groups)
        kernel_mul = kernel_h * kernel_w * (in_c // max(1, groups))
        overall_conv_flops = batch * out_h * out_w * out_c * kernel_mul * 2
        add_flops("conv2d", overall_conv_flops)

    def conv1d_hook(module: nn.Conv1d, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(output):
            return
        batch = int(x.shape[0])
        out_l = int(output.shape[-1])
        out_c = int(output.shape[1])
        kernel = int(module.kernel_size[0])
        in_c = int(module.in_channels)
        groups = int(module.groups)
        kernel_mul = kernel * (in_c // max(1, groups))
        overall_conv_flops = batch * out_l * out_c * kernel_mul * 2
        add_flops("conv1d", overall_conv_flops)

    def linear_hook(module: nn.Linear, inputs, output):
        x = inputs[0]
        if not torch.is_tensor(output):
            return
        in_features = int(module.in_features)
        output_elements = int(output.numel())
        add_flops("linear", output_elements * in_features * 2)

    def multihead_attention_hook(module: nn.MultiheadAttention, inputs, output):
        if len(inputs) < 3:
            return
        query, key, value = inputs[0], inputs[1], inputs[2]
        if not (torch.is_tensor(query) and torch.is_tensor(key) and torch.is_tensor(value)):
            return
        if query.dim() != 3 or key.dim() != 3:
            return
        lq, bsz, embed_dim = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
        lk = int(key.shape[0])
        num_heads = int(module.num_heads)
        head_dim = embed_dim // max(1, num_heads)
        in_proj_flops = (lq + 2 * lk) * bsz * embed_dim * embed_dim * 2
        attn_matmul_flops = bsz * num_heads * lq * lk * head_dim * 4
        add_flops("mha_in_proj", in_proj_flops)
        add_flops("mha_attn", attn_matmul_flops)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv1d_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(multihead_attention_hook))

    return hooks


@torch.no_grad()
def profile_params_flops(model: nn.Module, img_size: int = 224, token_len: int = 18, n_channels: int = 3):
    model = model.eval()
    params_m = count_parameters_m(model)
    flops = {}
    hooks = _register_flops_hooks(model, flops)
    try:
        images = torch.randn(1, n_channels, img_size, img_size)
        text_token = torch.randint(low=0, high=10000, size=(1, token_len), dtype=torch.int64)
        text_mask = torch.ones(1, token_len, dtype=torch.int32)
        _ = model(images, None, text_token, text_mask, mode="test")
    finally:
        for h in hooks:
            h.remove()
    total_flops = sum(flops.values())
    flops_g = float(total_flops) / 1e9
    return params_m, flops_g
