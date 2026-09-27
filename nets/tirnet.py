# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_clip import CLIP, convert_weights, _PT_NAME
from .tir_modules import TIRDecoderStage


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class TIRNet(nn.Module):
    def __init__(self, global_config, config, n_channels=3, n_classes=1, img_size=224):
        super().__init__()
        self.config = config
        self.global_config = global_config
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.loss_weight = global_config.loss_weight
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)

        self.up4 = TIRDecoderStage(up_in_channels=in_channels * 8, skip_in_channels=in_channels * 8,
                                   out_channels=in_channels * 4, text_dim=512, upsampling_method="bilinear")
        self.up3 = TIRDecoderStage(up_in_channels=in_channels * 4, skip_in_channels=in_channels * 4,
                                   out_channels=in_channels * 2, text_dim=512, upsampling_method="bilinear")
        self.up2 = TIRDecoderStage(up_in_channels=in_channels * 2, skip_in_channels=in_channels * 2,
                                   out_channels=in_channels, text_dim=512, upsampling_method="bilinear")
        self.up1 = TIRDecoderStage(up_in_channels=in_channels, skip_in_channels=in_channels,
                                   out_channels=in_channels, text_dim=512, upsampling_method="bilinear")
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.last_activation = nn.Sigmoid()
        self.text_module4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.load_clip(config)

    def load_clip(self, config):
        backbone = config.clip_backbone
        if hasattr(self.global_config, 'clip_pretrained_path'):
            model_path = os.path.join(self.global_config.clip_pretrained_path, _PT_NAME[backbone])
        else:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CLIP pretrained model not found at: {model_path}")

        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        print("use clip version:", model_path)
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)
        print(transformer_width, transformer_heads, transformer_layers)
        if torch.cuda.is_available():
            convert_weights(self.clip)
        self.clip.load_state_dict(state_dict, strict=False)
        self.clip.float()

        if self.config.frozen_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

    def forward(self, images, masks, text_token, text_mask, mode="train"):
        loss_dic = {}
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        cls, text_feat = self.clip.encode_text(text_token, return_hidden=True, mask=text_mask)
        x = images.float()
        x1 = self.inc(x)

        text_feat = self.text_module4(text_feat.float().transpose(1, 2)).transpose(1, 2)
        img_feat1 = x1
        img_feat1 = self.down1(img_feat1)
        img_feat2 = self.down2(img_feat1)
        img_feat3 = self.down3(img_feat2)
        img_feat4 = self.down4(img_feat3)

        text_emb_pooled = text_feat.mean(dim=1)

        x, map_sim4, _, L_neg4 = self.up4(img_feat4, img_feat3, text_emb_pooled)
        x, map_sim3, _, L_neg3 = self.up3(x, img_feat2, text_emb_pooled)
        x, map_sim2, _, L_neg2 = self.up2(x, img_feat1, text_emb_pooled)
        x, map_sim1, _, L_neg1 = self.up1(x, x1, text_emb_pooled)

        if mode != "test" and masks is not None:
            gt = (masks > 0.5).float()
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            eps = 1e-8

            A_list = [map_sim4, map_sim3, map_sim2, map_sim1]
            Lneg_list = [L_neg4, L_neg3, L_neg2, L_neg1]

            loss_rgc_sum = 0.0
            loss_bs_sum = 0.0
            used = 0
            for A_s, L_neg_s in zip(A_list, Lneg_list):
                h, w = A_s.shape[-2], A_s.shape[-1]
                G_s = F.interpolate(gt, size=(h, w), mode="nearest")

                A_safe = torch.clamp(A_s, min=-20.0, max=20.0)
                P = torch.exp(A_safe)

                fg_area = G_s.sum(dim=(2, 3))
                bg_area = (1.0 - G_s).sum(dim=(2, 3))

                cover = (G_s * P).sum(dim=(2, 3)) / (fg_area + eps)
                leak = ((1.0 - G_s) * P).sum(dim=(2, 3)) / (bg_area + eps)

                rgc = -torch.log((cover / (cover + leak + eps)).clamp_min(eps))
                valid = (fg_area > 0.5).float().squeeze(1)
                loss_rgc = (rgc.squeeze(1) * valid).sum() / (valid.sum() + eps)

                loss_bs = F.binary_cross_entropy(L_neg_s, 1.0 - G_s, reduction="mean")

                loss_rgc_sum = loss_rgc_sum + loss_rgc
                loss_bs_sum = loss_bs_sum + loss_bs
                used += 1

            if used > 0:
                loss_dic["loss_rgc"] = loss_rgc_sum / used
                loss_dic["loss_bs"] = loss_bs_sum / used
        if self.n_classes == 1:
            x = self.outc(x)
            logits = self.last_activation(x)
        else:
            logits = self.outc(x)
        if mode == "test":
            return logits, None, None
        return logits, loss_dic
