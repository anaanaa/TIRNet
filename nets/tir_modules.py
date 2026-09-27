# -*- coding: utf-8 -*-
"""TIRNet decoder modules: RTMB and CDCB."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RTMB(nn.Module):
    """Retinex-based Text Modulation Block (RTMB)."""

    def __init__(
        self,
        img_channels: int,
        text_dim: int = 512,
        init_beta: float = 1.0,
        tau: float = 0.1,
        rho_max: float = 2.0,
        kappa_max: float = 1.0,
        beta_min: float = 1e-4,
        learnable: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.img_channels = img_channels
        self.text_dim = text_dim
        self.eps = eps
        self.tau = float(tau)
        self.rho_max = float(rho_max)
        self.kappa_max = float(kappa_max)
        self.beta_min = float(beta_min)

        self.text_projector = nn.Linear(text_dim, img_channels)
        self.lpf = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        init_beta = max(float(init_beta) - self.beta_min, 1e-8)
        init_beta_hat = float(torch.log(torch.expm1(torch.tensor(init_beta))).item())

        if learnable:
            self.beta_hat = nn.Parameter(torch.tensor(init_beta_hat))
            self.rho_hat = nn.Parameter(torch.tensor(0.0))
            self.kappa_hat = nn.Parameter(torch.tensor(0.0))
            self.gamma_hat = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("beta_hat", torch.tensor(init_beta_hat), persistent=False)
            self.register_buffer("rho_hat", torch.tensor(0.0), persistent=False)
            self.register_buffer("kappa_hat", torch.tensor(0.0), persistent=False)
            self.register_buffer("gamma_hat", torch.tensor(0.0), persistent=False)

    def forward(self, img_feat: torch.Tensor, text_emb: torch.Tensor):
        B, C, H, W = img_feat.shape

        text_proj = self.text_projector(text_emb).view(B, C, 1, 1)

        img_norm = F.normalize(img_feat, p=2, dim=1, eps=self.eps)
        text_norm = F.normalize(text_proj, p=2, dim=1, eps=self.eps)

        A = (img_norm * text_norm).sum(dim=1, keepdim=True) / max(self.tau, 1e-8)

        beta = F.softplus(self.beta_hat) + self.beta_min
        rho = self.rho_max * torch.sigmoid(self.rho_hat)
        kappa = self.kappa_max * torch.sigmoid(self.kappa_hat)
        gamma = torch.sigmoid(self.gamma_hat)

        L_pos = torch.sigmoid(self.lpf(A))
        L_neg = torch.sigmoid(-beta * A)

        out = img_feat * (1.0 + rho * L_pos) * (1.0 - kappa * L_neg) + gamma * img_feat

        return out, A, L_pos, L_neg


def _pick_gn_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class CDCB(nn.Module):
    """Consistent Detail Compensation Block (CDCB)."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

        self.base_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.base_act = nn.GELU()
        self.base_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

        self.lpf = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.proj_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm_q = nn.GroupNorm(_pick_gn_groups(channels), channels, affine=False)
        self.norm_k = nn.GroupNorm(_pick_gn_groups(channels), channels, affine=False)
        self.delta = nn.Parameter(torch.tensor(1.0))

        self.gate_conv = nn.Sequential(
            nn.Conv2d(2 * channels + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, decoder_feat: torch.Tensor, encoder_feat: torch.Tensor, reliability: torch.Tensor):
        B_s = self.base_conv1(decoder_feat)
        B_s = self.base_act(B_s)
        B_s = self.base_conv2(B_s)

        E_low = self.lpf(encoder_feat)
        D_raw = encoder_feat - E_low

        q = self.norm_q(self.proj_q(B_s))
        k = self.norm_k(self.proj_k(D_raw))
        C_s = (q * k).sum(dim=1, keepdim=True)
        W_cons = torch.sigmoid(self.delta * C_s)
        D_s = W_cons * D_raw

        if reliability.shape[1] != 1:
            reliability = reliability.mean(dim=1, keepdim=True)
        G_s = torch.sigmoid(self.gate_conv(torch.cat([B_s, D_s, reliability], dim=1)))

        out = B_s + G_s * D_s
        return out, W_cons, G_s


class TIRDecoderStage(nn.Module):
    """Decoder stage combining RTMB and CDCB."""

    def __init__(
        self,
        up_in_channels: int,
        skip_in_channels: int,
        out_channels: int,
        text_dim: int = 512,
        upsampling_method: str = "conv_transpose",
    ):
        super().__init__()

        self.up_in_channels = up_in_channels
        self.skip_in_channels = skip_in_channels
        self.out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_in_channels, out_channels,
                kernel_size=2, stride=2
            )
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False),
                nn.Conv2d(up_in_channels, out_channels, kernel_size=1, stride=1),
            )
        else:
            raise ValueError(f"Unsupported upsampling_method: {upsampling_method}")

        self.skip_proj = nn.Conv2d(skip_in_channels, out_channels, kernel_size=1, bias=False)

        self.rtmb = RTMB(
            img_channels=out_channels,
            text_dim=text_dim,
            learnable=True
        )
        self.cdcb = CDCB(channels=out_channels)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor, text_emb: torch.Tensor):
        decoder_feat = self.upsample(x)
        skip_feat = self.skip_proj(skip_x)

        decoder_feat_refined, A, L_pos, L_neg = self.rtmb(
            img_feat=decoder_feat,
            text_emb=text_emb
        )

        output, W_cons, G_s = self.cdcb(decoder_feat_refined, skip_feat, L_pos)

        return output, A, L_pos, L_neg
