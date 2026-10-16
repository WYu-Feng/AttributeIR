from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def zero_init_conv(conv: nn.Conv2d) -> None:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def small_init_conv(conv: nn.Conv2d, std: float = 1e-3) -> None:
    std = 1e-2
    nn.init.normal_(conv.weight, mean=0.0, std=std)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LearnableSoftplusScale(nn.Module):
    def __init__(self, init: float = 0.1):
        super().__init__()
        inv = math.log(math.expm1(init)) if init > 1e-6 else -12.0
        self.raw = nn.Parameter(torch.tensor(inv, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return F.softplus(self.raw)


def target_effective_residual(
    delta_feat: torch.Tensor,
    scale: torch.Tensor,
    target_rms: torch.Tensor,
    max_gain: torch.Tensor,
    max_abs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the actual injected residual plus diagnostics.

    `gain` is diagnostic only: it tells how much amplification would be needed
    to reach the target RMS. We no longer force that gain into the forward path
    because hard residual boosting caused restoration collapse when a branch,
    especially stage-2 corruption, became overactive.
    """
    raw_eff = scale.to(delta_feat) * delta_feat
    rms = torch.sqrt(raw_eff.float().pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-8).to(raw_eff)
    gain = target_rms.to(raw_eff).view(1, 1, 1, 1) / (rms + 1e-6)
    gain = torch.minimum(gain.clamp_min(1.0), max_gain.to(raw_eff).view(1, 1, 1, 1))
    cap = max_abs.to(raw_eff).view(1, 1, 1, 1)
    eff = cap * torch.tanh(raw_eff / (cap + 1e-6))
    eff_rms = torch.sqrt(eff.float().pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-8).to(eff)
    return eff, gain.detach(), eff_rms.detach()


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class ResidualDWBlock(nn.Module):
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)
        self.norm = RMSNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = F.gelu(self.dw(y))
        y = self.conv2(y)
        return self.norm(x + y)


class ConvHeadReadout(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 128, bias: bool = True):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=bias),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(in_channels, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x).flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=1)


class CrossAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=bias)
        self.q_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=bias)
        self.kv_proj = nn.Conv2d(channels, channels * 2, 1, bias=bias)
        self.kv_dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2, bias=bias)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=bias)

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = q_feat.shape
        q = self.q_dw(self.q_proj(q_feat))
        kv = self.kv_dw(self.kv_proj(kv_feat))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.out_proj(out)


class FixedLowPass2d(nn.Module):
    """Parameter-free spatial low-pass filter.

    We use a binomial blur to explicitly suppress grid-like high-frequency
    residuals in the photometric branch.
    """
    def __init__(self, channels: int, passes: int = 2):
        super().__init__()
        kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum()
        weight = kernel_2d[None, None, :, :].repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)
        self.channels = channels
        self.passes = passes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.passes):
            x = F.conv2d(x, self.weight.to(dtype=x.dtype), stride=1, padding=2, groups=self.channels)
        return x


# -----------------------------------------------------------------------------
# Context handling
# -----------------------------------------------------------------------------

class ContextMixin:
    NUM_ATTRS = 8
    BRIGHTNESS = 0
    CONTRAST = 1
    COLOR = 2
    SHARPNESS = 3
    STRUCTURE = 4
    TEXTURE = 5
    NOISE = 6
    ARTIFACT = 7

    @staticmethod
    def rgb_to_luma(img: torch.Tensor) -> torch.Tensor:
        if img.size(1) != 3:
            return img.mean(dim=1, keepdim=True)
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    @staticmethod
    def rgb_to_hsv(rgb: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        maxc, _ = torch.max(rgb, dim=1, keepdim=True)
        minc, _ = torch.min(rgb, dim=1, keepdim=True)
        v = maxc
        delt = maxc - minc
        s = delt / (maxc + eps)

        rc = (maxc - r) / (delt + eps)
        gc = (maxc - g) / (delt + eps)
        bc = (maxc - b) / (delt + eps)
        h = torch.zeros_like(v)
        mask = delt > eps
        h_r = (bc - gc) % 6.0
        h_g = rc - bc + 2.0
        h_b = gc - rc + 4.0
        is_r = (maxc == r) & mask
        is_g = (maxc == g) & mask
        is_b = (maxc == b) & mask
        h = torch.where(is_r, h_r / 6.0, h)
        h = torch.where(is_g, h_g / 6.0, h)
        h = torch.where(is_b, h_b / 6.0, h)
        return torch.cat([h, s, v], dim=1)

    @staticmethod
    def box_filter(x: torch.Tensor, k: int) -> torch.Tensor:
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)

    @classmethod
    def local_std(cls, x: torch.Tensor, k: int, eps: float = 1e-6) -> torch.Tensor:
        mean = cls.box_filter(x, k)
        mean_sq = cls.box_filter(x * x, k)
        var = torch.clamp(mean_sq - mean * mean, min=0.0)
        return torch.sqrt(var + eps)

    @staticmethod
    def sobel_mag(y: torch.Tensor) -> torch.Tensor:
        device, dtype = y.device, y.dtype
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)[None, None]
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=dtype)[None, None]
        gx = F.conv2d(y, kx, padding=1)
        gy = F.conv2d(y, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    @staticmethod
    def laplace_mag(y: torch.Tensor) -> torch.Tensor:
        device, dtype = y.device, y.dtype
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=dtype)[None, None]
        return torch.abs(F.conv2d(y, k, padding=1))

    @staticmethod
    def tenengrad(y: torch.Tensor) -> torch.Tensor:
        g = ContextMixin.sobel_mag(y)
        return g * g

    def parse_context(
        self,
        context,
        b: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context is None:
            attr_tokens = torch.zeros(b, self.NUM_ATTRS, self.ctx_token_dim, device=device, dtype=dtype)
            attr_logits = torch.zeros(b, self.NUM_ATTRS, self.attr_score_dim, device=device, dtype=dtype)
            return attr_tokens, attr_logits
        if isinstance(context, dict):
            if "attr_tokens" in context and "attr_logits" in context:
                return context["attr_tokens"], context["attr_logits"]
            raise ValueError("context dict must contain 'attr_tokens' and 'attr_logits'")
        if torch.is_tensor(context):
            if context.dim() == 3:
                if context.size(0) != b or context.size(1) != self.NUM_ATTRS or context.size(2) != self.token_in_dim:
                    raise ValueError(
                        f"3D context tensor must be [B, {self.NUM_ATTRS}, {self.token_in_dim}], got {tuple(context.shape)}"
                    )
                attr_tokens = context[:, :, : self.ctx_token_dim]
                attr_logits = context[:, :, self.ctx_token_dim :]
                return attr_tokens, attr_logits
            if context.dim() == 2:
                if context.size(0) != b or context.size(1) != self.token_in_dim:
                    raise ValueError(f"2D context tensor must be [B, {self.token_in_dim}], got {tuple(context.shape)}")
                row_token = context[:, : self.ctx_token_dim].unsqueeze(1).repeat(1, self.NUM_ATTRS, 1)
                row_logits = context[:, self.ctx_token_dim :].unsqueeze(1).repeat(1, self.NUM_ATTRS, 1)
                return row_token, row_logits
        raise TypeError("context must be None, Tensor, or Dict")


class AttributeControlEncoder(nn.Module, ContextMixin):
    def __init__(self, token_in_dim: int, attr_score_dim: int, d_token: int = 128, d_ctrl: int = 64, bias: bool = True):
        super().__init__()
        assert token_in_dim > attr_score_dim
        self.token_in_dim = token_in_dim
        self.attr_score_dim = attr_score_dim
        self.ctx_token_dim = token_in_dim - attr_score_dim
        self.d_token = d_token
        self.d_ctrl = d_ctrl

        self.token_proj = nn.Linear(self.ctx_token_dim, d_token, bias=bias)
        self.score_proj = nn.Linear(attr_score_dim, 1, bias=bias)

        in3 = d_token * 3 + 3
        in2 = d_token * 2 + 2
        self.photometric_encoder = TinyMLP(in3, 2 * d_ctrl, d_ctrl, bias=bias)
        self.fidelity_encoder = TinyMLP(in3, 2 * d_ctrl, d_ctrl, bias=bias)
        self.corruption_encoder = TinyMLP(in2, 2 * d_ctrl, d_ctrl, bias=bias)
        self.global_encoder = TinyMLP(d_ctrl * 3, 2 * d_ctrl, d_ctrl, bias=bias)

    def forward(self, context, b: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        attr_tokens, attr_logits = self.parse_context(context, b, device, dtype)
        z = self.token_proj(attr_tokens)
        s = torch.sigmoid(self.score_proj(attr_logits))
        q = 2.0 * s - 1.0

        def pack(indices: Sequence[int]) -> torch.Tensor:
            z_sel = z[:, indices, :]
            s_sel = q[:, indices, :]
            return torch.cat([z_sel.flatten(1), s_sel.flatten(1)], dim=1)

        photometric_ctrl = self.photometric_encoder(pack([self.BRIGHTNESS, self.CONTRAST, self.COLOR]))
        fidelity_ctrl = self.fidelity_encoder(pack([self.SHARPNESS, self.STRUCTURE, self.TEXTURE]))
        corruption_ctrl = self.corruption_encoder(pack([self.NOISE, self.ARTIFACT]))
        global_ctrl = self.global_encoder(torch.cat([photometric_ctrl, fidelity_ctrl, corruption_ctrl], dim=1))

        return {
            "attr_tokens_proj": z,
            "attr_scores": q,
            "photometric_ctrl": photometric_ctrl,
            "fidelity_ctrl": fidelity_ctrl,
            "corruption_ctrl": corruption_ctrl,
            "global_ctrl": global_ctrl,
        }


def _compress_group_target(vec: torch.Tensor, out_dim: int = 128) -> torch.Tensor:
    vec = F.adaptive_avg_pool1d(vec.unsqueeze(1), out_dim).squeeze(1)
    return F.normalize(vec, dim=1)


def build_group_targets(
    attr_tokens: torch.Tensor,
    attr_logits: torch.Tensor,
    indices: Sequence[int],
    out_dim: int = 128,
) -> torch.Tensor:
    group_vec = torch.cat([attr_tokens[:, indices, :].flatten(1), attr_logits[:, indices, :].flatten(1)], dim=1)
    return _compress_group_target(group_vec, out_dim=out_dim)


# -----------------------------------------------------------------------------
# Subspace adapters
# -----------------------------------------------------------------------------

class AAMPreprocessor(nn.Module):
    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        self.norm = RMSNorm2d(channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class PhotometricSubspaceAdapter(nn.Module, ContextMixin):
    """Photometric branch with explicitly low-frequency residual output and
    image-level photometric map supervision.

    Design goals:
      1) keep the branch active but unlikely to dominate every stage;
      2) force it to behave like illumination/contrast/color correction instead
         of arbitrary dense texture editing.
    """
    def __init__(self, channels: int, hidden_channels: int, d_ctrl: int, target_dim: int = 128, bias: bool = False, stage_id: int = 0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.stage_id = stage_id
        scale_caps = {0: 0.20, 1: 0.23, 2: 0.26}
        self.register_buffer('scale_cap', torch.tensor(scale_caps.get(stage_id, 0.12), dtype=torch.float32))
        eff_targets = {0: 0.045, 1: 0.050, 2: 0.055}
        self.register_buffer('eff_target', torch.tensor(eff_targets.get(stage_id, 0.050), dtype=torch.float32), persistent=False)
        self.register_buffer('eff_max_gain', torch.tensor({0: 16.0, 1: 16.0, 2: 24.0}.get(stage_id, 16.0), dtype=torch.float32), persistent=False)
        self.register_buffer('eff_max_abs', torch.tensor(0.12, dtype=torch.float32), persistent=False)
        self.in_proj = nn.Conv2d(channels + 7, hidden_channels, 3, padding=1, bias=bias)
        self.pre = ResidualDWBlock(hidden_channels, bias=bias)
        self.affine = TinyMLP(d_ctrl, 2 * d_ctrl, 2 * hidden_channels, bias=True)
        self.post = ResidualDWBlock(hidden_channels, bias=bias)
        self.lowres_proj = nn.Conv2d(hidden_channels, channels, 1, bias=bias)
        small_init_conv(self.lowres_proj, std=1e-3)
        self.delta_blur = FixedLowPass2d(channels, passes=2)
        self.scale = LearnableSoftplusScale(0.09)
        self.readout = ConvHeadReadout(hidden_channels, out_dim=target_dim)
        self.illum_head = nn.Conv2d(hidden_channels, 1, 1, bias=True)
        self.contrast_head = nn.Conv2d(hidden_channels, 1, 1, bias=True)
        self.chroma_head = nn.Conv2d(hidden_channels, 3, 1, bias=True)

    def forward(self, x: torch.Tensor, fmid: torch.Tensor, guide_img: torch.Tensor, ctrl: Dict[str, torch.Tensor], targets: Dict[str, Optional[torch.Tensor]]):
        b, _, h, w = x.shape
        guide = F.interpolate(guide_img, size=(h, w), mode="bilinear", align_corners=False)
        guide = torch.clamp((guide + 1.0) / 2.0, 0.0, 1.0) if guide.min() < 0 else torch.clamp(guide, 0.0, 1.0)

        luma = self.rgb_to_luma(guide)
        low_rgb = self.box_filter(guide, 7)
        chroma = guide - low_rgb
        priors = torch.cat([luma, chroma, low_rgb], dim=1)

        z = self.pre(self.in_proj(torch.cat([fmid, priors], dim=1)))
        gamma_raw, beta_raw = self.affine(ctrl["photometric_ctrl"]).chunk(2, dim=1)
        gamma = 1.0 + 0.25 * torch.tanh(gamma_raw).view(b, self.hidden_channels, 1, 1)
        beta = 0.10 * torch.tanh(beta_raw).view(b, self.hidden_channels, 1, 1)
        z_hat = gamma * z + beta
        z_post = self.post(z_hat)

        z_low = F.avg_pool2d(z_post, kernel_size=2, stride=2)
        delta_low = self.lowres_proj(z_low)
        delta_raw = F.interpolate(delta_low, size=(h, w), mode="bilinear", align_corners=False)
        delta_base = self.delta_blur(delta_raw)
        scale = torch.clamp(self.scale(), max=self.scale_cap.to(x))
        delta_feat, eff_gain, eff_rms = target_effective_residual(
            delta_base,
            scale,
            self.eff_target,
            self.eff_max_gain,
            self.eff_max_abs,
        )
        out = x + delta_feat

        hsv_v = self.rgb_to_hsv(guide)[:, 2:3]
        contrast_proxy = (hsv_v - self.box_filter(hsv_v, 7)) / (self.local_std(hsv_v, 7) + 1e-6)
        hf_residual = delta_raw - delta_base
        delta_map = delta_feat.abs().mean(dim=1, keepdim=True)

        regs = {
            "scale": scale,
            "delta_feat": delta_feat,
            "delta_base": delta_base,
            "delta_raw": delta_raw,
            "delta_low": delta_low,
            "delta_hf_residual": hf_residual,
            "delta_map": delta_map,
            "eff_gain": eff_gain,
            "eff_rms": eff_rms,
            "state_pre": self.readout(z),
            "state_post": self.readout(z_post),
            "group_target_d": targets["photometric_d"],
            "group_target_h": targets["photometric_h"],
            "illum_pred": torch.sigmoid(self.illum_head(z_post)),
            "contrast_pred": self.contrast_head(z_post),
            "chroma_pred": torch.sigmoid(self.chroma_head(z_post)),
            "guide_rgb": guide,
            "guide_luma": luma,
            "guide_contrast": contrast_proxy,
            "alpha_b": beta.mean(dim=1, keepdim=True),
            "alpha_c": (gamma - 1.0).mean(dim=1, keepdim=True),
            "alpha_col": chroma.abs().mean(dim=1, keepdim=True),
            "lowpass_strength": delta_feat.detach().abs().mean(),
        }
        return out, regs


class FidelitySubspaceAdapter(nn.Module, ContextMixin):
    """Fidelity branch with band-pass residual and structure-aware masking."""
    def __init__(self, channels: int, hidden_channels: int, d_ctrl: int, target_dim: int = 128, bias: bool = False, stage_id: int = 0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.stage_id = stage_id
        mask_floor = {0: 0.18, 1: 0.22, 2: 0.26}
        boost = {0: 1.35, 1: 1.55, 2: 1.70}
        scale_caps = {0: 0.24, 1: 0.27, 2: 0.30}
        self.register_buffer('mask_floor', torch.tensor(mask_floor.get(stage_id, 0.18), dtype=torch.float32))
        self.register_buffer('delta_boost', torch.tensor(boost.get(stage_id, 1.3), dtype=torch.float32))
        self.register_buffer('scale_cap', torch.tensor(scale_caps.get(stage_id, 0.15), dtype=torch.float32))
        eff_targets = {0: 0.030, 1: 0.035, 2: 0.040}
        self.register_buffer('eff_target', torch.tensor(eff_targets.get(stage_id, 0.035), dtype=torch.float32), persistent=False)
        self.register_buffer('eff_max_gain', torch.tensor(32.0, dtype=torch.float32), persistent=False)
        self.register_buffer('eff_max_abs', torch.tensor(0.10, dtype=torch.float32), persistent=False)
        self.in_proj = nn.Conv2d(channels + 3, hidden_channels, 3, padding=1, bias=bias)
        self.pre = ResidualDWBlock(hidden_channels, bias=bias)
        self.enhance = TinyMLP(d_ctrl, 2 * d_ctrl, hidden_channels, bias=True)
        self.cross_attn = CrossAttention2d(hidden_channels, num_heads=4, bias=bias)
        self.post = ResidualDWBlock(hidden_channels, bias=bias)
        self.out_proj = nn.Conv2d(hidden_channels, channels, 1, bias=bias)
        small_init_conv(self.out_proj, std=3e-3)
        self.hp_filter = FixedLowPass2d(channels, passes=1)
        self.scale = LearnableSoftplusScale(0.14)
        self.readout = ConvHeadReadout(hidden_channels, out_dim=target_dim)
        self.map_head = nn.Conv2d(hidden_channels, 4, 1, bias=True)

    def forward(self, x: torch.Tensor, fmid: torch.Tensor, guide_img: torch.Tensor, ctrl: Dict[str, torch.Tensor], targets: Dict[str, Optional[torch.Tensor]]):
        _, _, h, w = x.shape
        guide = F.interpolate(guide_img, size=(h, w), mode="bilinear", align_corners=False)
        guide = torch.clamp((guide + 1.0) / 2.0, 0.0, 1.0) if guide.min() < 0 else torch.clamp(guide, 0.0, 1.0)

        y = self.rgb_to_luma(guide)
        edge = self.sobel_mag(y)
        edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
        energy = self.local_std(y, 7)
        energy = energy / (energy.amax(dim=(2, 3), keepdim=True) + 1e-6)
        tene = self.tenengrad(y)
        tene = tene / (tene.amax(dim=(2, 3), keepdim=True) + 1e-6)

        z = self.pre(self.in_proj(torch.cat([fmid, edge, energy, tene], dim=1)))
        e_fid = torch.sigmoid(self.enhance(ctrl["fidelity_ctrl"])).view(x.size(0), self.hidden_channels, 1, 1)
        r_fid = z * (1.0 + e_fid)
        z_att = self.cross_attn(z, r_fid)
        z_post = self.post(z_att)

        maps = torch.sigmoid(self.map_head(z_post))
        fidelity_mask_raw = torch.clamp(0.55 * maps[:, 0:1] + 0.25 * maps[:, 1:2] + 0.12 * maps[:, 2:3] + 0.08 * maps[:, 3:4], 0.0, 1.0)
        mask_floor = self.mask_floor.to(x)
        fidelity_mask = mask_floor + (1.0 - mask_floor) * fidelity_mask_raw
        delta_raw = self.out_proj(z_post)
        delta_low = self.hp_filter(delta_raw)
        delta_hp = delta_raw - delta_low
        delta_base = self.delta_boost.to(x) * fidelity_mask * delta_hp
        scale = torch.clamp(self.scale(), max=self.scale_cap.to(x))
        delta_feat, eff_gain, eff_rms = target_effective_residual(
            delta_base,
            scale,
            self.eff_target,
            self.eff_max_gain,
            self.eff_max_abs,
        )
        out = x + delta_feat

        regs = {
            "scale": scale,
            "delta_feat": delta_feat,
            "delta_base": delta_base,
            "delta_raw": delta_raw,
            "delta_low": delta_low,
            "delta_hp": delta_hp,
            "delta_map": delta_feat.abs().mean(dim=1, keepdim=True),
            "eff_gain": eff_gain,
            "eff_rms": eff_rms,
            "fidelity_mask": fidelity_mask,
            "state_pre": self.readout(z),
            "state_post": self.readout(z_post),
            "group_target_d": targets["fidelity_d"],
            "group_target_h": targets["fidelity_h"],
            "content_map_pred": maps[:, 0:1],
            "structure_map_pred": maps[:, 1:2],
            "texture_map_pred": maps[:, 2:3],
            "sharpness_map_pred": maps[:, 3:4],
            "structure_proxy": edge,
            "texture_proxy": energy,
            "sharpness_proxy": tene,
        }
        return out, regs


class CorruptionSubspaceAdapter(nn.Module, ContextMixin):
    """Corruption branch with explicit localization to noisy / artifact regions."""
    def __init__(self, channels: int, hidden_channels: int, d_ctrl: int, target_dim: int = 128, bias: bool = False, stage_id: int = 0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.stage_id = stage_id
        mask_floor = {0: 0.05, 1: 0.06, 2: 0.08}
        scale_caps = {0: 0.22, 1: 0.25, 2: 0.28}
        mask_caps = {0: 0.70, 1: 0.76, 2: 0.82}
        self.register_buffer('mask_floor', torch.tensor(mask_floor.get(stage_id, 0.06), dtype=torch.float32))
        self.register_buffer('mask_cap', torch.tensor(mask_caps.get(stage_id, 0.76), dtype=torch.float32))
        self.register_buffer('scale_cap', torch.tensor(scale_caps.get(stage_id, 0.13), dtype=torch.float32))
        eff_targets = {0: 0.020, 1: 0.025, 2: 0.030}
        self.register_buffer('eff_target', torch.tensor(eff_targets.get(stage_id, 0.025), dtype=torch.float32), persistent=False)
        self.register_buffer('eff_max_gain', torch.tensor(16.0, dtype=torch.float32), persistent=False)
        self.register_buffer('eff_max_abs', torch.tensor(0.075, dtype=torch.float32), persistent=False)
        self.in_proj = nn.Conv2d(channels + 2, hidden_channels, 3, padding=1, bias=bias)
        self.pre = ResidualDWBlock(hidden_channels, bias=bias)
        self.cond = TinyMLP(d_ctrl, 2 * d_ctrl, hidden_channels, bias=True)
        self.prior_fuse = nn.Sequential(
            nn.Conv2d(2, hidden_channels, 3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=bias),
        )
        self.mask_head = nn.Conv2d(hidden_channels, 1, 1, bias=True)
        self.suppress = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=bias),
        )
        self.post = ResidualDWBlock(hidden_channels, bias=bias)
        self.out_proj = nn.Conv2d(hidden_channels, channels, 1, bias=bias)
        small_init_conv(self.out_proj, std=1.8e-3)
        self.scale = LearnableSoftplusScale(0.12)
        self.readout = ConvHeadReadout(hidden_channels, out_dim=target_dim)

    def forward(self, x: torch.Tensor, fmid: torch.Tensor, guide_img: torch.Tensor, ctrl: Dict[str, torch.Tensor], targets: Dict[str, Optional[torch.Tensor]]):
        _, _, h, w = x.shape
        guide = F.interpolate(guide_img, size=(h, w), mode="bilinear", align_corners=False)
        guide = torch.clamp((guide + 1.0) / 2.0, 0.0, 1.0) if guide.min() < 0 else torch.clamp(guide, 0.0, 1.0)

        y = self.rgb_to_luma(guide)
        low = self.box_filter(y, 3)
        hf = y - low
        p_var = self.local_std(hf, 5)
        p_var = p_var / (p_var.amax(dim=(2, 3), keepdim=True) + 1e-6)
        p_irr = torch.abs(hf - self.box_filter(hf, 3))
        p_irr = p_irr / (p_irr.amax(dim=(2, 3), keepdim=True) + 1e-6)

        z = self.pre(self.in_proj(torch.cat([fmid, p_var, p_irr], dim=1)))
        e_cor = torch.sigmoid(self.cond(ctrl["corruption_ctrl"])).view(x.size(0), self.hidden_channels, 1, 1)
        r_cor = self.prior_fuse(torch.cat([p_var, p_irr], dim=1)) * (1.0 + e_cor)
        m_cor_raw = torch.sigmoid(self.mask_head(r_cor))
        mask_floor = self.mask_floor.to(x)
        m_cor = mask_floor + (1.0 - mask_floor) * m_cor_raw
        m_cor = torch.clamp(m_cor, max=self.mask_cap.to(x))
        z_hat = z - m_cor * self.suppress(z)
        z_post = self.post(z_hat)
        delta_raw = self.out_proj(z_post)
        delta_base = m_cor * delta_raw
        scale = torch.clamp(self.scale(), max=self.scale_cap.to(x))
        delta_feat, eff_gain, eff_rms = target_effective_residual(
            delta_base,
            scale,
            self.eff_target,
            self.eff_max_gain,
            self.eff_max_abs,
        )
        out = x + delta_feat

        regs = {
            "scale": scale,
            "delta_feat": delta_feat,
            "delta_base": delta_base,
            "delta_raw": delta_raw,
            "delta_map": delta_feat.abs().mean(dim=1, keepdim=True),
            "eff_gain": eff_gain,
            "eff_rms": eff_rms,
            "state_pre": self.readout(z),
            "state_post": self.readout(z_post),
            "group_target_d": targets["corruption_d"],
            "group_target_h": targets["corruption_h"],
            "pollution_map_pred": m_cor,
            "noise_map_pred": m_cor * p_var,
            "artifact_map_pred": m_cor * p_irr,
            "noise_proxy": p_var,
            "artifact_proxy": p_irr,
            "guide_gray": y,
        }
        return out, regs


class AdaptiveBranchFusion(nn.Module):
    """Fusion with stage-aware expected usage and non-zero minimum contribution.

    The gate is still sample-adaptive, but its optimization is regularized only
    at the batch expectation level. This matches the intended behavior: we do
    not force every sample to use all three adapters equally, while preventing
    long-term expert starvation.
    """
    def __init__(
        self,
        channels: int,
        d_ctrl: int,
        bias: bool = False,
        min_branch_weight: float = 0.04,
        stage_id: int = 0,
    ):
        super().__init__()
        self.min_branch_weight = min_branch_weight
        self.stage_id = stage_id
        self.ctrl_norm = nn.LayerNorm(4 * d_ctrl + 3)
        self.branch_gate = TinyMLP(4 * d_ctrl + 3, 2 * d_ctrl, 3, bias=True)
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=max(channels // 8, 1), bias=bias),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=bias),
        )
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=bias)
        small_init_conv(self.out_proj, std=2e-3)
        self.scale = LearnableSoftplusScale(0.12)
        self.temperature = nn.Parameter(torch.tensor(1.4, dtype=torch.float32))
        # small initial spatial-fusion contribution
        self.mix_alpha_raw = nn.Parameter(torch.tensor(-1.7, dtype=torch.float32))
        self.stage_bias = nn.Parameter(torch.zeros(1, 3))
        stage_priors = {
            0: torch.tensor([0.22, 0.46, 0.32], dtype=torch.float32),
            1: torch.tensor([0.34, 0.38, 0.28], dtype=torch.float32),
            2: torch.tensor([0.46, 0.31, 0.23], dtype=torch.float32),
        }
        fuse_caps = {0: 1.35, 1: 1.05, 2: 0.90}
        self.register_buffer('target_usage', stage_priors.get(stage_id, torch.tensor([1/3, 1/3, 1/3], dtype=torch.float32)).view(1, 3))
        self.register_buffer('fuse_cap', torch.tensor(fuse_caps.get(stage_id, 0.75), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        d_pho: torch.Tensor,
        d_fid: torch.Tensor,
        d_cor: torch.Tensor,
        ctrl_pack: torch.Tensor,
        branch_stat_pack: torch.Tensor,
    ):
        gate_in = torch.cat([ctrl_pack, branch_stat_pack], dim=1)
        logits = self.branch_gate(self.ctrl_norm(gate_in)) + self.stage_bias
        temp = self.temperature.clamp(min=0.7, max=3.0)
        base_w = torch.softmax(logits / temp, dim=1)
        # Mild stage prior mixing helps avoid pathological saturation without
        # forcing per-sample equal usage.
        prior_mix = 0.06
        mixed_w = (1.0 - prior_mix) * base_w + prior_mix * self.target_usage.to(base_w)
        floor = self.min_branch_weight
        weights = floor + (1.0 - 3.0 * floor) * mixed_w

        d_weighted = (
            weights[:, 0:1].view(-1, 1, 1, 1) * d_pho
            + weights[:, 1:2].view(-1, 1, 1, 1) * d_fid
            + weights[:, 2:3].view(-1, 1, 1, 1) * d_cor
        )
        # Normalize branch residuals before spatial competition to reduce pure
        # scale dominance while preserving their spatial structure.
        def _rms_norm_feat(z: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(z.pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
            return z / rms
        d_spatial = self.spatial_mix(torch.cat([
            _rms_norm_feat(d_pho), _rms_norm_feat(d_fid), _rms_norm_feat(d_cor)
        ], dim=1))
        mix_alpha = torch.sigmoid(self.mix_alpha_raw)
        d_mix = d_weighted + mix_alpha * d_spatial
        fuse_raw = self.out_proj(d_mix)
        branch_anchor = branch_stat_pack.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1).detach().clamp_min(5e-3)
        d_fuse = self.fuse_cap.to(x) * branch_anchor * torch.tanh(fuse_raw / (0.65 * branch_anchor + 1e-6))
        out = x + self.scale() * d_fuse
        regs = {
            'branch_weights': weights,
            'raw_branch_weights': base_w,
            'batch_usage': weights.mean(dim=0, keepdim=True),
            'target_usage': self.target_usage,
            'd_weighted': d_weighted,
            'd_spatial': d_spatial,
            'd_fuse': d_fuse,
            'd_pho': d_pho,
            'd_fid': d_fid,
            'd_cor': d_cor,
            'scale': self.scale(),
            'branch_anchor': branch_anchor,
            'mix_alpha': mix_alpha.detach(),
        }
        return out, regs


class SerialDecoderAttrAdjustmentStage(nn.Module):
    def __init__(
        self,
        channels: int,
        token_in_dim: int = 4096 + 8,
        attr_score_dim: int = 8,
        d_token: int = 128,
        d_ctrl: int = 64,
        hidden_channels: Optional[int] = None,
        target_dim: int = 128,
        bias: bool = False,
        stage_id: int = 0,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(channels // 2, 24)
        self.control_encoder = AttributeControlEncoder(token_in_dim, attr_score_dim, d_token=d_token, d_ctrl=d_ctrl, bias=True)
        self.preprocess = AAMPreprocessor(channels, bias=bias)
        self.photometric_adapter = PhotometricSubspaceAdapter(channels, hidden_channels, d_ctrl=d_ctrl, target_dim=target_dim, bias=bias, stage_id=stage_id)
        self.fidelity_adapter = FidelitySubspaceAdapter(channels, hidden_channels, d_ctrl=d_ctrl, target_dim=target_dim, bias=bias, stage_id=stage_id)
        self.corruption_adapter = CorruptionSubspaceAdapter(channels, hidden_channels, d_ctrl=d_ctrl, target_dim=target_dim, bias=bias, stage_id=stage_id)
        self.fusion = AdaptiveBranchFusion(channels, d_ctrl=d_ctrl, bias=bias, min_branch_weight=0.04, stage_id=stage_id)

    def _build_targets(self, context, clean_context, b: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Optional[torch.Tensor]]:
        attr_tokens_d, attr_logits_d = self.control_encoder.parse_context(context, b, device, dtype)
        targets: Dict[str, Optional[torch.Tensor]] = {
            'photometric_d': build_group_targets(attr_tokens_d, attr_logits_d, [ContextMixin.BRIGHTNESS, ContextMixin.CONTRAST, ContextMixin.COLOR]),
            'fidelity_d': build_group_targets(attr_tokens_d, attr_logits_d, [ContextMixin.SHARPNESS, ContextMixin.STRUCTURE, ContextMixin.TEXTURE]),
            'corruption_d': build_group_targets(attr_tokens_d, attr_logits_d, [ContextMixin.NOISE, ContextMixin.ARTIFACT]),
            'photometric_h': None,
            'fidelity_h': None,
            'corruption_h': None,
        }
        if clean_context is None:
            return targets
        attr_tokens_h, attr_logits_h = self.control_encoder.parse_context(clean_context, b, device, dtype)
        targets['photometric_h'] = build_group_targets(attr_tokens_h, attr_logits_h, [ContextMixin.BRIGHTNESS, ContextMixin.CONTRAST, ContextMixin.COLOR])
        targets['fidelity_h'] = build_group_targets(attr_tokens_h, attr_logits_h, [ContextMixin.SHARPNESS, ContextMixin.STRUCTURE, ContextMixin.TEXTURE])
        targets['corruption_h'] = build_group_targets(attr_tokens_h, attr_logits_h, [ContextMixin.NOISE, ContextMixin.ARTIFACT])
        return targets

    def forward(
        self,
        x: torch.Tensor,
        guide_img: torch.Tensor,
        context=None,
        clean_context=None,
        clean_guide_img: Optional[torch.Tensor] = None,
        steps: int = 0,
    ):
        del steps
        b = x.size(0)
        ctrl = self.control_encoder(context, b, x.device, x.dtype)
        targets = self._build_targets(context, clean_context, b, x.device, x.dtype)
        fmid = self.preprocess(x)

        x_pho, regs_pho = self.photometric_adapter(x, fmid, guide_img, ctrl, targets)
        x_fid, regs_fid = self.fidelity_adapter(x, fmid, guide_img, ctrl, targets)
        x_cor, regs_cor = self.corruption_adapter(x, fmid, guide_img, ctrl, targets)

        d_pho = x_pho - x
        d_fid = x_fid - x
        d_cor = x_cor - x

        ctrl_pack = torch.cat([
            ctrl['photometric_ctrl'],
            ctrl['fidelity_ctrl'],
            ctrl['corruption_ctrl'],
            ctrl['global_ctrl'],
        ], dim=1)
        branch_stat_pack = torch.cat([
            regs_pho['delta_map'].mean(dim=(2, 3)),
            regs_fid['delta_map'].mean(dim=(2, 3)),
            regs_cor['delta_map'].mean(dim=(2, 3)),
        ], dim=1)
        out, regs_fuse = self.fusion(x, d_pho, d_fid, d_cor, ctrl_pack, branch_stat_pack)

        regs_pho['clean_guide_img'] = clean_guide_img
        regs_fid['clean_guide_img'] = clean_guide_img
        regs_cor['clean_guide_img'] = clean_guide_img
        regs = {
            'photometric': regs_pho,
            'structure_texture': regs_fid,
            'frequency_detail': regs_cor,
            'parallel_fuse': regs_fuse,
            'control': {'attr_scores': ctrl['attr_scores'].detach()},
        }
        return out, regs


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def _safe_zero(regs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    if isinstance(regs, dict):
        for v in regs.values():
            if torch.is_tensor(v):
                return torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.tensor(0.0)


def _prep_clean(clean_guide_img: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    clean = F.interpolate(clean_guide_img, size=size_hw, mode="bilinear", align_corners=False)
    clean = torch.clamp((clean + 1.0) / 2.0, 0.0, 1.0) if clean.min() < 0 else torch.clamp(clean, 0.0, 1.0)
    return clean


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=1).mean()


def _adapter_transition_loss(regs: Dict[str, torch.Tensor], weight_post: float = 1.0, weight_gap: float = 0.5) -> torch.Tensor:
    if regs is None:
        return _safe_zero(regs)
    t_d = regs.get("group_target_d", None)
    t_h = regs.get("group_target_h", None)
    s_pre = regs.get("state_pre", None)
    s_post = regs.get("state_post", None)
    if any(x is None for x in [t_d, t_h, s_pre, s_post]):
        return _safe_zero(regs)
    loss_post = _cosine_distance(s_pre, t_d) + _cosine_distance(s_post, t_h)
    loss_gap = _cosine_distance(s_post - s_pre, t_h - t_d)
    return weight_post * loss_post + weight_gap * loss_gap


def adapter_activity_band_loss(
    regs: Dict[str, torch.Tensor],
    target: float,
    upper: float,
    lower_weight: float = 12.0,
    upper_weight: float = 60.0,
    weight: float = 1.0,
) -> torch.Tensor:
    """Encourage useful adapter activity without forcing large residuals.

    The forward path already caps the injected residual. This loss only nudges
    the RMS into a healthy band and penalizes overshoot strongly, which is safer
    than multiplying the residual to hit a target RMS directly.
    """
    if regs is None or regs.get("delta_feat", None) is None:
        return _safe_zero(regs)
    delta = regs["delta_feat"].float()
    rms = torch.sqrt(delta.pow(2).mean(dim=(1, 2, 3)) + 1e-8)
    target_t = torch.as_tensor(target, device=rms.device, dtype=rms.dtype)
    upper_t = torch.as_tensor(upper, device=rms.device, dtype=rms.dtype)
    low = F.relu(target_t - rms).pow(2).mean()
    high = F.relu(rms - upper_t).pow(2).mean()
    return weight * (lower_weight * low + upper_weight * high)


def photometric_guide_target_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or "illum_pred" not in regs:
        return _safe_zero(regs)
    if regs.get("clean_guide_img", None) is None:
        return _safe_zero(regs)
    clean = _prep_clean(regs["clean_guide_img"], regs["illum_pred"].shape[-2:])
    y_clean = ContextMixin.rgb_to_luma(clean)
    v_clean = ContextMixin.rgb_to_hsv(clean)[:, 2:3]
    c_clean = (v_clean - ContextMixin.box_filter(v_clean, 7)) / (ContextMixin.local_std(v_clean, 7) + 1e-6)
    loss = (
        F.l1_loss(regs["illum_pred"], y_clean)
        + 0.5 * F.l1_loss(regs["contrast_pred"], c_clean)
        + 0.5 * F.l1_loss(regs["chroma_pred"], clean)
    )
    return weight * loss


def photometric_lowfreq_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or "delta_hf_residual" not in regs:
        return _safe_zero(regs)
    return weight * regs["delta_hf_residual"].abs().mean()


def photometric_delta_alignment_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or regs.get("clean_guide_img", None) is None or regs.get("delta_map", None) is None:
        return _safe_zero(regs)
    clean = _prep_clean(regs["clean_guide_img"], regs["delta_map"].shape[-2:])
    deg = regs["guide_rgb"]
    y_clean = ContextMixin.rgb_to_luma(clean)
    y_deg = ContextMixin.rgb_to_luma(deg)
    low_clean = ContextMixin.box_filter(clean, 9)
    low_deg = ContextMixin.box_filter(deg, 9)
    target = (ContextMixin.box_filter((y_clean - y_deg).abs(), 9)
              + 0.5 * (low_clean - low_deg).abs().mean(dim=1, keepdim=True))
    target = target / (target.amax(dim=(2, 3), keepdim=True) + 1e-6)
    pred = regs["delta_map"] / (regs["delta_map"].amax(dim=(2, 3), keepdim=True) + 1e-6)
    return weight * F.l1_loss(pred, target)


def photometric_excess_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0, ratio_tau: float = 1.25) -> torch.Tensor:
    if regs is None or regs.get('clean_guide_img', None) is None or regs.get('delta_map', None) is None:
        return _safe_zero(regs)
    clean = _prep_clean(regs['clean_guide_img'], regs['delta_map'].shape[-2:])
    deg = regs['guide_rgb']
    y_clean = ContextMixin.rgb_to_luma(clean)
    y_deg = ContextMixin.rgb_to_luma(deg)
    low_clean = ContextMixin.box_filter(clean, 9)
    low_deg = ContextMixin.box_filter(deg, 9)
    target = ContextMixin.box_filter((y_clean - y_deg).abs(), 9) + 0.5 * (low_clean - low_deg).abs().mean(dim=1, keepdim=True)
    target = _norm_map01(target).detach()
    pred = _norm_map01(regs['delta_map'])
    excess = F.relu(pred - ratio_tau * target).mean()
    return weight * excess


def _content_targets(clean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    y = ContextMixin.rgb_to_luma(clean)
    e = ContextMixin.sobel_mag(y)
    e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-6)
    t = ContextMixin.local_std(y, 7)
    t = t / (t.amax(dim=(2, 3), keepdim=True) + 1e-6)
    q = ContextMixin.tenengrad(y)
    q = q / (q.amax(dim=(2, 3), keepdim=True) + 1e-6)
    c = torch.sigmoid((2.0 * e + 0.5 * t + 1.5 * q) / 3.0)
    return e, t, q, c


def structure_guide_target_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or "content_map_pred" not in regs:
        return _safe_zero(regs)
    if regs.get("clean_guide_img", None) is None:
        return _safe_zero(regs)
    clean = _prep_clean(regs["clean_guide_img"], regs["content_map_pred"].shape[-2:])
    e, t, q, c = _content_targets(clean)
    loss = (
        F.l1_loss(regs["content_map_pred"], c)
        + 0.5 * F.l1_loss(regs["structure_map_pred"], c * e)
        + 0.5 * F.l1_loss(regs["texture_map_pred"], c * t)
        + 0.5 * F.l1_loss(regs["sharpness_map_pred"], c * q)
    )
    return weight * loss


def fidelity_bandpass_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or regs.get("delta_low", None) is None:
        return _safe_zero(regs)
    return weight * regs["delta_low"].abs().mean()


def fidelity_localization_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or regs.get("fidelity_mask", None) is None or regs.get("delta_feat", None) is None:
        return _safe_zero(regs)
    mask = regs["fidelity_mask"]
    delta = regs["delta_feat"].abs().mean(dim=1, keepdim=True)
    outside = ((1.0 - mask) * delta).mean()
    inside = (mask * delta).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
    return weight * (outside + 0.05 * F.relu(0.02 - inside).mean())


def fidelity_delta_alignment_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or regs.get("clean_guide_img", None) is None or regs.get("delta_map", None) is None:
        return _safe_zero(regs)
    clean = _prep_clean(regs["clean_guide_img"], regs["delta_map"].shape[-2:])
    _, _, _, c = _content_targets(clean)
    pred = regs["delta_map"] / (regs["delta_map"].amax(dim=(2, 3), keepdim=True) + 1e-6)
    return weight * F.l1_loss(pred, c)


def _pollution_targets(regs: Dict[str, torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    guide = regs.get("guide_gray", None)
    clean_img = regs.get("clean_guide_img", None)
    if guide is None or clean_img is None:
        return None, None
    clean = _prep_clean(clean_img, guide.shape[-2:])
    y_clean = ContextMixin.rgb_to_luma(clean)
    mu_clean = ContextMixin.box_filter(y_clean, 3)
    hf_clean = y_clean - mu_clean
    clean_noise = ContextMixin.local_std(hf_clean, 5)
    clean_noise = clean_noise / (clean_noise.amax(dim=(2, 3), keepdim=True) + 1e-6)
    clean_art = torch.abs(hf_clean - ContextMixin.box_filter(hf_clean, 3))
    clean_art = clean_art / (clean_art.amax(dim=(2, 3), keepdim=True) + 1e-6)
    noise_t = F.relu(regs["noise_proxy"] - clean_noise)
    art_t = F.relu(regs["artifact_proxy"] - clean_art)
    return noise_t, art_t


def frequency_guide_target_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or "noise_map_pred" not in regs:
        return _safe_zero(regs)
    noise_t, art_t = _pollution_targets(regs)
    if noise_t is None or art_t is None:
        return _safe_zero(regs)
    loss = F.l1_loss(regs["noise_map_pred"], noise_t) + 0.5 * F.l1_loss(regs["artifact_map_pred"], art_t)
    return weight * loss


def corruption_localization_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or regs.get("pollution_map_pred", None) is None or regs.get("delta_feat", None) is None:
        return _safe_zero(regs)
    mask = regs["pollution_map_pred"]
    delta = regs["delta_feat"].abs().mean(dim=1, keepdim=True)
    outside = ((1.0 - mask) * delta).mean()
    return weight * outside


def corruption_mask_area_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0, upper: float = 0.68) -> torch.Tensor:
    if regs is None or regs.get("pollution_map_pred", None) is None:
        return _safe_zero(regs)
    mask_mean = regs["pollution_map_pred"].float().mean(dim=(1, 2, 3))
    return weight * F.relu(mask_mean - upper).pow(2).mean()


def corruption_delta_alignment_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None or regs.get("delta_map", None) is None:
        return _safe_zero(regs)
    noise_t, art_t = _pollution_targets(regs)
    if noise_t is None or art_t is None:
        return _safe_zero(regs)
    target = noise_t + 0.5 * art_t
    target = target / (target.amax(dim=(2, 3), keepdim=True) + 1e-6)
    pred = regs["delta_map"] / (regs["delta_map"].amax(dim=(2, 3), keepdim=True) + 1e-6)
    return weight * F.l1_loss(pred, target)


def branch_diversity_loss(regs_fuse: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs_fuse is None:
        return _safe_zero(regs_fuse)
    d_pho = regs_fuse.get('d_pho', None)
    d_fid = regs_fuse.get('d_fid', None)
    d_cor = regs_fuse.get('d_cor', None)
    if any(v is None for v in [d_pho, d_fid, d_cor]):
        return _safe_zero(regs_fuse)

    def _map(x: torch.Tensor) -> torch.Tensor:
        m = x.abs().mean(dim=1, keepdim=True)
        m = m - m.mean(dim=(2, 3), keepdim=True)
        m = m.flatten(1)
        return F.normalize(m, dim=1)

    p, f, c = _map(d_pho), _map(d_fid), _map(d_cor)
    loss = (p * f).sum(dim=1).abs().mean() + (p * c).sum(dim=1).abs().mean() + (f * c).sum(dim=1).abs().mean()
    return weight * loss


def fusion_expected_load_loss(regs_fuse: Dict[str, torch.Tensor], weight: float = 1.0, floor_tau: float = 0.12) -> torch.Tensor:
    """Batch-expected load balancing.

    This does *not* force every sample to use all branches equally. Instead it
    regularizes the expected branch usage over a batch toward a stage-specific
    prior and penalizes long-term starvation.
    """
    if regs_fuse is None or regs_fuse.get('branch_weights', None) is None:
        return _safe_zero(regs_fuse)
    w = regs_fuse['branch_weights']
    w_mean = w.mean(dim=0)
    target = regs_fuse.get('target_usage', None)
    if target is None:
        target = torch.full_like(w_mean, 1.0 / float(w_mean.numel()))
    else:
        target = target.view(-1).to(w_mean)
    kl = (target * (target.clamp_min(1e-8).log() - w_mean.clamp_min(1e-8).log())).sum()
    floor_pen = F.relu(floor_tau - w_mean).pow(2).mean()
    # Keep the *batch* entropy from becoming too small, but do not constrain
    # per-sample entropy.
    batch_entropy = -(w_mean.clamp_min(1e-8) * w_mean.clamp_min(1e-8).log()).sum()
    batch_entropy_floor = 0.82 * math.log(float(w_mean.numel()))
    entropy_pen = F.relu(batch_entropy_floor - batch_entropy)
    return weight * (kl + 0.5 * floor_pen + 0.05 * entropy_pen)


def fusion_capacity_loss(regs_fuse: Dict[str, torch.Tensor], weight: float = 1.0, ratio_tau: float = 1.80) -> torch.Tensor:
    if regs_fuse is None:
        return _safe_zero(regs_fuse)
    d_fuse = regs_fuse.get('d_fuse', None)
    d_pho = regs_fuse.get('d_pho', None)
    d_fid = regs_fuse.get('d_fid', None)
    d_cor = regs_fuse.get('d_cor', None)
    if any(v is None for v in [d_fuse, d_pho, d_fid, d_cor]):
        return _safe_zero(regs_fuse)
    fuse_map = d_fuse.abs().mean(dim=1, keepdim=True)
    ref_map = torch.max(torch.stack([
        d_pho.abs().mean(dim=1, keepdim=True),
        d_fid.abs().mean(dim=1, keepdim=True),
        d_cor.abs().mean(dim=1, keepdim=True),
    ], dim=0), dim=0).values.detach()
    excess = F.relu(fuse_map - ratio_tau * ref_map).mean()
    return weight * excess


def _norm_map01(m: torch.Tensor) -> torch.Tensor:
    return m / (m.amax(dim=(2, 3), keepdim=True) + 1e-6)


def _cos_sim_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = _norm_map01(a).flatten(1)
    bb = _norm_map01(b).flatten(1)
    return F.cosine_similarity(aa, bb, dim=1).mean()


def _branch_target_maps(
    regs_pho: Dict[str, torch.Tensor],
    regs_fid: Dict[str, torch.Tensor],
    regs_cor: Dict[str, torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    clean_img = regs_pho.get('clean_guide_img', None)
    guide_rgb = regs_pho.get('guide_rgb', None)
    if clean_img is None or guide_rgb is None:
        return None, None, None
    clean = _prep_clean(clean_img, guide_rgb.shape[-2:])
    # photometric target: low-frequency luminance / chroma / contrast mismatch
    y_clean = ContextMixin.rgb_to_luma(clean)
    y_deg = ContextMixin.rgb_to_luma(guide_rgb)
    low_clean = ContextMixin.box_filter(clean, 9)
    low_deg = ContextMixin.box_filter(guide_rgb, 9)
    v_clean = ContextMixin.rgb_to_hsv(clean)[:, 2:3]
    c_clean = (v_clean - ContextMixin.box_filter(v_clean, 7)) / (ContextMixin.local_std(v_clean, 7) + 1e-6)
    c_deg = regs_pho['guide_contrast']
    t_pho = ContextMixin.box_filter((y_clean - y_deg).abs(), 9)         + 0.50 * (low_clean - low_deg).abs().mean(dim=1, keepdim=True)         + 0.35 * ContextMixin.box_filter((c_clean - c_deg).abs(), 7)
    t_pho = _norm_map01(t_pho)

    # fidelity target: where clean image has stronger structure / texture / sharpness
    e_c, t_c, q_c, _ = _content_targets(clean)
    e_d = regs_fid['structure_proxy']
    t_d = regs_fid['texture_proxy']
    q_d = regs_fid['sharpness_proxy']
    t_fid = F.relu(e_c - e_d) + 0.75 * F.relu(t_c - t_d) + 0.85 * F.relu(q_c - q_d)
    t_fid = _norm_map01(t_fid)

    noise_t, art_t = _pollution_targets(regs_cor)
    if noise_t is None or art_t is None:
        return t_pho, t_fid, None
    t_cor = _norm_map01(noise_t + 0.60 * art_t)
    return t_pho, t_fid, t_cor


def branch_feature_map_loss(stage_regs: Dict[str, Dict[str, torch.Tensor]], weight: float = 1.0, rank_margin: float = 0.05) -> torch.Tensor:
    if stage_regs is None:
        return torch.tensor(0.0)
    regs_pho = stage_regs.get('photometric', None)
    regs_fid = stage_regs.get('structure_texture', None)
    regs_cor = stage_regs.get('frequency_detail', None)
    if any(v is None for v in [regs_pho, regs_fid, regs_cor]):
        return _safe_zero(regs_pho)
    if any(regs.get('delta_map', None) is None for regs in [regs_pho, regs_fid, regs_cor]):
        return _safe_zero(regs_pho)
    t_pho, t_fid, t_cor = _branch_target_maps(regs_pho, regs_fid, regs_cor)
    if t_pho is None or t_fid is None or t_cor is None:
        return _safe_zero(regs_pho)

    a_pho = _norm_map01(regs_pho['delta_map'])
    a_fid = _norm_map01(regs_fid['delta_map'])
    a_cor = _norm_map01(regs_cor['delta_map'])

    l_align = F.l1_loss(a_pho, t_pho) + F.l1_loss(a_fid, t_fid) + F.l1_loss(a_cor, t_cor)

    sim_pp = _cos_sim_map(a_pho, t_pho)
    sim_pf = _cos_sim_map(a_pho, t_fid)
    sim_pc = _cos_sim_map(a_pho, t_cor)
    sim_ff = _cos_sim_map(a_fid, t_fid)
    sim_fp = _cos_sim_map(a_fid, t_pho)
    sim_fc = _cos_sim_map(a_fid, t_cor)
    sim_cc = _cos_sim_map(a_cor, t_cor)
    sim_cp = _cos_sim_map(a_cor, t_pho)
    sim_cf = _cos_sim_map(a_cor, t_fid)

    l_rank = F.relu(rank_margin + sim_pf - sim_pp) + F.relu(rank_margin + sim_pc - sim_pp)         + F.relu(rank_margin + sim_fp - sim_ff) + F.relu(rank_margin + sim_fc - sim_ff)         + F.relu(rank_margin + sim_cp - sim_cc) + F.relu(rank_margin + sim_cf - sim_cc)

    return weight * (l_align + 0.5 * l_rank)


def photometric_adapter_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None:
        return _safe_zero(regs)
    loss = photometric_guide_target_loss(regs, weight=1.0)
    loss = loss + 0.50 * _adapter_transition_loss(regs, weight_post=1.0, weight_gap=0.5)
    loss = loss + adapter_activity_band_loss(regs, target=0.050, upper=0.105, lower_weight=12.0, upper_weight=45.0)
    loss = loss + 0.18 * photometric_lowfreq_loss(regs, weight=1.0)
    loss = loss + 0.28 * photometric_delta_alignment_loss(regs, weight=1.0)
    if 'scale' in regs:
        loss = loss + 1e-4 * regs['scale']
    return weight * loss


def fidelity_adapter_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None:
        return _safe_zero(regs)
    loss = structure_guide_target_loss(regs, weight=1.0)
    loss = loss + 0.50 * _adapter_transition_loss(regs, weight_post=1.0, weight_gap=0.5)
    loss = loss + adapter_activity_band_loss(regs, target=0.038, upper=0.085, lower_weight=14.0, upper_weight=55.0)
    loss = loss + 0.22 * fidelity_bandpass_loss(regs, weight=1.0)
    loss = loss + 0.20 * fidelity_localization_loss(regs, weight=1.0)
    loss = loss + 0.24 * fidelity_delta_alignment_loss(regs, weight=1.0)
    if 'scale' in regs:
        loss = loss + 1e-4 * regs['scale']
    return weight * loss


def corruption_adapter_loss(regs: Dict[str, torch.Tensor], weight: float = 1.0) -> torch.Tensor:
    if regs is None:
        return _safe_zero(regs)
    loss = frequency_guide_target_loss(regs, weight=1.0)
    loss = loss + 0.50 * _adapter_transition_loss(regs, weight_post=1.0, weight_gap=0.5)
    loss = loss + adapter_activity_band_loss(regs, target=0.028, upper=0.060, lower_weight=10.0, upper_weight=100.0)
    loss = loss + 0.28 * corruption_localization_loss(regs, weight=1.0)
    loss = loss + 2.0 * corruption_mask_area_loss(regs, weight=1.0, upper=0.68)
    loss = loss + 0.22 * corruption_delta_alignment_loss(regs, weight=1.0)
    if 'scale' in regs:
        loss = loss + 1e-4 * regs['scale']
    return weight * loss


def stage2_delta_balance_loss(
    stage_regs: Dict[str, Dict[str, torch.Tensor]],
    weight: float = 1.0,
    pho_fid_ratio: float = 2.5,
    pho_cor_ratio: float = 2.0,
) -> torch.Tensor:
    """Penalize only excessive stage-2 photometric dominance.

    This does *not* force equality between the three adapters. It only applies
    a hinge loss when the stage-2 photometric residual becomes much larger than
    the fidelity/corruption residuals.
    """
    if stage_regs is None:
        return torch.tensor(0.0)
    regs_pho = stage_regs.get('photometric', None)
    regs_fid = stage_regs.get('structure_texture', None)
    regs_cor = stage_regs.get('frequency_detail', None)
    if any(v is None for v in [regs_pho, regs_fid, regs_cor]):
        return _safe_zero(regs_pho)
    if any(regs.get('delta_map', None) is None for regs in [regs_pho, regs_fid, regs_cor]):
        return _safe_zero(regs_pho)

    pho = regs_pho['delta_map'].mean(dim=(1, 2, 3))
    fid = regs_fid['delta_map'].mean(dim=(1, 2, 3))
    cor = regs_cor['delta_map'].mean(dim=(1, 2, 3))

    loss_pf = F.relu(pho - pho_fid_ratio * fid).mean()
    loss_pc = F.relu(pho - pho_cor_ratio * cor).mean()
    return weight * (loss_pf + loss_pc)


__all__ = [
    'SerialDecoderAttrAdjustmentStage',
    'photometric_adapter_loss',
    'fidelity_adapter_loss',
    'corruption_adapter_loss',
    'photometric_lowfreq_loss',
    'fusion_expected_load_loss',
    'branch_diversity_loss',
    'branch_feature_map_loss',
    'fusion_capacity_loss',
    'stage2_delta_balance_loss',
]
