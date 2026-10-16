from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.paired_dataset import AlignedDataset
from models.attribute_ir import AttributeIR
from models.attribute_adapters import (
    branch_diversity_loss,
    branch_feature_map_loss,
    corruption_adapter_loss,
    fidelity_adapter_loss,
    fusion_capacity_loss,
    fusion_expected_load_loss,
    photometric_adapter_loss,
    stage2_delta_balance_loss,
)
import utils.image_utils as image_utils


LOSS_WEIGHTS = {
    "photometric": 2.3e-2,
    "fidelity": 3.8e-2,
    "corruption": 2.6e-2,
    "fusion_expected_load": 6.0e-3,
    "branch_diversity": 4.5e-3,
    "branch_feature_map": 1.9e-2,
    "fusion_capacity": 1.5e-3,
    "stage2_delta_balance": 5.0e-3,
}

STAGE_LOSS_MULTIPLIERS = {
    0: {"photometric": 1.0, "fidelity": 1.0, "corruption": 1.0},
    1: {"photometric": 1.0, "fidelity": 1.0, "corruption": 1.0},
    2: {"photometric": 0.70, "fidelity": 1.60, "corruption": 1.30},
}


def parse_int_tuple(value: str, expected_len: int, name: str) -> Tuple[int, ...]:
    out = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(out) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated integers")
    return out


def parse_float_tuple(value: str, expected_len: int, name: str) -> Tuple[float, ...]:
    out = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if len(out) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated numbers")
    return out


def parse_task_names(value: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in value.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AttributeIR for all-in-one image restoration.")
    p.add_argument("--dataset-root", type=str, default="./datasets")
    p.add_argument("--context-root", type=str, default="./q_instruct_results")
    p.add_argument("--degraded-context-root", type=str, default="")
    p.add_argument("--clean-context-root", type=str, default="")
    p.add_argument(
        "--online-iqa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Q-Instruct contexts from full source images inside each train/validation batch (default).",
    )
    p.add_argument("--online-iqa-device", type=str, default="cuda:0")
    p.add_argument("--online-iqa-model-path", type=str, default="q-future")
    p.add_argument("--online-iqa-model-base", type=str, default="")
    p.add_argument("--online-iqa-model-name", type=str, default="mplug_owl2")
    p.add_argument(
        "--online-iqa-center-crop-square",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a center crop for Q-Instruct; disabled by default so full images are letterboxed.",
    )
    # p.add_argument("--tasks", type=str, default="derain,dehaze,deblur,light,noise,test2800,hide,alignformer,jpeg_bsd500,uieb")
    p.add_argument("--tasks", type=str, default="derain,dehaze,deblur,light,noise")
    p.add_argument("--eval-tasks", type=str, default="")
    p.add_argument("--normalize-to", choices=["zero1", "minus1_1"], default="minus1_1")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--target-per-task", type=int, default=10000)
    p.add_argument("--eval-center-crop", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--eval-num-workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--backbone-lr-scale", type=float, default=1.0)
    p.add_argument("--aam-lr-scale", type=float, default=1.0)
    p.add_argument("--freeze-backbone-epochs", type=int, default=0)
    p.add_argument("--freeze-aam-epochs", type=int, default=0)
    p.add_argument(
        "--backbone-train-scope",
        choices=["all", "decoder", "decoder_output", "output", "none"],
        default="all",
        help="Limit trainable backbone parameters after the freeze stage to reduce GPU memory.",
    )
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-6)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--warmup-updates", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--attr-loss-warmup-steps", type=int, default=0)
    p.add_argument("--attr-loss-weight-min", type=float, default=0.2)
    p.add_argument("--attr-loss-weight-max", type=float, default=0.2)
    p.add_argument("--photometric-scale-caps", type=str, default="0.20,0.23,0.26")
    p.add_argument("--fidelity-scale-caps", type=str, default="0.24,0.27,0.30")
    p.add_argument("--corruption-scale-caps", type=str, default="0.22,0.25,0.28")
    p.add_argument("--override-scale-caps", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-aam-scale", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--aam-scale-freeze-scope",
        choices=["branch", "all"],
        default="all",
        help="Freeze branch p/f/c scale parameters only, or all AAM scale parameters including fusion.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--skip-nonfinite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stop-on-nonfinite", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--nan-debug-dir", type=str, default="")

    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--resume", type=str, default="./checkpoints/attribute_ir_safe.pth")
    p.add_argument("--resume-training-state", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--max-val-batches", type=int, default=10)
    p.add_argument("--save-val-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-image-dir", type=str, default="")
    p.add_argument("--val-at-epoch-end", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-safe-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-optimizer-state", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cpu-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--empty-cache-after-validation", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--empty-cache-after-checkpoint", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num-blocks", type=str, default="4,6,6,8")
    p.add_argument("--heads", type=str, default="1,2,4,8")
    p.add_argument("--num-refinement-blocks", type=int, default=4)
    p.add_argument("--d-token", type=int, default=256)
    p.add_argument("--d-ctrl", type=int, default=64)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return F.pad(x, (left, right, top, bottom), mode=mode), (left, right, top, bottom)


def unpad(x: torch.Tensor, pads: Tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = pads
    h, w = x.shape[-2:]
    return x[..., top : h - bottom if bottom > 0 else h, left : w - right if right > 0 else w]


def to_zero_one(x: torch.Tensor, normalize_to: str) -> torch.Tensor:
    if normalize_to == "minus1_1":
        return (x + 1.0) * 0.5
    return x


def get_attr_loss_weight(global_step: int, warmup_steps: int, min_weight: float, max_weight: float) -> float:
    if warmup_steps <= 0:
        return max_weight
    ratio = min(float(global_step) / float(warmup_steps), 1.0)
    return min_weight + (max_weight - min_weight) * ratio


def build_scheduler(optimizer: optim.Optimizer, total_epochs: int, warmup_epochs: int, min_lr_ratio: float) -> LambdaLR:
    total_epochs = max(int(total_epochs), 1)
    warmup_epochs = max(int(warmup_epochs), 0)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        denom = max(total_epochs - warmup_epochs, 1)
        progress = min(max(float(epoch - warmup_epochs) / float(denom), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def unpack_batch(batch: Dict[str, Any], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        target = batch["target"].to(device, non_blocking=True)
        input_ = batch["input"].to(device, non_blocking=True)
        degraded_context = batch["degraded_context"].to(device, non_blocking=True)
        clean_context = batch["clean_context"].to(device, non_blocking=True)
        return target, input_, degraded_context, clean_context
    target = batch[0].to(device, non_blocking=True)
    input_ = batch[1].to(device, non_blocking=True)
    degraded_context = batch[2].to(device, non_blocking=True)
    clean_context = batch[3].to(device, non_blocking=True)
    return target, input_, degraded_context, clean_context


def contexts_for_batch(
    batch: Dict[str, Any],
    device: torch.device,
    online_extractor: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move images and obtain contexts either online or from the optional CSV cache.

    Online mode deliberately uses the original image paths in the batch. The
    restoration tensors may be random 128x128 training crops, but Q-Instruct
    sees the complete source and target images, matching the paper's frozen
    IQA stage and avoiding crop-derived attribute labels.
    """
    target = batch["target"].to(device, non_blocking=True)
    input_ = batch["input"].to(device, non_blocking=True)
    if online_extractor is None:
        degraded_context = batch["degraded_context"].to(device, non_blocking=True)
        clean_context = batch["clean_context"].to(device, non_blocking=True)
        return target, input_, degraded_context, clean_context

    degraded_paths = list(batch["degraded_path"])
    target_paths = list(batch["target_path"])
    with torch.inference_mode():
        contexts = online_extractor(degraded_paths + target_paths)
    split = len(degraded_paths)
    degraded_context = contexts[:split].to(device, non_blocking=True)
    clean_context = contexts[split:].to(device, non_blocking=True)
    return target, input_, degraded_context, clean_context


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().item())


def first_nonfinite_tensor_name(obj: Any, prefix: str = "value") -> Optional[str]:
    if torch.is_tensor(obj):
        return None if is_finite_tensor(obj) else prefix
    if isinstance(obj, dict):
        for key, value in obj.items():
            hit = first_nonfinite_tensor_name(value, f"{prefix}.{key}")
            if hit is not None:
                return hit
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            hit = first_nonfinite_tensor_name(value, f"{prefix}[{idx}]")
            if hit is not None:
                return hit
    return None


def batch_is_finite(target: torch.Tensor, input_: torch.Tensor, degraded_context: torch.Tensor, clean_context: torch.Tensor) -> Tuple[bool, str]:
    for name, tensor in (
        ("target", target),
        ("input", input_),
        ("degraded_context", degraded_context),
        ("clean_context", clean_context),
    ):
        if not is_finite_tensor(tensor):
            return False, name
    return True, ""


def compute_stage_attr_loss_detail(stage_regs: Optional[Dict[str, Any]], stage_idx: int) -> Tuple[torch.Tensor, OrderedDict]:
    if stage_regs is None:
        return torch.tensor(0.0), OrderedDict()

    regs_pho = stage_regs.get("photometric")
    regs_fid = stage_regs.get("structure_texture")
    regs_cor = stage_regs.get("frequency_detail")
    regs_fuse = stage_regs.get("parallel_fuse")
    mul = STAGE_LOSS_MULTIPLIERS.get(stage_idx, STAGE_LOSS_MULTIPLIERS[0])

    loss_dict = OrderedDict()
    loss_dict["photometric"] = LOSS_WEIGHTS["photometric"] * mul["photometric"] * photometric_adapter_loss(regs_pho)
    loss_dict["fidelity"] = LOSS_WEIGHTS["fidelity"] * mul["fidelity"] * fidelity_adapter_loss(regs_fid)
    loss_dict["corruption"] = LOSS_WEIGHTS["corruption"] * mul["corruption"] * corruption_adapter_loss(regs_cor)
    loss_dict["fusion_expected_load"] = LOSS_WEIGHTS["fusion_expected_load"] * fusion_expected_load_loss(regs_fuse)
    loss_dict["branch_diversity"] = LOSS_WEIGHTS["branch_diversity"] * branch_diversity_loss(regs_fuse)
    loss_dict["branch_feature_map"] = LOSS_WEIGHTS["branch_feature_map"] * branch_feature_map_loss(stage_regs)
    loss_dict["fusion_capacity"] = LOSS_WEIGHTS["fusion_capacity"] * fusion_capacity_loss(regs_fuse)
    if stage_idx == 2:
        loss_dict["stage2_delta_balance"] = LOSS_WEIGHTS["stage2_delta_balance"] * stage2_delta_balance_loss(stage_regs)
    return sum(loss_dict.values()), loss_dict


def compute_attr_loss_detail(regs_list: Iterable[Optional[Dict[str, Any]]]) -> Tuple[torch.Tensor, OrderedDict]:
    merged = OrderedDict()
    total: Optional[torch.Tensor] = None
    count = 0
    for stage_idx, stage_regs in enumerate(regs_list):
        if stage_regs is None:
            continue
        stage_total, stage_dict = compute_stage_attr_loss_detail(stage_regs, stage_idx)
        total = stage_total if total is None else total + stage_total
        count += 1
        for key, value in stage_dict.items():
            merged[key] = merged.get(key, 0.0) + value

    if total is None or count == 0:
        return torch.tensor(0.0), merged
    total = total / float(count)
    for key in list(merged.keys()):
        merged[key] = merged[key] / float(count)
    return total, merged


def _add_stat_if_exists(dst: OrderedDict, name: str, tensor: Any, mode: str = "mean") -> None:
    if tensor is None or not torch.is_tensor(tensor):
        return
    tensor = tensor.detach()
    if mode == "mean":
        dst[name] = float(tensor.mean().item())
    elif mode == "abs":
        dst[name] = float(tensor.abs().mean().item())
    elif mode == "l2":
        dst[name] = float(torch.sqrt((tensor ** 2).mean()).item())
    elif mode == "std":
        dst[name] = float(tensor.float().std(unbiased=False).item())
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _effective_adapter_delta(adapter_regs: Optional[Dict[str, Any]], fuse_regs: Optional[Dict[str, Any]], fuse_key: str):
    if fuse_regs is not None and torch.is_tensor(fuse_regs.get(fuse_key)):
        return fuse_regs[fuse_key]
    if adapter_regs is None or not torch.is_tensor(adapter_regs.get("delta_feat")):
        return None
    return adapter_regs["delta_feat"]


def _add_state_shift_if_exists(dst: OrderedDict, name: str, regs: Optional[Dict[str, Any]]) -> None:
    if regs is None:
        return
    pre = regs.get("state_pre")
    post = regs.get("state_post")
    if not (torch.is_tensor(pre) and torch.is_tensor(post)):
        return
    if pre.shape != post.shape:
        return
    _add_stat_if_exists(dst, name, post - pre, mode="l2")


def extract_stage_adapter_health(stage_regs: Optional[Dict[str, Any]], stage_name: str = "stage") -> OrderedDict:
    out = OrderedDict()
    if stage_regs is None:
        return out

    regs_pho = stage_regs.get("photometric")
    regs_fid = stage_regs.get("structure_texture")
    regs_cor = stage_regs.get("frequency_detail")
    regs_fuse = stage_regs.get("parallel_fuse")

    d_pho = _effective_adapter_delta(regs_pho, regs_fuse, "d_pho")
    d_fid = _effective_adapter_delta(regs_fid, regs_fuse, "d_fid")
    d_cor = _effective_adapter_delta(regs_cor, regs_fuse, "d_cor")

    if regs_pho is not None:
        _add_stat_if_exists(out, f"{stage_name}_pho_scale", regs_pho.get("scale"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_pho_raw", regs_pho.get("delta_feat"), mode="l2")
        _add_stat_if_exists(out, f"{stage_name}_pho_eff", d_pho, mode="l2")
        _add_stat_if_exists(out, f"{stage_name}_pho_gain", regs_pho.get("eff_gain"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_pho_map", regs_pho.get("delta_map"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_pho_hf", regs_pho.get("delta_hf_residual"), mode="abs")
        _add_state_shift_if_exists(out, f"{stage_name}_pho_state", regs_pho)
    if regs_fid is not None:
        _add_stat_if_exists(out, f"{stage_name}_fid_scale", regs_fid.get("scale"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_fid_raw", regs_fid.get("delta_feat"), mode="l2")
        _add_stat_if_exists(out, f"{stage_name}_fid_eff", d_fid, mode="l2")
        _add_stat_if_exists(out, f"{stage_name}_fid_gain", regs_fid.get("eff_gain"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_fid_map", regs_fid.get("delta_map"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_fid_mask", regs_fid.get("fidelity_mask"), mode="mean")
        _add_state_shift_if_exists(out, f"{stage_name}_fid_state", regs_fid)
    if regs_cor is not None:
        _add_stat_if_exists(out, f"{stage_name}_cor_scale", regs_cor.get("scale"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_cor_raw", regs_cor.get("delta_feat"), mode="l2")
        _add_stat_if_exists(out, f"{stage_name}_cor_eff", d_cor, mode="l2")
        _add_stat_if_exists(out, f"{stage_name}_cor_gain", regs_cor.get("eff_gain"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_cor_map", regs_cor.get("delta_map"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_cor_mask", regs_cor.get("pollution_map_pred"), mode="mean")
        _add_state_shift_if_exists(out, f"{stage_name}_cor_state", regs_cor)
    if regs_fuse is not None:
        _add_stat_if_exists(out, f"{stage_name}_fuse_scale", regs_fuse.get("scale"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_fuse_delta", regs_fuse.get("d_fuse"), mode="l2")
        _add_stat_if_exists(out, f"{stage_name}_fuse_anchor", regs_fuse.get("branch_anchor"), mode="mean")
        _add_stat_if_exists(out, f"{stage_name}_mix_alpha", regs_fuse.get("mix_alpha"), mode="mean")
        branch_weights = regs_fuse.get("branch_weights")
        if torch.is_tensor(branch_weights) and branch_weights.ndim == 2 and branch_weights.size(1) >= 3:
            out[f"{stage_name}_w_pho"] = float(branch_weights[:, 0].mean().item())
            out[f"{stage_name}_w_fid"] = float(branch_weights[:, 1].mean().item())
            out[f"{stage_name}_w_cor"] = float(branch_weights[:, 2].mean().item())
    return out


def extract_model_adapter_health(regs_list: Iterable[Optional[Dict[str, Any]]]) -> OrderedDict:
    all_stats = OrderedDict()
    for idx, stage_regs in enumerate(regs_list):
        for key, value in extract_stage_adapter_health(stage_regs, stage_name=f"s{idx}").items():
            all_stats[key] = value
    return all_stats


def init_running_adapter_health_meter() -> OrderedDict:
    return OrderedDict()


def update_running_adapter_health_meter(adapter_meter: OrderedDict, stats_dict: Dict[str, float]) -> None:
    for key, value in stats_dict.items():
        if value is None:
            continue
        adapter_meter[key] = adapter_meter.get(key, 0.0) + float(value)


def _avg_stat(meter: OrderedDict, interval: int, key: str) -> Optional[float]:
    if key not in meter:
        return None
    return meter[key] / max(interval, 1)


def _fmt_triplet(meter: OrderedDict, interval: int, stage: str, suffix: str, precision: int = 4) -> Optional[str]:
    vals = [
        _avg_stat(meter, interval, f"{stage}_pho_{suffix}"),
        _avg_stat(meter, interval, f"{stage}_fid_{suffix}"),
        _avg_stat(meter, interval, f"{stage}_cor_{suffix}"),
    ]
    if any(v is None for v in vals):
        return None
    return "/".join(f"{v:.{precision}f}" for v in vals)


def _fmt_branch_weight_triplet(meter: OrderedDict, interval: int, stage: str, precision: int = 4) -> Optional[str]:
    vals = [
        _avg_stat(meter, interval, f"{stage}_w_pho"),
        _avg_stat(meter, interval, f"{stage}_w_fid"),
        _avg_stat(meter, interval, f"{stage}_w_cor"),
    ]
    if any(v is None for v in vals):
        return None
    return "/".join(f"{v:.{precision}f}" for v in vals)


def _adapter_health_warnings(meter: OrderedDict, interval: int) -> List[str]:
    warnings = []
    adapter_names = ("pho", "fid", "cor")
    for stage in ("s0", "s1", "s2"):
        weights = [_avg_stat(meter, interval, f"{stage}_w_{name}") for name in adapter_names]
        deltas = [_avg_stat(meter, interval, f"{stage}_{name}_eff") for name in adapter_names]

        for name, value in zip(adapter_names, weights):
            if value is not None and math.isfinite(value) and value < 0.08:
                warnings.append(f"{stage}.{name}:low_w")
        finite_weights = [v for v in weights if v is not None and math.isfinite(v)]
        if finite_weights and max(finite_weights) > 0.85:
            warnings.append(f"{stage}:weight_dominance")

        for name, value in zip(adapter_names, deltas):
            if value is not None and math.isfinite(value) and value < 1e-7:
                warnings.append(f"{stage}.{name}:near_zero_delta")

        finite_deltas = [v for v in deltas if v is not None and math.isfinite(v) and v > 0.0]
        if len(finite_deltas) >= 2 and max(finite_deltas) / max(min(finite_deltas), 1e-12) > 25.0:
            warnings.append(f"{stage}:delta_imbalance")

    for key, value in meter.items():
        avg_value = value / max(interval, 1)
        if not math.isfinite(avg_value):
            warnings.append(f"{key}:non_finite")
            break
    return warnings


def format_running_adapter_health_meter(adapter_meter: OrderedDict, interval: int, epoch: int, global_step: int) -> str:
    if len(adapter_meter) == 0:
        return ""
    msg = [f"[ADAPTER] epoch={epoch} step={global_step}"]
    for stage in ("s0", "s1", "s2"):
        eff = _fmt_triplet(adapter_meter, interval, stage, "eff")
        scale = _fmt_triplet(adapter_meter, interval, stage, "scale")
        gain = _fmt_triplet(adapter_meter, interval, stage, "gain")
        weight = _fmt_branch_weight_triplet(adapter_meter, interval, stage)
        state = _fmt_triplet(adapter_meter, interval, stage, "state")
        if eff is None and scale is None and gain is None and weight is None and state is None:
            continue
        parts = [f"{stage}:"]
        if eff is not None:
            parts.append(f"eff(p/f/c)={eff}")
        if scale is not None:
            parts.append(f"scale(p/f/c)={scale}")
        if gain is not None:
            parts.append(f"gain(p/f/c)={gain}")
        if weight is not None:
            parts.append(f"w(p/f/c)={weight}")
        if state is not None:
            parts.append(f"state(p/f/c)={state}")

        fid_mask = _avg_stat(adapter_meter, interval, f"{stage}_fid_mask")
        cor_mask = _avg_stat(adapter_meter, interval, f"{stage}_cor_mask")
        pho_hf = _avg_stat(adapter_meter, interval, f"{stage}_pho_hf")
        fuse_delta = _avg_stat(adapter_meter, interval, f"{stage}_fuse_delta")
        if fid_mask is not None:
            parts.append(f"fid_mask={fid_mask:.4f}")
        if cor_mask is not None:
            parts.append(f"cor_mask={cor_mask:.4f}")
        if pho_hf is not None:
            parts.append(f"pho_hf={pho_hf:.4f}")
        if fuse_delta is not None:
            parts.append(f"fuse={fuse_delta:.4f}")
        msg.append(" ".join(parts))

    warnings = _adapter_health_warnings(adapter_meter, interval)
    if warnings:
        msg.append("health_warn=" + ",".join(warnings[:8]))
    return " | ".join(msg)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps * self.eps).mean()


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def load_shape_compatible_checkpoint(model: nn.Module, checkpoint_path: str, map_location: str = "cpu") -> None:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = normalize_state_dict_keys(state_dict)
    model_state = model.state_dict()
    compatible = OrderedDict((k, v) for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"[CKPT] loaded compatible tensors: {len(compatible)}/{len(state_dict)} from {checkpoint_path}")
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")


def override_aam_scale_caps(model: nn.Module, args: argparse.Namespace) -> None:
    if not args.override_scale_caps:
        return

    cap_values = {
        "photometric_adapter": parse_float_tuple(args.photometric_scale_caps, 3, "photometric-scale-caps"),
        "fidelity_adapter": parse_float_tuple(args.fidelity_scale_caps, 3, "fidelity-scale-caps"),
        "corruption_adapter": parse_float_tuple(args.corruption_scale_caps, 3, "corruption-scale-caps"),
    }
    stage_names = ("attr_stage_l3", "attr_stage_l2", "attr_stage_l1")

    for stage_idx, stage_name in enumerate(stage_names):
        stage = getattr(model, stage_name, None)
        if stage is None:
            continue
        for adapter_name, caps in cap_values.items():
            adapter = getattr(stage, adapter_name, None)
            if adapter is not None and hasattr(adapter, "scale_cap"):
                adapter.scale_cap.fill_(float(caps[stage_idx]))

    print(
        "[AAM] scale caps active: "
        f"photometric={cap_values['photometric_adapter']} "
        f"fidelity={cap_values['fidelity_adapter']} "
        f"corruption={cap_values['corruption_adapter']}"
    )


def is_aam_parameter(name: str) -> bool:
    return name.startswith("attr_stage_")


def is_aam_scale_parameter(name: str, args: argparse.Namespace) -> bool:
    if not is_aam_parameter(name) or not name.endswith(".scale.raw"):
        return False
    if args.aam_scale_freeze_scope == "all":
        return True
    if args.aam_scale_freeze_scope == "branch":
        return any(
            marker in name
            for marker in (
                ".photometric_adapter.",
                ".fidelity_adapter.",
                ".corruption_adapter.",
            )
        )
    raise ValueError(f"Unsupported AAM scale freeze scope: {args.aam_scale_freeze_scope}")


def freeze_aam_scale_parameters(model: nn.Module, args: argparse.Namespace) -> None:
    if not args.freeze_aam_scale:
        return
    frozen = []
    for name, param in model.named_parameters():
        if is_aam_scale_parameter(name, args):
            param.requires_grad_(False)
            frozen.append(name)
    print(
        f"[FREEZE] locked {len(frozen)} AAM scale parameters "
        f"from loaded weights (scope={args.aam_scale_freeze_scope})"
    )


def print_aam_scale_snapshot(model: nn.Module, label: str) -> None:
    stage_names = ("attr_stage_l3", "attr_stage_l2", "attr_stage_l1")
    branch_names = ("photometric_adapter", "fidelity_adapter", "corruption_adapter")
    msg = []
    with torch.no_grad():
        for stage_idx, stage_name in enumerate(stage_names):
            stage = getattr(model, stage_name, None)
            if stage is None:
                continue
            parts = []
            for branch_name in branch_names:
                branch = getattr(stage, branch_name, None)
                if branch is None:
                    continue
                raw_scale = float(branch.scale().detach().cpu())
                cap = float(branch.scale_cap.detach().cpu()) if hasattr(branch, "scale_cap") else raw_scale
                parts.append(f"{min(raw_scale, cap):.4f}/{raw_scale:.4f}/{cap:.4f}")
            fuse = getattr(stage, "fusion", None)
            if fuse is not None and hasattr(fuse, "scale"):
                parts.append(f"fuse={float(fuse.scale().detach().cpu()):.4f}")
            msg.append(f"s{stage_idx}: eff/raw/cap(p/f/c)=" + ",".join(parts))
    if msg:
        print(f"[AAM] scale snapshot {label} | " + " | ".join(msg))


def is_backbone_parameter_in_scope(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "none":
        return False
    if scope == "output":
        return name.startswith("output.")
    if scope == "decoder_output":
        return name.startswith(("decoder_level1.", "refinement.", "output."))
    if scope == "decoder":
        return name.startswith((
            "up4_3.",
            "reduce_chan_level3.",
            "decoder_level3.",
            "up3_2.",
            "reduce_chan_level2.",
            "decoder_level2.",
            "up2_1.",
            "decoder_level1.",
            "refinement.",
            "output.",
        ))
    raise ValueError(f"Unsupported backbone train scope: {scope}")


def build_finetune_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    backbone_params = []
    aam_params = []
    for name, param in model.named_parameters():
        if is_aam_parameter(name):
            aam_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            "lr": args.lr * args.backbone_lr_scale,
            "weight_decay": args.weight_decay,
            "name": "backbone",
        },
        {
            "params": aam_params,
            "lr": args.lr * args.aam_lr_scale,
            "weight_decay": args.weight_decay,
            "name": "aam",
        },
    ]
    print(
        f"[OPT] backbone params={sum(p.numel() for p in backbone_params):,} "
        f"lr={args.lr * args.backbone_lr_scale:.3e}; "
        f"AAM params={sum(p.numel() for p in aam_params):,} "
        f"lr={args.lr * args.aam_lr_scale:.3e}"
    )
    return optim.AdamW(param_groups, betas=(0.9, 0.999))


def set_epoch_trainable(model: nn.Module, epoch: int, args: argparse.Namespace) -> None:
    train_backbone = epoch >= args.freeze_backbone_epochs
    train_aam = epoch >= args.freeze_aam_epochs
    for name, param in model.named_parameters():
        if is_aam_parameter(name):
            lock_scale = args.freeze_aam_scale and is_aam_scale_parameter(name, args)
            param.requires_grad = train_aam and not lock_scale
        else:
            param.requires_grad = train_backbone and is_backbone_parameter_in_scope(name, args.backbone_train_scope)
    trainable_backbone = sum(p.numel() for n, p in model.named_parameters() if (not is_aam_parameter(n)) and p.requires_grad)
    trainable_aam = sum(p.numel() for n, p in model.named_parameters() if is_aam_parameter(n) and p.requires_grad)
    frozen_scale = sum(p.numel() for n, p in model.named_parameters() if args.freeze_aam_scale and is_aam_scale_parameter(n, args))
    print(
        f"[FREEZE] epoch={epoch + 1} "
        f"backbone={'train' if train_backbone else 'frozen'}({args.backbone_train_scope}, {trainable_backbone:,} params) "
        f"AAM={'train' if train_aam else 'frozen'}({trainable_aam:,} params) "
        f"AAM_scale_locked={frozen_scale:,}"
    )


def move_tensors_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, OrderedDict):
        return OrderedDict((key, move_tensors_to_cpu(value)) for key, value in obj.items())
    if isinstance(obj, dict):
        return {key: move_tensors_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [move_tensors_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(move_tensors_to_cpu(value) for value in obj)
    return obj


def clear_cuda_cache(enabled: bool = True) -> None:
    gc.collect()
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LambdaLR,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_psnr: float,
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": best_psnr,
        "state_dict": model.state_dict(),
        "args": vars(args),
    }
    if args.save_optimizer_state:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
        payload["scaler"] = scaler.state_dict()
    if args.cpu_checkpoint:
        payload = move_tensors_to_cpu(payload)
    torch.save(payload, path)
    del payload
    clear_cuda_cache(args.empty_cache_after_checkpoint)


def save_nonfinite_debug(
    debug_dir: str,
    epoch: int,
    global_step: int,
    iter_idx: int,
    reason: str,
    loss: Optional[torch.Tensor] = None,
    loss_rec: Optional[torch.Tensor] = None,
    loss_attr: Optional[torch.Tensor] = None,
    first_nonfinite: Optional[str] = None,
) -> None:
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)

    def scalar(x: Optional[torch.Tensor]):
        if x is None or not torch.is_tensor(x):
            return None
        value = float(x.detach().float().cpu().item())
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "iter_idx": iter_idx,
        "reason": reason,
        "first_nonfinite": first_nonfinite,
        "loss": scalar(loss),
        "loss_rec": scalar(loss_rec),
        "loss_attr": scalar(loss_attr),
    }
    path = os.path.join(debug_dir, f"nonfinite_step_{global_step:08d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def model_has_nonfinite_parameter(model: nn.Module) -> Optional[str]:
    for name, param in model.named_parameters():
        if not is_finite_tensor(param):
            return name
    return None


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(0, 1)
    arr = x.permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def save_rgb_image(x: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(tensor_to_uint8(x)).save(path)


def save_validation_images(
    batch: Dict[str, Any],
    input_01: torch.Tensor,
    restored_01: torch.Tensor,
    target_01: torch.Tensor,
    save_dir: str,
    global_step: int,
    batch_idx: int,
) -> None:
    if not save_dir:
        return
    if isinstance(batch, dict):
        degraded_paths = batch.get("degraded_path", ["sample"] * restored_01.size(0))
        tasks = batch.get("task", ["val"] * restored_01.size(0))
    else:
        degraded_paths = ["sample"] * restored_01.size(0)
        tasks = ["val"] * restored_01.size(0)
    step_dir = os.path.join(save_dir, f"step_{global_step:08d}")
    for sample_idx in range(restored_01.size(0)):
        degraded_path = degraded_paths[sample_idx] if isinstance(degraded_paths, (list, tuple)) else str(degraded_paths)
        task = tasks[sample_idx] if isinstance(tasks, (list, tuple)) else str(tasks)
        stem = Path(degraded_path).stem if degraded_path else f"sample_{batch_idx:03d}_{sample_idx:02d}"
        name = f"{batch_idx:03d}_{sample_idx:02d}_{stem}.png"

        restored_path = os.path.join(step_dir, task, "restored", name)
        compare_path = os.path.join(step_dir, task, "compare", name)
        save_rgb_image(restored_01[sample_idx], restored_path)

        compare = torch.cat(
            [input_01[sample_idx], restored_01[sample_idx], target_01[sample_idx]],
            dim=2,
        )
        save_rgb_image(compare, compare_path)


@torch.inference_mode()
def run_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    global_step: int,
    normalize_to: str,
    max_batches: int,
    save_dir: str = "",
    pad_multiple: int = 64,
    online_extractor: Optional[nn.Module] = None,
) -> float:
    model.eval()
    scores = []
    for batch_idx, batch in enumerate(tqdm(loader, desc="Validation", leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        target, input_, degraded_context, clean_context = contexts_for_batch(batch, device, online_extractor)
        input_pad, pads = pad_to_multiple(input_, pad_multiple)
        restored, _ = model(input_pad, context=degraded_context, clean_context=None, clean_guide_img=None, steps=global_step)
        restored = unpad(restored, pads)
        restored_01 = to_zero_one(restored, normalize_to).clamp(0, 1)
        target_01 = to_zero_one(target, normalize_to).clamp(0, 1)
        input_01 = to_zero_one(input_, normalize_to).clamp(0, 1)
        save_validation_images(batch, input_01, restored_01, target_01, save_dir, global_step, batch_idx)
        for pred, gt in zip(restored_01, target_01):
            scores.append(image_utils.torchPSNR(pred, gt))
    model.train()
    if not scores:
        return 0.0
    return torch.stack(scores).mean().item()


def build_model(args: argparse.Namespace) -> AttributeIR:
    num_blocks = parse_int_tuple(args.num_blocks, 4, "num-blocks")
    heads = parse_int_tuple(args.heads, 4, "heads")
    return AttributeIR(
        dim=args.dim,
        num_blocks=num_blocks,
        heads=heads,
        num_refinement_blocks=args.num_refinement_blocks,
        d_token=args.d_token,
        d_ctrl=args.d_ctrl,
        token_in_dim=4096 + 8,
        attr_score_dim=8,
        decoder=True,
    )


def main() -> None:
    args = parse_args()
    if args.warmup_updates is not None:
        args.warmup_epochs = int(args.warmup_updates)
        print("[LR] --warmup-updates is deprecated; treating its value as --warmup-epochs.")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_tasks = parse_task_names(args.tasks)
    val_tasks = parse_task_names(args.eval_tasks) if args.eval_tasks else train_tasks
    train_set = AlignedDataset(
        root=args.dataset_root,
        mode="train",
        train_size=args.crop_size,
        augment_flip=True,
        target_per_task=args.target_per_task,
        use_tasks=train_tasks,
        normalize_to=args.normalize_to,
        eval_info_root=args.context_root,
        deg_eval_info_root=args.degraded_context_root or None,
        clean_eval_info_root=args.clean_context_root or None,
        load_contexts=not args.online_iqa,
        return_paths=args.online_iqa,
    )
    val_set = AlignedDataset(
        root=args.dataset_root,
        mode="eval",
        train_size=args.eval_center_crop or args.crop_size,
        augment_flip=False,
        target_per_task=args.target_per_task,
        use_tasks=val_tasks,
        normalize_to=args.normalize_to,
        eval_info_root=args.context_root,
        deg_eval_info_root=args.degraded_context_root or None,
        clean_eval_info_root=args.clean_context_root or None,
        load_contexts=not args.online_iqa,
        return_paths=args.online_iqa,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=True,
        drop_last=False,
        num_workers=args.eval_num_workers,
        pin_memory=device.type == "cuda",
    )
    val_image_dir = args.val_image_dir or os.path.join(args.checkpoint_dir, "val_images")
    nan_debug_dir = args.nan_debug_dir or os.path.join(args.checkpoint_dir, "nan_debug")
    print(
        f"[VAL] interval={args.val_every} iters, random_samples={args.max_val_batches}, "
        f"save_images={args.save_val_images}, image_dir={val_image_dir}"
    )

    online_extractor = None
    if args.online_iqa:
        from tools.q_instruct_online import QInstructAttributeExtractor

        online_extractor = QInstructAttributeExtractor(
            model_path=args.online_iqa_model_path,
            model_base=args.online_iqa_model_base or None,
            model_name=args.online_iqa_model_name,
            device=args.online_iqa_device,
            center_crop_square=args.online_iqa_center_crop_square,
        )
        print(
            f"[IQA] online Q-Instruct enabled for train/validation; "
            f"device={args.online_iqa_device}, full_image={not args.online_iqa_center_crop_square}"
        )
    else:
        print("[IQA] online Q-Instruct disabled; reading precomputed CSV contexts.")

    model = build_model(args).to(device)
    override_aam_scale_caps(model, args)
    if args.pretrained:
        load_shape_compatible_checkpoint(model, args.pretrained, map_location="cpu")
        override_aam_scale_caps(model, args)

    optimizer = build_finetune_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    print(
        f"[LR] epoch-level warmup+cosine scheduler: base_lr={args.lr:.3e}, "
        f"warmup_epochs={args.warmup_epochs}, min_lr={args.lr * args.min_lr_ratio:.3e}; "
        "scheduler.step() runs once after each epoch."
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    # rec_loss_fn = CharbonnierLoss()
    rec_loss_fn = nn.L1Loss()

    start_epoch = 0
    global_step = 0
    best_psnr = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(normalize_state_dict_keys(state_dict), strict=True)
        override_aam_scale_caps(model, args)
        if args.resume_training_state:
            required_keys = ("optimizer", "scheduler")
            missing_keys = [key for key in required_keys if not isinstance(ckpt, dict) or key not in ckpt]
            if missing_keys:
                raise KeyError(f"Checkpoint lacks training state keys: {missing_keys}")
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_psnr = float(ckpt.get("best_psnr", 0.0))
            print(f"[CKPT] resumed full training state from {args.resume} at epoch={start_epoch}, step={global_step}")
        else:
            print(
                f"[CKPT] loaded model weights from {args.resume}; "
                f"optimizer/scheduler/epoch/global_step reset, lr={args.lr:.3e}"
            )

    freeze_aam_scale_parameters(model, args)
    print_aam_scale_snapshot(model, "after weight load")

    latest_path = os.path.join(args.checkpoint_dir, "attribute_ir_last.pth")
    best_path = os.path.join(args.checkpoint_dir, "attribute_ir_best.pth")
    safe_path = os.path.join(args.checkpoint_dir, "attribute_ir_safe.pth")
    next_val_step = ((global_step // args.val_every) + 1) * args.val_every if args.val_every > 0 else None

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        set_epoch_trainable(model, epoch, args)
        model.train()
        running = OrderedDict((k, 0.0) for k in ["loss_total", "loss_rec", "loss_attr"])
        running_adapter = init_running_adapter_health_meter()
        running_count = 0
        seen = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", ncols=120)

        for iter_idx, batch in enumerate(pbar):
            target, input_, degraded_context, clean_context = contexts_for_batch(batch, device, online_extractor)
            finite_batch, bad_batch_name = batch_is_finite(target, input_, degraded_context, clean_context)
            if not finite_batch:
                global_step += 1
                msg = f"[NONFINITE] epoch={epoch + 1} step={global_step} bad_batch_tensor={bad_batch_name}"
                pbar.write(msg)
                save_nonfinite_debug(
                    nan_debug_dir,
                    epoch=epoch + 1,
                    global_step=global_step,
                    iter_idx=iter_idx,
                    reason="batch",
                    first_nonfinite=bad_batch_name,
                )
                optimizer.zero_grad(set_to_none=True)
                seen = 0
                if args.stop_on_nonfinite:
                    raise FloatingPointError(msg)
                if args.skip_nonfinite:
                    continue

            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                restored, regs_list = model(
                    input_,
                    context=degraded_context,
                    clean_context=clean_context,
                    clean_guide_img=target,
                    steps=global_step,
                )
                loss_rec = rec_loss_fn(restored, target)
                loss_attr, attr_detail = compute_attr_loss_detail(regs_list)
                attr_weight = get_attr_loss_weight(
                    global_step,
                    args.attr_loss_warmup_steps,
                    args.attr_loss_weight_min,
                    args.attr_loss_weight_max,
                )
                loss = loss_rec + attr_weight * loss_attr
                loss_for_backward = loss / max(args.accumulation_steps, 1)

            first_bad = None
            if not is_finite_tensor(loss) or not is_finite_tensor(loss_rec) or not is_finite_tensor(loss_attr) or not is_finite_tensor(restored):
                first_bad = first_nonfinite_tensor_name(
                    {
                        "restored": restored,
                        "loss": loss,
                        "loss_rec": loss_rec,
                        "loss_attr": loss_attr,
                        "attr_detail": attr_detail,
                        "regs": regs_list,
                    },
                    prefix="train",
                )
            if first_bad is not None:
                global_step += 1
                msg = f"[NONFINITE] epoch={epoch + 1} step={global_step} first_nonfinite={first_bad}; skipping update"
                pbar.write(msg)
                save_nonfinite_debug(
                    nan_debug_dir,
                    epoch=epoch + 1,
                    global_step=global_step,
                    iter_idx=iter_idx,
                    reason="forward_or_loss",
                    loss=loss,
                    loss_rec=loss_rec,
                    loss_attr=loss_attr,
                    first_nonfinite=first_bad,
                )
                optimizer.zero_grad(set_to_none=True)
                seen = 0
                if args.stop_on_nonfinite:
                    raise FloatingPointError(msg)
                if args.skip_nonfinite:
                    continue

            adapter_stats = extract_model_adapter_health(regs_list)
            update_running_adapter_health_meter(running_adapter, adapter_stats)
            scaler.scale(loss_for_backward).backward()
            seen += 1
            running_count += 1
            global_step += 1

            do_update = seen % max(args.accumulation_steps, 1) == 0 or iter_idx == len(train_loader) - 1
            if do_update:
                scaler.unscale_(optimizer)
                clip_limit = args.grad_clip if args.grad_clip > 0 else float("inf")
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
                if not torch.isfinite(grad_norm):
                    msg = f"[NONFINITE] epoch={epoch + 1} step={global_step} nonfinite_grad_norm={float(grad_norm)}; skipping optimizer step"
                    pbar.write(msg)
                    save_nonfinite_debug(
                        nan_debug_dir,
                        epoch=epoch + 1,
                        global_step=global_step,
                        iter_idx=iter_idx,
                        reason="grad",
                        loss=loss,
                        loss_rec=loss_rec,
                        loss_attr=loss_attr,
                        first_nonfinite="grad_norm",
                    )
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    seen = 0
                    if args.stop_on_nonfinite:
                        raise FloatingPointError(msg)
                    if args.skip_nonfinite:
                        continue
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    bad_param = model_has_nonfinite_parameter(model)
                    if bad_param is not None:
                        msg = f"[NONFINITE] epoch={epoch + 1} step={global_step} nonfinite_parameter={bad_param}"
                        pbar.write(msg)
                        save_nonfinite_debug(
                            nan_debug_dir,
                            epoch=epoch + 1,
                            global_step=global_step,
                            iter_idx=iter_idx,
                            reason="parameter",
                            loss=loss,
                            loss_rec=loss_rec,
                            loss_attr=loss_attr,
                            first_nonfinite=bad_param,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        if args.stop_on_nonfinite:
                            raise FloatingPointError(msg)
                        if args.skip_nonfinite:
                            continue
                optimizer.zero_grad(set_to_none=True)

            running["loss_total"] += float(loss.detach().item())
            running["loss_rec"] += float(loss_rec.detach().item())
            running["loss_attr"] += float(loss_attr.detach().item())
            for key, value in attr_detail.items():
                running[key] = running.get(key, 0.0) + float(value.detach().item())

            if global_step % args.print_every == 0:
                denom = max(running_count, 1)
                log_bits = [
                    f"epoch={epoch + 1}",
                    f"step={global_step}",
                    f"lr={optimizer.param_groups[0]['lr']:.3e}",
                    f"attr_w={attr_weight:.3f}",
                    f"loss={running['loss_total'] / denom:.5f}",
                    f"rec={running['loss_rec'] / denom:.5f}",
                    f"attr={running['loss_attr'] / denom:.5f}",
                ]
                pbar.set_postfix_str(" | ".join(log_bits))
                pbar.write("[TRAIN] " + " | ".join(log_bits))
                adapter_msg = format_running_adapter_health_meter(
                    running_adapter,
                    interval=denom,
                    epoch=epoch + 1,
                    global_step=global_step,
                )
                if adapter_msg:
                    pbar.write(adapter_msg)
                running = OrderedDict((k, 0.0) for k in running.keys())
                running_adapter = init_running_adapter_health_meter()
                running_count = 0

            if args.val_every > 0 and do_update and next_val_step is not None and global_step >= next_val_step:
                scheduled_val_step = next_val_step
                psnr = run_validation(
                    model,
                    val_loader,
                    device,
                    global_step,
                    args.normalize_to,
                    args.max_val_batches,
                    save_dir=val_image_dir if args.save_val_images else "",
                    online_extractor=online_extractor,
                )
                clear_cuda_cache(args.empty_cache_after_validation)
                print(f"[VAL] step={global_step} scheduled_step={scheduled_val_step} psnr={psnr:.4f} best={best_psnr:.4f}")
                if args.save_safe_checkpoint and model_has_nonfinite_parameter(model) is None and math.isfinite(psnr):
                    save_checkpoint(safe_path, model, optimizer, scheduler, scaler, epoch, global_step, best_psnr, args)
                    print(f"[CKPT] saved latest finite checkpoint: {safe_path}")
                if psnr > best_psnr:
                    best_psnr = psnr
                    save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, global_step, best_psnr, args)
                    print(f"[CKPT] saved best: {best_path}")
                while next_val_step is not None and next_val_step <= global_step:
                    next_val_step += args.val_every

        if args.val_at_epoch_end:
            psnr = run_validation(
                model,
                val_loader,
                device,
                global_step,
                args.normalize_to,
                args.max_val_batches,
                save_dir=val_image_dir if args.save_val_images else "",
                online_extractor=online_extractor,
            )
            clear_cuda_cache(args.empty_cache_after_validation)
            improved = psnr > best_psnr
            if improved:
                best_psnr = psnr
            print(f"[VAL] epoch={epoch + 1} step={global_step} psnr={psnr:.4f} best={best_psnr:.4f}")
            if improved:
                save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, global_step, best_psnr, args)
                print(f"[CKPT] saved best: {best_path}")

        lr_before = optimizer.param_groups[0]["lr"]
        scheduler.step()
        lr_after = optimizer.param_groups[0]["lr"]
        print(f"[LR] epoch={epoch + 1} lr updated once: {lr_before:.3e} -> {lr_after:.3e}")
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, global_step, best_psnr, args)
        if args.save_every_epoch:
            save_checkpoint(
                os.path.join(args.checkpoint_dir, f"attribute_ir_epoch_{epoch + 1:03d}.pth"),
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_psnr,
                args,
            )


if __name__ == "__main__":
    main()
