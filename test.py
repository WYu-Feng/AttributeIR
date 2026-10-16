from __future__ import annotations

import argparse
import csv
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.test_dataset import AttributeTestDataset
from models.attribute_ir import AttributeIR
import utils.image_utils as image_utils


def parse_int_tuple(value: str, expected_len: int, name: str) -> Tuple[int, ...]:
    out = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(out) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated integers")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AttributeIR on paired restoration datasets.")
    p.add_argument("--checkpoint", type=str, default="./check/combined_5d_backbone_aam.pth")
    p.add_argument("--dataset-root", type=str, default="./datasets")
    p.add_argument("--context-root", type=str, default="./q_instruct_results")
    p.add_argument("--degraded-context-root", type=str, default="")
    p.add_argument("--clean-context-root", type=str, default="")
    p.add_argument(
        "--online-iqa",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate Q-Instruct context from each complete test image during evaluation (default).",
    )
    p.add_argument("--online-iqa-device", type=str, default="cuda:0")
    p.add_argument("--online-iqa-model-path", type=str, default="q-future")
    p.add_argument("--online-iqa-model-base", type=str, default="")
    p.add_argument("--online-iqa-model-name", type=str, default="mplug_owl2")
    p.add_argument("--online-iqa-center-crop-square", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--split", type=str, default="eval", help=argparse.SUPPRESS)
    p.add_argument("--tasks", type=str, default="gopro")
    p.add_argument("--output-dir", type=str, default="./results/attribute_ir")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-batches", type=int, default=-1)
    p.add_argument("--test-size", type=int, default=0,
                   help="Maximum center-crop size; 0 keeps the complete image (paper-aligned inference).")
    p.add_argument("--eval-center-crop", type=int, default=0)
    p.add_argument("--missing-context", choices=["error", "zeros"], default="error")
    p.add_argument("--normalize-to", choices=["auto", "zero1", "minus1_1"], default="auto")
    p.add_argument("--pad-multiple", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-input", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-target", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use-checkpoint-config", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num-blocks", type=str, default="4,6,6,8")
    p.add_argument("--heads", type=str, default="1,2,4,8")
    p.add_argument("--num-refinement-blocks", type=int, default=4)
    p.add_argument("--d-token", type=int, default=256)
    p.add_argument("--d-ctrl", type=int, default=64)
    return p.parse_args()


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


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(0, 1)
    arr = x.permute(1, 2, 0).numpy()
    return (arr * 255.0).round().astype(np.uint8)


def save_tensor_image(x: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(tensor_to_uint8(x)).save(path)


def luminance_mean(x: torch.Tensor) -> float:
    x = x.detach().float().clamp(0, 1)
    y = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
    return float(y.mean().item())


def clip_ratio(x: torch.Tensor, low: bool = True) -> float:
    x = x.detach().float()
    if low:
        return float((x <= 0.0).float().mean().item())
    return float((x >= 1.0).float().mean().item())


def maybe_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float("nan")
    pred_np = tensor_to_uint8(pred)
    target_np = tensor_to_uint8(target)
    return float(structural_similarity(target_np, pred_np, channel_axis=2, data_range=255))


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def _arg_from_ckpt(args: argparse.Namespace, key: str, fallback: Any) -> Any:
    ckpt_args = getattr(args, "_ckpt_args", {}) or {}
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)
    if not isinstance(ckpt_args, dict):
        return fallback
    return ckpt_args.get(key.replace("-", "_"), fallback)


def resolve_normalize_to(args: argparse.Namespace) -> str:
    if args.normalize_to != "auto":
        return args.normalize_to
    value = _arg_from_ckpt(args, "normalize_to", "minus1_1")
    if value not in {"zero1", "minus1_1"}:
        print(f"[DATA] checkpoint normalize_to={value!r} is invalid; fallback to minus1_1")
        return "minus1_1"
    return str(value)


def coerce_int_tuple(value: Any, expected_len: int, name: str) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        out = tuple(int(x) for x in value)
        if len(out) != expected_len:
            raise ValueError(f"{name} must contain {expected_len} integers")
        return out
    return parse_int_tuple(str(value), expected_len, name)


def build_model(args: argparse.Namespace) -> AttributeIR:
    num_blocks = coerce_int_tuple(_arg_from_ckpt(args, "num_blocks", args.num_blocks), 4, "num-blocks")
    heads = coerce_int_tuple(_arg_from_ckpt(args, "heads", args.heads), 4, "heads")
    return AttributeIR(
        dim=int(_arg_from_ckpt(args, "dim", args.dim)),
        num_blocks=num_blocks,
        heads=heads,
        num_refinement_blocks=int(_arg_from_ckpt(args, "num_refinement_blocks", args.num_refinement_blocks)),
        d_token=int(_arg_from_ckpt(args, "d_token", args.d_token)),
        d_ctrl=int(_arg_from_ckpt(args, "d_ctrl", args.d_ctrl)),
        token_in_dim=4096 + 8,
        attr_score_dim=8,
        decoder=True,
    )


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    if isinstance(ckpt, dict):
        return {"state_dict": ckpt}
    raise ValueError(f"Unsupported checkpoint format: {path}")


@torch.no_grad()
def evaluate(
    model: AttributeIR,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    global_step: int,
    online_extractor: QInstructAttributeExtractor | None = None,
) -> None:
    model.eval()
    rows = []
    psnr_values = []
    ssim_values = []

    if args.save_images:
        os.makedirs(os.path.join(args.output_dir, "restored"), exist_ok=True)
        if args.save_input:
            os.makedirs(os.path.join(args.output_dir, "input"), exist_ok=True)
        if args.save_target:
            os.makedirs(os.path.join(args.output_dir, "target"), exist_ok=True)

    pbar = tqdm(loader, desc="Testing", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        target = batch["target"].to(device, non_blocking=True)
        input_ = batch["input"].to(device, non_blocking=True)
        if online_extractor is not None:
            degraded_context = online_extractor(list(batch["degraded_path"])).to(device, non_blocking=True)
        else:
            degraded_context = batch["degraded_context"].to(device, non_blocking=True)

        input_pad, pads = pad_to_multiple(input_, args.pad_multiple)
        restored, _ = model(input_pad, context=degraded_context, clean_context=None, clean_guide_img=None, steps=global_step)
        restored = unpad(restored, pads)
        restored_01 = to_zero_one(restored, args.normalize_to).clamp(0, 1)
        target_01 = to_zero_one(target, args.normalize_to).clamp(0, 1)
        input_01 = to_zero_one(input_, args.normalize_to).clamp(0, 1)

        paths = batch["degraded_path"]
        tasks = batch["task"]
        for i in range(restored_01.size(0)):
            name = Path(paths[i]).stem
            task = tasks[i]
            psnr = float(image_utils.torchPSNR(restored_01[i], target_01[i]).item())
            ssim = maybe_ssim(restored_01[i], target_01[i])
            input_y = luminance_mean(input_01[i])
            restored_y = luminance_mean(restored_01[i])
            target_y = luminance_mean(target_01[i])
            low_clip = clip_ratio(restored_01[i], low=True)
            high_clip = clip_ratio(restored_01[i], low=False)
            psnr_values.append(psnr)
            if not np.isnan(ssim):
                ssim_values.append(ssim)
            rows.append({
                "name": name,
                "task": task,
                "psnr": psnr,
                "ssim": ssim,
                "input_y": input_y,
                "restored_y": restored_y,
                "target_y": target_y,
                "restored_minus_input_y": restored_y - input_y,
                "restored_minus_target_y": restored_y - target_y,
                "low_clip_ratio": low_clip,
                "high_clip_ratio": high_clip,
                "path": paths[i],
            })

            if args.save_images:
                save_tensor_image(restored_01[i], os.path.join(args.output_dir, "restored", task, f"{name}.png"))
                if args.save_input:
                    save_tensor_image(input_01[i], os.path.join(args.output_dir, "input", task, f"{name}.png"))
                if args.save_target:
                    save_tensor_image(target_01[i], os.path.join(args.output_dir, "target", task, f"{name}.png"))

        pbar.set_postfix({"psnr": f"{np.mean(psnr_values):.3f}", "samples": len(rows)})

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "task",
                "psnr",
                "ssim",
                "input_y",
                "restored_y",
                "target_y",
                "restored_minus_input_y",
                "restored_minus_target_y",
                "low_clip_ratio",
                "high_clip_ratio",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    mean_ssim = float(np.mean(ssim_values)) if ssim_values else float("nan")
    mean_input_y = float(np.mean([r["input_y"] for r in rows])) if rows else float("nan")
    mean_restored_y = float(np.mean([r["restored_y"] for r in rows])) if rows else float("nan")
    mean_target_y = float(np.mean([r["target_y"] for r in rows])) if rows else float("nan")
    mean_dark_shift = float(np.mean([r["restored_minus_input_y"] for r in rows])) if rows else float("nan")
    mean_target_gap = float(np.mean([r["restored_minus_target_y"] for r in rows])) if rows else float("nan")
    mean_low_clip = float(np.mean([r["low_clip_ratio"] for r in rows])) if rows else float("nan")
    with open(os.path.join(args.output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"samples: {len(rows)}\n")
        f.write(f"mean_psnr: {mean_psnr:.6f}\n")
        f.write(f"mean_ssim: {mean_ssim:.6f}\n")
        f.write(f"mean_input_y: {mean_input_y:.6f}\n")
        f.write(f"mean_restored_y: {mean_restored_y:.6f}\n")
        f.write(f"mean_target_y: {mean_target_y:.6f}\n")
        f.write(f"mean_restored_minus_input_y: {mean_dark_shift:.6f}\n")
        f.write(f"mean_restored_minus_target_y: {mean_target_gap:.6f}\n")
        f.write(f"mean_low_clip_ratio: {mean_low_clip:.6f}\n")
    print(f"[FINAL] samples={len(rows)} mean_psnr={mean_psnr:.4f} mean_ssim={mean_ssim:.4f}")
    print(
        f"[FINAL] y_mean input/restored/target={mean_input_y:.4f}/{mean_restored_y:.4f}/{mean_target_y:.4f} "
        f"restored-input={mean_dark_shift:.4f} restored-target={mean_target_gap:.4f} low_clip={mean_low_clip:.4f}"
    )
    print(f"[FINAL] metrics saved to {csv_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    args._ckpt_args = ckpt.get("args", {}) if args.use_checkpoint_config else {}
    args.normalize_to = resolve_normalize_to(args)
    print(f"[DATA] normalize_to={args.normalize_to}")

    model = build_model(args).to(device)
    missing, unexpected = model.load_state_dict(normalize_state_dict_keys(ckpt["state_dict"]), strict=False)
    print(f"[CKPT] loaded {args.checkpoint}")
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")
    global_step = int(ckpt.get("global_step", 0))

    dataset = AttributeTestDataset(
        root=args.dataset_root,
        test_size=args.test_size,
        eval_center_crop=args.eval_center_crop,
        tasks=args.tasks,
        normalize_to=args.normalize_to,
        context_root=args.context_root,
        degraded_context_root=args.degraded_context_root or None,
        clean_context_root=args.clean_context_root or None,
        missing_context=args.missing_context,
        load_contexts=not args.online_iqa,
    )
    print(f"[DATA] tasks={args.tasks} samples={len(dataset)} test_size={args.test_size}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
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
        print(f"[IQA] online extractor loaded on {args.online_iqa_device}")
    evaluate(model, loader, args, device, global_step=global_step, online_extractor=online_extractor)


if __name__ == "__main__":
    main()
