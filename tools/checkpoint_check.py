from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.attribute_ir import AttributeIR


def parse_int_tuple(value: str, expected_len: int, name: str) -> Tuple[int, ...]:
    out = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if len(out) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain {expected_len} comma-separated integers")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dry-check AttributeIR construction, checkpoint loading, and forward pass.")
    p.add_argument("--checkpoint", type=str, default="./check/combined_5d_backbone_aam.pth")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num-blocks", type=str, default="4,6,6,8")
    p.add_argument("--heads", type=str, default="1,2,4,8")
    p.add_argument("--num-refinement-blocks", type=int, default=4)
    p.add_argument("--d-token", type=int, default=256)
    p.add_argument("--d-ctrl", type=int, default=64)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def build_model(args: argparse.Namespace) -> AttributeIR:
    return AttributeIR(
        dim=args.dim,
        num_blocks=parse_int_tuple(args.num_blocks, 4, "num-blocks"),
        heads=parse_int_tuple(args.heads, 4, "heads"),
        num_refinement_blocks=args.num_refinement_blocks,
        d_token=args.d_token,
        d_ctrl=args.d_ctrl,
        token_in_dim=4096 + 8,
        attr_score_dim=8,
        decoder=True,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(normalize_state_dict_keys(state_dict), strict=args.strict)
        print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    x = torch.rand(args.batch_size, 3, args.image_size, args.image_size, device=device)
    context = torch.zeros(args.batch_size, 4096 + 8, device=device)
    clean_context = torch.zeros(args.batch_size, 4096 + 8, device=device)
    with torch.no_grad():
        out, regs = model(x, context=context, clean_context=clean_context, clean_guide_img=x, steps=0)
    print(f"[OK] output_shape={tuple(out.shape)} output_mean={out.mean().item():.6f} stages={len(regs)}")


if __name__ == "__main__":
    main()
