#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q-Instruct quality features: stable GLOBAL CSV + OPTIONAL independent local views.

Core principle
--------------
* The CSV ALWAYS contains the Q-Instruct GLOBAL image representation only:
  8 rows (attributes) x (hidden_dim + 9) feature columns. No averaging, no
  learned or heuristic fusion of hidden states / word logits.
* Optional original-pixel local crops are scored INDEPENDENTLY and saved to
  .views.npz with boxes. They never change the CSV features.
* The visual encoder always receives one 448x448 input for each independent
  image/view. This is NOT native multi-image VLM reasoning.
* Default global preprocessing is the user's original letterbox (ratio-preserving
  bicubic resize + original (123,117,104) padding); the original prompts and cache
  computation are preserved. Optional 'processor' permits an official-default
  preprocessor baseline for comparison.
* Separate cache keys for global CSV and local views: changing tile settings
  need NOT recompute a valid global CSV.

This is feature extraction, not a claim that local Q-Instruct judgments are
calibrated for crop-wise image quality. Test downstream effectiveness before
using these features to train a restoration policy.

Examples
--------
# Preserve original global features, no local inference:
python3 model_score_ours.py --dataset-root ./datasets --context-root ./out \\
  --include JPEG --view-mode global --batch-size 32 --only-missing

# Extract independent local evidence as well:
python3 model_score_ours.py --dataset-root ./datasets --context-root ./out \\
  --include JPEG --view-mode global-local --save-view-features \\
  --tile-strategy native --max-local-tiles 8 \\
  --batch-size 32 --view-batch-size 2 --only-missing

Outputs
-------
<stem>.csv             : GLOBAL Q-Instruct features [8, D+9], float32 -> decimal CSV
<stem>.meta.json       : global preprocessing settings and source fingerprint
<stem>.views.npz       : optional view features [V,8,D+9] (view 0 is GLOBAL)
<stem>.views.meta.json : optional local tile settings, boxes and source fingerprint

Note: the first feature stored in NPZ can be read from a valid global CSV if
local features are produced on a later run, hence has CSV's 8-decimal rounding.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tqdm import tqdm

VISION_SIZE = 448
CACHE_VERSION = "q_instruct_global_primary_v2"
ATTRIBUTES = [
    "Brightness", "Contrast", "Color", "Sharpness",
    "Structure", "Texture", "Noise", "Artifact",
]
PREFERENCE_WORDS = [
    "good", "average", "poor",
    "high", "medium", "low",
    "fine", "acceptable", "bad",
]


@dataclass(frozen=True)
class ViewInfo:
    kind: str  # global or local
    box_xyxy: Tuple[int, int, int, int]  # coordinates in processed source orientation
    row: int = -1
    col: int = -1


def longest_common_prefix_len(seq_list: Sequence[torch.Tensor]) -> int:
    """Unchanged original longest common prompt prefix in token-id space."""
    m = min(x.shape[1] for x in seq_list)
    i = 0
    while i < m:
        ref = seq_list[0][0, i].item()
        if all(s[0, i].item() == ref for s in seq_list[1:]):
            i += 1
        else:
            break
    return i


def read_image_rgb(path: str, exif_transpose: bool = True) -> Image.Image:
    with Image.open(path) as f:
        src = ImageOps.exif_transpose(f) if exif_transpose else f
        return src.convert("RGB")


def mean_pad_color(image_processor) -> Tuple[int, int, int]:
    """Derive RGB pad color from image mean; CLIP defaults -> (123, 117, 104)."""
    means = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    if isinstance(means, (float, int)):
        means = [float(means)] * 3
    if len(means) != 3:
        raise ValueError(f"Expected three-channel image_mean, got {means!r}")
    mul = 255 if max(means) <= 1 else 1
    return tuple(int(round(max(0, min(255, float(x) * mul)))) for x in means)


def letterbox_for_model(
    image: Image.Image,
    target_size: int = VISION_SIZE,
    pad_color: Tuple[int, int, int] = (123, 117, 104),
) -> Image.Image:
    """Same full-image, ratio-preserving 448 letterbox as original user script."""
    if image.width < 1 or image.height < 1:
        raise ValueError("Image width/height must be positive")
    factor = min(target_size / image.width, target_size / image.height)
    wh = (
        max(1, min(target_size, round(image.width * factor))),
        max(1, min(target_size, round(image.height * factor))),
    )
    resized = image.resize(wh, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    canvas.paste(resized, ((target_size - wh[0]) // 2, (target_size - wh[1]) // 2))
    return canvas


def _start_windows(length: int, count: int, window: int) -> List[Tuple[int, int]]:
    if count < 1 or window < 1 or window > length:
        raise ValueError("Bad crop window/count")
    if count == 1:
        # A single independent sample should be central (not an accidental top-left crop).
        start = (length - window) // 2
        return [(start, start + window)]
    starts = [round((length - window) * i / (count - 1)) for i in range(count)]
    return [(int(x), int(x) + window) for x in starts]


def _coverage_grid(width: int, height: int, budget: int, min_source_cell: int) -> Tuple[int, int]:
    """Choose a near-square grid; all tiles combined cover original pixels."""
    options = []
    for rows in range(1, budget + 1):
        for cols in range(1, budget // rows + 1):
            n = rows * cols
            if n < 2:
                continue
            side_w, side_h = width / cols, height / rows
            worst_side = max(side_w, side_h)
            if worst_side < min_source_cell:
                continue
            shape_penalty = abs(math.log(side_w / side_h))
            options.append(((round(worst_side, 5), round(shape_penalty, 5), n), cols, rows))
    if not options:
        return (1, 1)
    _, cols, rows = min(options, key=lambda t: t[0])
    return (cols, rows)


def _coverage_ranges(length: int, count: int, overlap: float) -> List[Tuple[int, int]]:
    if count == 1:
        return [(0, length)]
    win = min(length, int(math.ceil(length / (count - (count - 1) * overlap))))
    return _start_windows(length, count, win)


def _native_grid(width: int, height: int, budget: int, tile_side: int) -> Tuple[int, int]:
    """For near-1:1 source pixel crops, maximize spatial coverage within budget.

    On large images with a small tile budget this samples, rather than fully
    covers, the image. Recorded boxes make this limitation inspectable.
    """
    max_cols = min(budget, max(1, math.ceil(width / tile_side)))
    max_rows = min(budget, max(1, math.ceil(height / tile_side)))
    options = []
    for rows in range(1, max_rows + 1):
        for cols in range(1, min(max_cols, budget // rows) + 1):
            n = rows * cols
            if n < 2:
                continue
            fx = min(1.0, cols * min(tile_side, width) / width)
            fy = min(1.0, rows * min(tile_side, height) / height)
            fraction = fx * fy
            spatial_balance = abs(math.log((width / cols) / (height / rows)))
            options.append(((-round(fraction, 9), round(spatial_balance, 5), -n), cols, rows))
    if not options:
        return (1, 1)
    _, cols, rows = min(options, key=lambda t: t[0])
    return (cols, rows)


def estimate_coverage_fraction(plan: Sequence[ViewInfo], width: int, height: int) -> float:
    """Exact rectangle union for cartesian grids of independent crop intervals."""
    locals_ = [v for v in plan if v.kind == "local"]
    if not locals_:
        return 0.0
    x_intervals = sorted(set((v.box_xyxy[0], v.box_xyxy[2]) for v in locals_))
    y_intervals = sorted(set((v.box_xyxy[1], v.box_xyxy[3]) for v in locals_))

    def union_length(intervals):
        end = -1
        total = 0
        for start, stop in intervals:
            total += max(0, stop - max(start, end))
            end = max(end, stop)
        return total

    return min(1.0, union_length(x_intervals) * union_length(y_intervals) / (width * height))


def make_view_plan(
    width: int,
    height: int,
    view_mode: str = "global-local",
    max_local_tiles: int = 8,
    tile_strategy: str = "native",
    tile_source_size: int = 448,
    tile_overlap: float = 0.10,
    tile_trigger_long_side: int = 672,
    tile_trigger_aspect: float = 2.0,
    min_tile_source_side: int = 256,
    min_local_short_side: int = 224,
) -> Tuple[List[ViewInfo], Tuple[int, int]]:
    """Plan a global view followed by optional independent local crops.

    native: choose up to `max_local_tiles` source-pixel crops of nominal size
      `tile_source_size`, distributed across source; NOT full coverage on 4K.
    coverage: tile the entire source with overlap but resize large crops to
      448; local quality-frequency cues can be altered by downsampling.
    """
    if width < 1 or height < 1:
        raise ValueError("Invalid source dimensions")
    if view_mode not in ("global", "global-local"):
        raise ValueError(f"Unknown view_mode: {view_mode}")
    if tile_strategy not in ("native", "coverage"):
        raise ValueError(f"Unknown tile_strategy: {tile_strategy}")
    global_view = ViewInfo("global", (0, 0, width, height))
    plan = [global_view]
    if view_mode == "global" or max_local_tiles < 2:
        return plan, (1, 1)
    if min(width, height) < min_local_short_side:
        return plan, (1, 1)
    aspect = max(width / height, height / width)
    if max(width, height) < tile_trigger_long_side and aspect < tile_trigger_aspect:
        return plan, (1, 1)

    if tile_strategy == "native":
        cols, rows = _native_grid(width, height, max_local_tiles, tile_source_size)
        x_ranges = _start_windows(width, cols, min(width, tile_source_size))
        y_ranges = _start_windows(height, rows, min(height, tile_source_size))
    else:
        cols, rows = _coverage_grid(width, height, max_local_tiles, min_tile_source_side)
        x_ranges = _coverage_ranges(width, cols, tile_overlap)
        y_ranges = _coverage_ranges(height, rows, tile_overlap)

    if cols * rows < 2:
        return plan, (1, 1)
    for ri, (y0, y1) in enumerate(y_ranges):
        for ci, (x0, x1) in enumerate(x_ranges):
            plan.append(ViewInfo("local", (x0, y0, x1, y1), row=ri, col=ci))
    return plan, (cols, rows)


def companion_path(csv_path: str, suffix: str) -> str:
    return str(Path(csv_path).with_suffix(suffix))


def build_save_csv_path(image_path: str, save_root: str, dataset_root: str) -> str:
    abs_img, abs_root = os.path.abspath(image_path), os.path.abspath(dataset_root)
    try:
        is_inside = os.path.commonpath([abs_img, abs_root]) == abs_root
    except ValueError:
        is_inside = False
    rel_dir = (os.path.dirname(os.path.relpath(abs_img, abs_root)) if is_inside
               else os.path.basename(os.path.dirname(abs_img)))
    target_dir = os.path.join(save_root, rel_dir)
    os.makedirs(target_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(target_dir, f"{stem}.csv")


def _source_fingerprint(image_path: str) -> Dict:
    st = os.stat(image_path)
    return {
        "source_image_path": os.path.abspath(image_path),
        "source_size_bytes": st.st_size,
        "source_mtime_ns": st.st_mtime_ns,
    }


def _hash_dict(d: Dict) -> str:
    raw = json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def get_global_settings(args) -> Dict:
    return {
        "version": CACHE_VERSION,
        "model_path": args.model_path,
        "global_preprocess": args.global_preprocess,
        "vision_size": VISION_SIZE,
        "exif_transpose": not args.ignore_exif,
        "quality_attributes": ATTRIBUTES,
        "preference_words": PREFERENCE_WORDS,
        "feature_definition": "hidden_states[-1,-2,:] + logits[-1,9 preference IDs]",
        "pad_color_rule": args.letterbox_pad if args.global_preprocess == "letterbox" else None,
    }


def get_local_settings(args, global_key: str) -> Dict:
    return {
        "version": CACHE_VERSION,
        "global_key": global_key,  # NPZ includes the global view as its first entry.
        "tile_strategy": args.tile_strategy,
        "tile_source_size": args.tile_source_size,
        "max_local_tiles": args.max_local_tiles,
        "tile_overlap": args.tile_overlap,
        "tile_trigger_long_side": args.tile_trigger_long_side,
        "tile_trigger_aspect": args.tile_trigger_aspect,
        "min_tile_source_side": args.min_tile_source_side,
        "min_local_short_side": args.min_local_short_side,
        "view_feature_dtype": args.view_feature_dtype,
        "local_preprocess": "letterbox_448",
    }


def is_cache_current(image_path: str, csv_path: str, cache_key: str, *, local=False) -> bool:
    output_path = companion_path(csv_path, ".views.npz") if local else csv_path
    meta_path = companion_path(csv_path, ".views.meta.json" if local else ".meta.json")
    if not (os.path.isfile(output_path) and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        fp = _source_fingerprint(image_path)
        return (meta.get("cache_key") == cache_key and
                all(meta.get(k) == v for k, v in fp.items()))
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def read_global_csv_features(csv_path: str) -> torch.Tensor:
    """Reload feature for NPZ view[0] when global CSV cache is already valid."""
    df = pd.read_csv(csv_path)
    if "attribute" not in df or df["attribute"].tolist() != ATTRIBUTES:
        raise RuntimeError(f"Bad attribute rows in global CSV: {csv_path}")
    expected = [f"combined_{i:06d}" for i in range(len(df.columns) - 1)]
    if df.columns[1:].tolist() != expected:
        raise RuntimeError(f"Bad feature columns in global CSV: {csv_path}")
    arr = df.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)
    if arr.ndim != 2 or arr.shape[0] != 8 or not np.isfinite(arr).all():
        raise RuntimeError(f"Invalid global feature values: {csv_path}")
    return torch.from_numpy(arr)


def save_global_csv(image_path: str, features: torch.Tensor, csv_path: str) -> None:
    arr = features.detach().float().cpu().numpy()
    if arr.ndim != 2 or arr.shape[0] != len(ATTRIBUTES):
        raise ValueError(f"Expected [8,D+9], got {arr.shape}")
    cols = [f"combined_{i:06d}" for i in range(arr.shape[1])]
    df = pd.DataFrame(arr, columns=cols)
    df.insert(0, "attribute", ATTRIBUTES)
    tmp = csv_path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig", float_format="%.8f")
    os.replace(tmp, csv_path)


def _write_json_atomic(path: str, obj: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def save_global_meta(
    image_path: str, csv_path: str, global_settings: Dict, global_key: str,
    dimensions: Tuple[int, int], feature_shape: Sequence[int],
) -> None:
    data = {
        "cache_key": global_key,
        **_source_fingerprint(image_path),
        "source_image_size_wh": list(dimensions),
        "global_settings": global_settings,
        "csv_feature_shape": list(feature_shape),
        "csv_semantics": "GLOBAL_ONLY; no local feature fusion",
    }
    _write_json_atomic(companion_path(csv_path, ".meta.json"), data)


def save_views_npz(
    csv_path: str,
    global_feat: torch.Tensor,
    local_feat: torch.Tensor,
    plan: Sequence[ViewInfo],
    dimensions: Tuple[int, int],
    grid: Tuple[int, int],
    dtype: str,
) -> None:
    if len(plan) != local_feat.shape[0] + 1:
        raise ValueError("Mismatch in local feature and view plan lengths")
    joined = torch.cat([global_feat.unsqueeze(0), local_feat], dim=0).float().cpu().numpy()
    joined = joined.astype(np.float32 if dtype == "float32" else np.float16)
    out = companion_path(csv_path, ".views.npz")
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f,
            features=joined,
            boxes_xyxy=np.array([v.box_xyxy for v in plan], dtype=np.int32),
            is_global=np.array([v.kind == "global" for v in plan], dtype=np.bool_),
            row_col=np.array([[v.row, v.col] for v in plan], dtype=np.int16),
            image_size_wh=np.array(dimensions, dtype=np.int32),
            grid_cols_rows=np.array(grid, dtype=np.int16),
            attributes=np.array(ATTRIBUTES, dtype="U16"),
        )
    os.replace(tmp, out)


def save_views_meta(
    image_path: str, csv_path: str, settings: Dict, key: str,
    plan: Sequence[ViewInfo], dimensions: Tuple[int, int], grid: Tuple[int, int],
) -> None:
    meta = {
        "cache_key": key,
        **_source_fingerprint(image_path),
        "source_image_size_wh": list(dimensions),
        "tile_settings": settings,
        "num_views": len(plan),
        "num_local_views": len(plan) - 1,
        "grid_cols_rows": list(grid),
        "source_coverage_fraction_of_local_views": round(
            estimate_coverage_fraction(plan, *dimensions), 6),
        "views": [asdict(v) for v in plan],
        "semantics": "Independent per-view quality features; NOT fused with global CSV",
    }
    _write_json_atomic(companion_path(csv_path, ".views.meta.json"), meta)


def gather_image_paths(dataset_root: str, include_roots: Optional[Sequence[str]]) -> List[str]:
    if include_roots:
        patterns = [os.path.join(dataset_root, rel, "**", "*") for rel in include_roots]
    else:
        patterns = [
            os.path.join(dataset_root, "Real_world", "AlignFormer", "iphone_dataset", "test", "input", "*.png"),
            os.path.join(dataset_root, "Real_world", "AlignFormer", "iphone_dataset", "test", "target", "*.png"),
        ]
    found = []
    for pattern in patterns:
        found.extend(glob.glob(pattern, recursive=True))
    valid = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted({os.path.abspath(p) for p in found
                   if os.path.isfile(p) and Path(p).suffix.lower() in valid})


class QInstructScorer(nn.Module):
    """Never fuses global and local Hidden State / Logits."""

    def __init__(
        self, device: str = "cuda:0", model_path: str = "q-future",
        letterbox_pad: str = "original",
    ):
        super().__init__()
        # Lazy imports keep --plan-only usable without the heavy model dependency.
        from mplug_owl2.model.builder import load_pretrained_model
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX
        from mplug_owl2.mm_utils import tokenizer_image_token

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, "mplug_owl2", device=device
        )
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.model.eval()
        self.pad_color = ((123, 117, 104) if letterbox_pad == "original"
                          else mean_pad_color(image_processor))

        self.att_list = list(ATTRIBUTES)
        self.pref_words = list(PREFERENCE_WORDS)
        # Keep the original user's ID selection, verified in their runtime.
        self.preferential_ids_ = [ids[1] for ids in tokenizer(self.pref_words)["input_ids"]]

        self.full_prompt_ids = []
        for attribute in self.att_list:
            full_text = f"USER: <|image|> Rate the {attribute} quality of the image.\nASSISTANT: "
            tokens = tokenizer_image_token(
                full_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(model.device)
            self.full_prompt_ids.append(tokens)
        self.common_prefix_len = longest_common_prefix_len(self.full_prompt_ids)
        self.prefix_ids = self.full_prompt_ids[0][:, :self.common_prefix_len]
        self.tail_ids_list = [ids[:, self.common_prefix_len:] for ids in self.full_prompt_ids]

    @staticmethod
    def _extract_last_hidden(outputs):
        hidden = outputs.hidden_states
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[-1]
        return hidden[:, -2, :]

    def _process_images(
        self, images: Sequence[Image.Image],
        *, already_letterboxed: bool,
    ) -> torch.Tensor:
        preprocess = getattr(self.image_processor, "preprocess", self.image_processor)
        if not already_letterboxed:
            # Official processor baseline. Model must itself produce 448x448.
            encoded = preprocess(images, return_tensors="pt")
        else:
            # Avoid re-resize/center-crop of the already prepared 448x448 view.
            try:
                encoded = preprocess(
                    images, return_tensors="pt", do_resize=False, do_center_crop=False
                )
            except TypeError as err:
                if "do_resize" not in str(err) and "do_center_crop" not in str(err):
                    raise
                old_resize = getattr(self.image_processor, "do_resize", None)
                old_crop = getattr(self.image_processor, "do_center_crop", None)
                if old_resize is None or old_crop is None:
                    raise RuntimeError("Cannot safely disable image resize/center-crop") from err
                try:
                    self.image_processor.do_resize = False
                    self.image_processor.do_center_crop = False
                    encoded = preprocess(images, return_tensors="pt")
                finally:
                    self.image_processor.do_resize = old_resize
                    self.image_processor.do_center_crop = old_crop
        tensor = encoded["pixel_values"]
        if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (VISION_SIZE, VISION_SIZE):
            raise ValueError(
                f"Q-Instruct expected {VISION_SIZE}x{VISION_SIZE} from processor, got {tuple(tensor.shape)}"
            )
        return tensor.to(device=self.model.device, dtype=self.model.dtype)

    def _build_prefix_cache(self, image_tensor: torch.Tensor):
        return self.model(
            input_ids=self.prefix_ids.repeat(image_tensor.shape[0], 1),
            images=image_tensor,
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,
        ).past_key_values

    @torch.inference_mode()
    def _infer_tensor_combined(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Preserve the user's original verified KV-cache extraction path."""
        bsz = image_tensor.shape[0]
        prefix_cache = self._build_prefix_cache(image_tensor)
        combined_list = []
        for tail_ids in self.tail_ids_list:
            cache_out = self.model(
                input_ids=tail_ids.repeat(bsz, 1),
                past_key_values=prefix_cache,
                images=None,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            )
            combined = torch.cat([
                self._extract_last_hidden(cache_out),
                cache_out.logits[:, -1, self.preferential_ids_],
            ], dim=-1)
            combined_list.append(combined.detach().float().cpu())
            del cache_out
        del prefix_cache
        return torch.stack(combined_list, dim=1)  # CPU float32 [B, 8, D+9]

    @torch.inference_mode()
    def infer_global_batch(
        self, image_paths: Sequence[str], *,
        global_preprocess: str = "letterbox",
        exif_transpose: bool = True,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Batched GLOBAL inference just like original --batch-size=32 workflow."""
        if not image_paths:
            raise ValueError("No global images to infer")
        pil_views = []
        source_dims = []
        for path in image_paths:
            original = read_image_rgb(path, exif_transpose=exif_transpose)
            source_dims.append(original.size)
            if global_preprocess == "letterbox":
                pil_views.append(letterbox_for_model(original, VISION_SIZE, self.pad_color))
            elif global_preprocess == "processor":
                pil_views.append(original)
            else:
                raise ValueError(f"Unknown global preprocessing: {global_preprocess}")
        pixels = self._process_images(
            pil_views, already_letterboxed=(global_preprocess == "letterbox")
        )
        features = self._infer_tensor_combined(pixels)
        del pixels, pil_views
        return features, source_dims

    @torch.inference_mode()
    def infer_local_views(
        self, source: Image.Image, local_plan: Sequence[ViewInfo],
        view_batch_size: int = 2,
    ) -> torch.Tensor:
        """Independent local inputs; their results are NEVER averaged into CSV."""
        if not local_plan:
            raise ValueError("Use an empty tensor at caller for zero local views")
        results = []
        for i in range(0, len(local_plan), view_batch_size):
            mini = local_plan[i:i + view_batch_size]
            imgs = [letterbox_for_model(source.crop(v.box_xyxy), VISION_SIZE, self.pad_color)
                    for v in mini]
            pixels = self._process_images(imgs, already_letterboxed=True)
            results.append(self._infer_tensor_combined(pixels))
            del imgs, pixels
        return torch.cat(results, dim=0)


def evaluate_dataset(
    scorer: QInstructScorer,
    paths: Sequence[str],
    args,
    global_settings: Dict,
    global_key: str,
    local_settings: Optional[Dict],
    local_key: Optional[str],
) -> None:
    """Global and local are cached independently; no accidental fused CSV."""
    os.makedirs(args.context_root, exist_ok=True)
    progress = tqdm(total=len(paths), desc="Q-Instruct", unit="img", dynamic_ncols=True)
    try:
        for begin in range(0, len(paths), args.batch_size):
            batch_paths = paths[begin:begin + args.batch_size]
            csv_paths = [build_save_csv_path(p, args.context_root, args.dataset_root)
                         for p in batch_paths]
            require_views = args.view_mode == "global-local" and args.save_view_features
            need_global = [args.overwrite or not is_cache_current(p, c, global_key)
                           for p, c in zip(batch_paths, csv_paths)]
            need_views = [
                require_views and (args.overwrite or not is_cache_current(p, c, local_key, local=True))
                for p, c in zip(batch_paths, csv_paths)
            ]
            if args.only_missing and not args.overwrite and not any(need_global) and not any(need_views):
                progress.update(len(batch_paths))
                continue

            # In the normal extraction mode (no only-missing flag) refresh
            # requested outputs, retaining the original CLI semantics.
            if not args.only_missing and not args.overwrite:
                need_global = [True] * len(batch_paths)
                need_views = [require_views] * len(batch_paths)

            fresh_global: Dict[int, torch.Tensor] = {}
            missing_indices = [i for i, flag in enumerate(need_global) if flag]
            if missing_indices:
                subset = [batch_paths[i] for i in missing_indices]
                global_features, dimensions = scorer.infer_global_batch(
                    subset,
                    global_preprocess=args.global_preprocess,
                    exif_transpose=not args.ignore_exif,
                )
                for sub_i, original_i in enumerate(missing_indices):
                    feature = global_features[sub_i]
                    fresh_global[original_i] = feature
                    save_global_csv(batch_paths[original_i], feature, csv_paths[original_i])
                    save_global_meta(
                        batch_paths[original_i], csv_paths[original_i], global_settings,
                        global_key, dimensions[sub_i], feature.shape,
                    )

            for i, (image_path, csv_path) in enumerate(zip(batch_paths, csv_paths)):
                if need_views[i]:
                    # Includes the GLOBAL view feature first, from same run or
                    # from a validated global CSV. No second global VLM call.
                    global_feature = (fresh_global[i] if i in fresh_global
                                      else read_global_csv_features(csv_path))
                    image = read_image_rgb(image_path, exif_transpose=not args.ignore_exif)
                    dims = image.size
                    plan, grid = make_view_plan(
                        *dims,
                        view_mode="global-local",
                        max_local_tiles=args.max_local_tiles,
                        tile_strategy=args.tile_strategy,
                        tile_source_size=args.tile_source_size,
                        tile_overlap=args.tile_overlap,
                        tile_trigger_long_side=args.tile_trigger_long_side,
                        tile_trigger_aspect=args.tile_trigger_aspect,
                        min_tile_source_side=args.min_tile_source_side,
                        min_local_short_side=args.min_local_short_side,
                    )
                    if len(plan) > 1:
                        local_feats = scorer.infer_local_views(
                            image, plan[1:], view_batch_size=args.view_batch_size)
                    else:
                        local_feats = global_feature.new_empty((0, *global_feature.shape))
                    save_views_npz(
                        csv_path, global_feature, local_feats, plan, dims, grid,
                        dtype=args.view_feature_dtype,
                    )
                    save_views_meta(
                        image_path, csv_path, local_settings, local_key, plan, dims, grid
                    )
                    del image, local_feats
                progress.update(1)
    finally:
        progress.close()


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract stable Q-Instruct GLOBAL quality features with optional local evidence."
    )
    p.add_argument("--dataset-root", default="./datasets")
    p.add_argument("--context-root", default="./q_instruct_results_global_primary")
    p.add_argument("--include", nargs="+", default=None,
                   help="Dataset-relative directories (searched recursively).")
    p.add_argument("--model-path", default="q-future",
                   help="Use your original Q-Instruct checkpoint or local model path.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Actual batched GLOBAL inference count; your previous 32 is supported.")
    p.add_argument("--view-batch-size", type=int, default=2,
                   help="Actual GPU microbatch for optional independent LOCAL views.")
    p.add_argument("--global-preprocess", choices=["letterbox", "processor"],
                   default="letterbox", help="letterbox reproduces original global strategy;")
    p.add_argument("--letterbox-pad", choices=["original", "processor-mean"],
                   default="original",
                   help="original uses exact (123,117,104) RGB from legacy script.")
    # Legacy script used Image.open(...).convert("RGB") without EXIF transpose.
    # Keep that behavior as the default for feature reproducibility; correct
    # orientation only in an explicitly requested experimental condition.
    exif_group = p.add_mutually_exclusive_group()
    exif_group.add_argument("--apply-exif", dest="ignore_exif", action="store_false",
                            help="Apply EXIF orientation (changes inputs for rotated JPEGs).")
    exif_group.add_argument("--ignore-exif", dest="ignore_exif", action="store_true",
                            help="Match original script: do not apply EXIF transpose (default).")
    p.set_defaults(ignore_exif=True)
    p.add_argument("--view-mode", choices=["global", "global-local"], default="global",
                   help="Default global-only is safest; local requires explicit opt-in.")
    p.add_argument("--save-view-features", action="store_true",
                   help="Save independent per-view features/boxes to .views.npz.")
    p.add_argument("--fusion-mode", choices=["global"], default="global",
                   help="Compatibility alias. Only GLOBAL CSV is supported; no uncalibrated fusion.")
    p.add_argument("--tile-strategy", choices=["native", "coverage"], default="native",
                   help="native: near-source-resolution samples; coverage: fully cover but may resize.")
    p.add_argument("--max-local-tiles", type=int, default=8)
    p.add_argument("--tile-source-size", type=int, default=448,
                   help="Native crop side in ORIGINAL pixels, before 448 input.")
    p.add_argument("--tile-overlap", type=float, default=0.10,
                   help="Applies to coverage tiling only, not native sampling.")
    p.add_argument("--tile-trigger-long-side", type=int, default=672)
    p.add_argument("--tile-trigger-aspect", type=float, default=2.0)
    p.add_argument("--min-tile-source-side", type=int, default=256,
                   help="Coverage strategy: avoid over-splitting source regions.")
    p.add_argument("--min-local-short-side", type=int, default=224,
                   help="Skip local inference if the shorter source side is too small.")
    p.add_argument("--view-feature-dtype", choices=["float32", "float16"], default="float32",
                   help="NPZ storage precision; CSV remains global float32 independently.")
    p.add_argument("--only-missing", action="store_true",
                   help="Reuse global CSV and/or view NPZ only if independently validated.")
    p.add_argument("--overwrite", action="store_true",
                   help="Force regeneration of all requested outputs.")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--plan-only", action="store_true",
                   help="Inspect tile boxes and estimated coverage without loading Q-Instruct.")
    return p


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.batch_size < 1 or args.view_batch_size < 1:
        parser.error("Batch sizes must be >= 1")
    if args.max_local_tiles < 0 or args.tile_source_size < 1:
        parser.error("Tile count must be nonnegative and --tile-source-size >= 1")
    if not 0 <= args.tile_overlap < 0.5:
        parser.error("--tile-overlap must lie in [0,0.5)")
    if args.tile_trigger_aspect < 1 or args.tile_trigger_long_side < 1:
        parser.error("Tile triggers must be positive; aspect must be >= 1")
    if args.min_local_short_side < 1 or args.min_tile_source_side < 1:
        parser.error("Minimum tile source dimensions must be >= 1")
    if args.view_mode == "global-local" and not args.save_view_features and not args.plan_only:
        parser.error("--view-mode global-local requires --save-view-features; local features must not be discarded")
    if args.view_mode == "global" and args.save_view_features:
        parser.error("Use --view-mode global-local with --save-view-features, or remove --save-view-features")

    global_settings = get_global_settings(args)
    global_key = _hash_dict(global_settings)
    local_settings = (get_local_settings(args, global_key)
                      if args.view_mode == "global-local" else None)
    local_key = _hash_dict(local_settings) if local_settings else None

    all_paths = gather_image_paths(args.dataset_root, args.include)
    if args.only_missing and not args.overwrite and not args.plan_only:
        all_paths = [
            p for p in all_paths if (
                not is_cache_current(
                    p, build_save_csv_path(p, args.context_root, args.dataset_root), global_key)
                or (
                    args.view_mode == "global-local" and
                    not is_cache_current(
                        p, build_save_csv_path(p, args.context_root, args.dataset_root),
                        local_key, local=True)
                )
            )
        ]
    if args.max_images > 0:
        all_paths = all_paths[:args.max_images]
    print(f"Found {len(all_paths)} images; global_key={global_key}; local_key={local_key}")
    print("Main CSV contains GLOBAL features only; no Hidden State / Logit fusion.")
    if not all_paths:
        return

    # Detect collisions from image.png and image.jpg sharing the same stem.
    # Both would otherwise silently overwrite the same CSV path.
    output_to_source = {}
    for source in all_paths:
        output = build_save_csv_path(source, args.context_root, args.dataset_root)
        old = output_to_source.get(output)
        if old is not None and old != source:
            raise RuntimeError(f"Output collision for '{output}': '{old}' vs '{source}'")
        output_to_source[output] = source

    if args.plan_only:
        for image_path in all_paths:
            with Image.open(image_path) as raw:
                image = ImageOps.exif_transpose(raw) if not args.ignore_exif else raw
                width, height = image.size
            plan, grid = make_view_plan(
                width, height, view_mode=args.view_mode,
                max_local_tiles=args.max_local_tiles,
                tile_strategy=args.tile_strategy,
                tile_source_size=args.tile_source_size,
                tile_overlap=args.tile_overlap,
                tile_trigger_long_side=args.tile_trigger_long_side,
                tile_trigger_aspect=args.tile_trigger_aspect,
                min_tile_source_side=args.min_tile_source_side,
                min_local_short_side=args.min_local_short_side,
            )
            coverage = estimate_coverage_fraction(plan, width, height)
            print(f"{image_path}: {width}x{height}, grid={grid}, local={len(plan)-1}, coverage={coverage:.1%}")
            for i, v in enumerate(plan):
                print(f"   {i:2d}: {v.kind:6} {v.box_xyxy} row={v.row} col={v.col}")
        return

    scorer = QInstructScorer(
        device=args.device, model_path=args.model_path, letterbox_pad=args.letterbox_pad
    )
    evaluate_dataset(
        scorer, all_paths, args, global_settings, global_key,
        local_settings, local_key,
    )


if __name__ == "__main__":
    main()
