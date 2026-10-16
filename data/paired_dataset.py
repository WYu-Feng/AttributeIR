from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


CANONICAL_ATTRS = [
    "Brightness",
    "Contrast",
    "Color",
    "Sharpness",
    "Structure",
    "Texture",
    "Noise",
    "Artifact",
]

PREF_WORDS = [
    "good",
    "average",
    "poor",
    "high",
    "medium",
    "low",
    "fine",
    "acceptable",
    "bad",
]

IMAGE_EXTS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
DEFAULT_CONTEXT_ROOT = str(Path(__file__).resolve().parents[1] / "q_instruct_results")

# Tasks in the original five degradation categories are balanced to 10000 samples.
# Other auxiliary / OOD-style tasks are balanced to 3000 samples.
MAJOR_BALANCE_TASKS = ("derain", "dehaze", "deblur", "light", "noise")
MAJOR_TARGET_PER_TASK = 10000
OTHER_TARGET_PER_TASK = 3000


def parse_task_list(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        return tuple(x.strip().lower() for x in value.split(",") if x.strip())
    return tuple(str(x).strip().lower() for x in value if str(x).strip())


def _glob_images(root: str, parts: Sequence[str], exts: Sequence[str] = IMAGE_EXTS, limit: int = 0) -> List[str]:
    paths: List[str] = []
    base = os.path.join(root, *parts)
    for ext in exts:
        paths.extend(glob.glob(os.path.join(base, f"*.{ext}")))
        paths.extend(glob.glob(os.path.join(base, f"*.{ext.upper()}")))
    paths = sorted(set(paths))
    return paths[:limit] if limit and limit > 0 else paths


def _basename_key(path: str) -> str:
    return os.path.basename(path).lower()


def pair_by_basename(a_paths: Sequence[str], b_paths: Sequence[str]) -> List[Tuple[str, str]]:
    a_map = {_basename_key(p): p for p in a_paths}
    b_map = {_basename_key(p): p for p in b_paths}
    keys = sorted(set(a_map) & set(b_map))
    return [(a_map[k], b_map[k]) for k in keys]


def balance_pairs(pairs: Sequence[Tuple[str, str]], target: int) -> List[Tuple[str, str]]:
    pairs = list(pairs)
    n = len(pairs)
    target = int(target)
    if n == 0 or target <= 0 or n == target:
        return pairs
    if n > target:
        idx = random.sample(range(n), target)
        return [pairs[i] for i in idx]
    k = target // n
    r = target % n
    return pairs * k + random.choices(pairs, k=r)


@dataclass(frozen=True)
class PairGroup:
    task: str
    input_parts: Tuple[str, ...]
    target_parts: Tuple[str, ...]
    limit: int = 0
    exts: Tuple[str, ...] = IMAGE_EXTS

    def read(self, root: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
        a_paths = _glob_images(root, self.input_parts, self.exts, self.limit)
        b_paths = _glob_images(root, self.target_parts, self.exts, self.limit)
        return a_paths, b_paths, pair_by_basename(a_paths, b_paths)


def _g(task: str, input_parts: Sequence[str], target_parts: Sequence[str], limit: int = 0, exts: Sequence[str] = IMAGE_EXTS) -> PairGroup:
    return PairGroup(task, tuple(input_parts), tuple(target_parts), int(limit), tuple(exts))


def _bsd400_wed_groups(task: str = "bsd400_wed_denoise") -> List[PairGroup]:
    groups: List[PairGroup] = []
    for dataset_name in ("BSD400", "WED"):
        for sigma in (15, 25, 50):
            groups.append(
                _g(
                    task,
                    ("Image denoise", dataset_name, f"noisy{sigma}"),
                    ("Image denoise", dataset_name, "original"),
                )
            )
            groups.append(
                _g(
                    task,
                    ("Image denoise", "BSD400_WED", f"noisy{sigma}", dataset_name),
                    ("Image denoise", "BSD400_WED", "clean", dataset_name),
                )
            )
    return groups


TASK_GROUPS: Dict[str, Tuple[PairGroup, ...]] = {
    "derain": (
        _g("derain", ("Image deraining", "Rain100L", "input"), ("Image deraining", "Rain100L", "target")),
    ),
    "derain_extra": (
        _g("derain_extra", ("Image deraining", "RainTrainL", "input"), ("Image deraining", "RainTrainL", "target")),
    ),
    "dehaze": (
        _g("dehaze", ("Image dehazing", "SOTS", "outdoor", "hazy"), ("Image dehazing", "SOTS", "outdoor", "gt")),
    ),
    "deblur": (
        _g("deblur", ("Image deblurring", "GoPro", "train", "input"), ("Image deblurring", "GoPro", "train", "target")),
        _g("deblur", ("Image deblurring", "GoPro", "test", "input"), ("Image deblurring", "GoPro", "test", "target")),
    ),
    "light": (
        _g("light", ("Low-light enhancement", "LOL", "train", "low"), ("Low-light enhancement", "LOL", "train", "high")),
        _g("light", ("Low-light enhancement", "LOL", "eval", "low"), ("Low-light enhancement", "LOL", "eval", "high")),
    ),
    "noise": (
        _g("noise", ("Image denoise", "BSD68", "noisy25"), ("Image denoise", "BSD68", "original")),
    ),
    "noise_extra": (
        _g("noise_extra", ("Image denoise", "Urban100", "2", "input"), ("Image denoise", "Urban100", "2", "target")),
        _g("noise_extra", ("Image denoise", "BSD400", "noisy25"), ("Image denoise", "BSD400", "original")),
    ),
    "test2800": (
        _g("test2800", ("Image deraining", "Test2800", "input"), ("Image deraining", "Test2800", "target")),
    ),
    "rain_mist": (
        _g("rain_mist", ("Real_world", "Rain Mist"), ("Real_world", "Rain Mist"), exts=("jpg", "jpeg", "png")),
    ),
    "hide": (
        _g("hide", ("Image deblurring", "HIDE", "input"), ("Image deblurring", "HIDE", "target")),
    ),
    "alignformer": (
        _g(
            "alignformer",
            ("Real_world", "AlignFormer", "iphone_dataset", "train", "input"),
            ("Real_world", "AlignFormer", "iphone_dataset", "train", "target")
        ),
    ),
    "dpdd": (
        _g("dpdd", ("Image deblurring", "DPDD", "input"), ("Image deblurring", "DPDD", "target")),
        _g("dpdd", ("Defocus deblurring", "DPDD", "input"), ("Defocus deblurring", "DPDD", "target")),
    ),
    "jpeg_bsd500": (
        _g("jpeg_bsd500", ("JPEG", "BSD500", "train", "input"), ("JPEG", "BSD500", "train", "target")),
    ),
    "bsd500_qf10": (
        _g("bsd500_qf10", ("JPEG", "BSD500", "test", "input"), ("JPEG", "BSD500", "test", "target")),
        _g("bsd500_qf10", ("JPEG", "BSD500", "input"), ("JPEG", "BSD500", "target")),
    ),
    "udc_ir": (
        _g("udc_ir", ("Real_world", "UDC-IR", "input"), ("Real_world", "UDC-IR", "target")),
        _g("udc_ir", ("UnderDisplayCamera", "UDC-IR", "input"), ("UnderDisplayCamera", "UDC-IR", "target")),
    ),
    "uieb": (
        _g("uieb", ("Underwater", "UIEB", "input"), ("Underwater", "UIEB", "target")),
    ),
}

DEFAULT_TRAIN_TASKS = (
    "derain",
    "dehaze",
    "deblur",
    "light",
    "noise",
    "test2800",
    "rain_mist",
    "hide",
    "alignformer",
    "jpeg_bsd500",
    "uieb",
)


def available_tasks() -> Tuple[str, ...]:
    return tuple(sorted(TASK_GROUPS))


def _read_task_pairs(root: str, task: str) -> Tuple[List[Tuple[str, str]], int, int]:
    if task not in TASK_GROUPS:
        raise ValueError(f"Unknown task '{task}'. Available tasks: {', '.join(available_tasks())}")
    all_pairs: List[Tuple[str, str]] = []
    total_a = 0
    total_b = 0
    seen_pairs = set()
    for group in TASK_GROUPS[task]:
        a_paths, b_paths, pairs = group.read(root)
        total_a += len(a_paths)
        total_b += len(b_paths)
        for pair in pairs:
            if pair not in seen_pairs:
                all_pairs.append(pair)
                seen_pairs.add(pair)
    return all_pairs, total_a, total_b


def _balance_target_for_task(
    task: str,
    major_target_per_task: int = MAJOR_TARGET_PER_TASK,
    other_target_per_task: int = OTHER_TARGET_PER_TASK,
    major_tasks: Sequence[str] = MAJOR_BALANCE_TASKS,
) -> int:
    major_task_set = set(parse_task_list(major_tasks))
    if task in major_task_set:
        return int(major_target_per_task)
    return int(other_target_per_task)


def build_pairs_for_tasks(
    root: str = "./datasets",
    tasks: Sequence[str] = DEFAULT_TRAIN_TASKS,
    target_per_task: int = MAJOR_TARGET_PER_TASK,
    other_target_per_task: int = OTHER_TARGET_PER_TASK,
    major_tasks: Sequence[str] = MAJOR_BALANCE_TASKS,
    balance: bool = False,
) -> List[Tuple[str, str, str]]:
    samples: List[Tuple[str, str, str]] = []
    for task in parse_task_list(tasks):
        pairs, total_a, total_b = _read_task_pairs(root, task)
        raw_count = len(pairs)

        task_target = 0
        if balance and int(target_per_task) > 0:
            task_target = _balance_target_for_task(
                task,
                major_target_per_task=target_per_task,
                other_target_per_task=other_target_per_task,
                major_tasks=major_tasks,
            )
            pairs = balance_pairs(pairs, task_target)
        else:
            pairs = list(pairs)

        samples.extend((a, b, task) for a, b in pairs)
        balance_note = f" balanced={len(pairs)} target={task_target}" if task_target > 0 else ""
        print(f"[DATA] {task}: A={total_a} B={total_b} paired={raw_count}{balance_note}")
    print(f"[DATA] Total paired samples: {len(samples)}")
    if not samples:
        raise RuntimeError("[DATA] No paired samples found. Check dataset paths and task names.")
    return samples


def build_image_manifest(
    root: str = "./datasets",
    split: str = "train",
    tasks: Sequence[str] = DEFAULT_TRAIN_TASKS,
    include_targets: bool = True,
) -> List[str]:
    del split
    paths: List[str] = []
    seen = set()
    for deg_path, target_path, _task in build_pairs_for_tasks(root=root, tasks=tasks, target_per_task=0, balance=False):
        for path in (deg_path, target_path) if include_targets else (deg_path,):
            norm = os.path.normpath(path)
            if norm not in seen and os.path.isfile(norm):
                paths.append(norm)
                seen.add(norm)
    return sorted(paths)


class CsvAttributeContextMixin:
    CANONICAL_ATTRS = CANONICAL_ATTRS
    PREF_WORDS = PREF_WORDS
    POS_IDXS = [0, 6]
    MID_IDXS = [1, 7]
    NEG_IDXS = [2, 8]

    @staticmethod
    def build_csv_path(image_path: str, save_root: str, dataset_root: str = "./datasets") -> str:
        abs_img = os.path.abspath(image_path)
        abs_dataset_root = os.path.abspath(dataset_root)
        try:
            common = os.path.commonpath([abs_img, abs_dataset_root])
        except ValueError:
            common = ""
        if common == abs_dataset_root:
            rel_path = os.path.relpath(abs_img, abs_dataset_root)
            rel_dir = os.path.dirname(rel_path)
        else:
            rel_dir = os.path.basename(os.path.dirname(abs_img))
        stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(save_root, rel_dir, f"{stem}.csv")

    def _read_attr_csv(self, csv_path: str) -> torch.Tensor:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[ATTR CSV] Not found: {csv_path}\n"
                f"Generate it with extract_attribute_context.py or model_score_getdataset.py first."
            )

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "attribute" not in df.columns:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "attribute"})

        df["attribute"] = df["attribute"].astype(str)
        present = df["attribute"].tolist()
        missing = [a for a in self.CANONICAL_ATTRS if a not in present]
        if missing:
            raise ValueError(f"[ATTR CSV] Missing attributes {missing} in {csv_path}")

        df = df.set_index("attribute").loc[self.CANONICAL_ATTRS].reset_index()
        value_cols = [c for c in df.columns if c != "attribute"]
        arr = df[value_cols].to_numpy(dtype="float32")

        if arr.shape[0] != len(self.CANONICAL_ATTRS):
            raise ValueError(f"[ATTR CSV] Expected 8 rows, got {arr.shape[0]} in {csv_path}")

        expected_total_dim = self.token_dim + self.pref_logit_dim
        if arr.shape[1] != expected_total_dim:
            raise ValueError(
                f"[ATTR CSV] Expected row dim={expected_total_dim} "
                f"(token_dim={self.token_dim}, pref_logit_dim={self.pref_logit_dim}), "
                f"but got {arr.shape[1]} in {csv_path}"
            )
        return torch.from_numpy(arr)

    def _pref_logits_to_scores(self, logits_9: torch.Tensor) -> torch.Tensor:
        sel = logits_9[:, self.POS_IDXS + self.MID_IDXS + self.NEG_IDXS]
        prob = torch.softmax(sel, dim=-1)
        pos = prob[:, 0:2].sum(dim=-1)
        mid = prob[:, 2:4].sum(dim=-1)
        return pos + 0.5 * mid

    def _csv_to_context(self, csv_tensor: torch.Tensor) -> torch.Tensor:
        attr_tokens = csv_tensor[:, : self.token_dim]
        pref_logits = csv_tensor[:, self.token_dim :]
        attr_scores = self._pref_logits_to_scores(pref_logits)
        attr_scores = attr_scores.unsqueeze(0).expand(attr_tokens.size(0), -1)
        return torch.cat([attr_tokens, attr_scores], dim=1).float()

    def load_context_from_image_path(self, image_path: str, info_root: str) -> torch.Tensor:
        csv_path = self.build_csv_path(image_path, info_root, dataset_root=self.root)
        csv_tensor = self._read_attr_csv(csv_path)
        return self._csv_to_context(csv_tensor)


class _PairTransformMixin:
    def _maybe_resize_if_small(self, img: Image.Image, size: int) -> Image.Image:
        if size <= 0:
            return img
        w, h = img.size
        if h < size or w < size:
            img = transforms.Resize([size, size], interpolation=InterpolationMode.BICUBIC)(img)
        return img

    @staticmethod
    def _normalize_tensor(x: torch.Tensor, normalize_to: str) -> torch.Tensor:
        if normalize_to == "minus1_1":
            return x * 2.0 - 1.0
        if normalize_to == "zero1":
            return x
        raise ValueError(f"Unsupported normalize_to: {normalize_to}")

    @staticmethod
    def _align_size(condition: Image.Image, target: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if condition.size == target.size:
            return condition, target
        w1, h1 = condition.size
        w2, h2 = target.size
        ww, hh = min(w1, w2), min(h1, h2)
        return TF.center_crop(condition, [hh, ww]), TF.center_crop(target, [hh, ww])


class AlignedDataset(data.Dataset, CsvAttributeContextMixin, _PairTransformMixin):
    """
    Training dataset following the per-task balancing strategy from the paper.
    CSV contexts are loaded by default for offline mode; ``load_contexts=False``
    returns source paths so the training loop can generate Q-Instruct features
    online for the complete images.
    """

    def __init__(
        self,
        root: str = "./datasets",
        mode: str = "train",
        train_size: int = 192,
        augment_flip: bool = True,
        target_per_task: int = MAJOR_TARGET_PER_TASK,
        other_target_per_task: int = OTHER_TARGET_PER_TASK,
        major_tasks: Sequence[str] = MAJOR_BALANCE_TASKS,
        use_tasks: Sequence[str] = DEFAULT_TRAIN_TASKS,
        normalize_to: str = "minus1_1",
        include_indoor_dehaze: bool = True,
        exts: Sequence[str] = IMAGE_EXTS,
        eval_info_root: str = DEFAULT_CONTEXT_ROOT,
        deg_eval_info_root: Optional[str] = None,
        clean_eval_info_root: Optional[str] = None,
        load_contexts: bool = True,
        return_paths: bool = False,
        token_dim: int = 4096,
        pref_logit_dim: int = 9,
    ):
        super().__init__()
        del include_indoor_dehaze, exts
        self.root = root
        self.mode = mode
        self.train_size = int(train_size)
        self.augment_flip = augment_flip
        self.target_per_task = int(target_per_task)
        self.other_target_per_task = int(other_target_per_task)
        self.major_tasks = parse_task_list(major_tasks)
        self.use_tasks = parse_task_list(use_tasks)
        self.normalize_to = normalize_to
        self.eval_info_root = eval_info_root
        self.deg_eval_info_root = deg_eval_info_root or eval_info_root
        self.clean_eval_info_root = clean_eval_info_root or eval_info_root
        self.load_contexts = bool(load_contexts)
        self.return_paths = bool(return_paths)
        self.token_dim = int(token_dim)
        self.pref_logit_dim = int(pref_logit_dim)
        self.to_tensor = transforms.ToTensor()

        self.pairs = build_pairs_for_tasks(
            root=self.root,
            tasks=self.use_tasks,
            target_per_task=self.target_per_task,
            other_target_per_task=self.other_target_per_task,
            major_tasks=self.major_tasks,
            balance=self.mode == "train" and self.target_per_task > 0,
        )

    @staticmethod
    def pair_by_basename(a_paths: Sequence[str], b_paths: Sequence[str]) -> List[Tuple[str, str]]:
        return pair_by_basename(a_paths, b_paths)

    @staticmethod
    def balance_pairs(pairs: Sequence[Tuple[str, str]], target: int) -> List[Tuple[str, str]]:
        return balance_pairs(pairs, target)

    def _load_pair(self, deg_path: str, target_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = Image.open(deg_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        condition, target = self._align_size(condition, target)

        size = self.train_size
        if self.mode == "train":
            condition = self._maybe_resize_if_small(condition, size)
            target = self._maybe_resize_if_small(target, size)
            w, h = condition.size
            crop_x = random.randint(0, w - size)
            crop_y = random.randint(0, h - size)
            condition = TF.crop(condition, crop_y, crop_x, size, size)
            target = TF.crop(target, crop_y, crop_x, size, size)
            if self.augment_flip and random.random() < 0.5:
                condition = TF.hflip(condition)
                target = TF.hflip(target)
            if self.augment_flip and random.random() < 0.5:
                condition = TF.vflip(condition)
                target = TF.vflip(target)
        elif size > 0:
            condition = self._maybe_resize_if_small(condition, size)
            target = self._maybe_resize_if_small(target, size)
            condition = TF.center_crop(condition, [size, size])
            target = TF.center_crop(target, [size, size])

        condition_t = self._normalize_tensor(self.to_tensor(condition), self.normalize_to)
        target_t = self._normalize_tensor(self.to_tensor(target), self.normalize_to)
        if self.load_contexts:
            deg_context = self.load_context_from_image_path(deg_path, self.deg_eval_info_root)
            clean_context = self.load_context_from_image_path(target_path, self.clean_eval_info_root)
        else:
            # Online Q-Instruct mode supplies contexts in the training process.
            zeros = torch.zeros(8, self.token_dim + len(self.CANONICAL_ATTRS), dtype=torch.float32)
            deg_context, clean_context = zeros, zeros.clone()
        return target_t, condition_t, deg_context, clean_context

    def __getitem__(self, idx: int):
        deg_path, target_path, task = self.pairs[idx]
        target, condition, deg_context, clean_context = self._load_pair(deg_path, target_path)
        if self.return_paths:
            return {
                "target": target,
                "input": condition,
                "degraded_context": deg_context,
                "clean_context": clean_context,
                "degraded_path": deg_path,
                "target_path": target_path,
                "task": task,
            }
        return target, condition, deg_context, clean_context

    def __len__(self) -> int:
        return len(self.pairs)


class AttributePairedDataset(data.Dataset, CsvAttributeContextMixin, _PairTransformMixin):
    """Evaluation/manifest-friendly paired dataset returning dictionaries."""

    def __init__(
        self,
        root: str = "./datasets",
        split: str = "eval",
        crop_size: int = 128,
        eval_center_crop: int = 0,
        tasks: Sequence[str] = ("test2800",),
        normalize_to: str = "zero1",
        context_root: str = DEFAULT_CONTEXT_ROOT,
        degraded_context_root: Optional[str] = None,
        clean_context_root: Optional[str] = None,
        missing_context: str = "error",
        token_dim: int = 4096,
        pref_logit_dim: int = 9,
    ):
        super().__init__()
        if missing_context not in {"error", "zeros"}:
            raise ValueError("missing_context must be 'error' or 'zeros'")
        self.root = root
        self.split = split
        self.mode = "train" if split == "train" else "eval"
        self.crop_size = int(crop_size)
        self.eval_center_crop = int(eval_center_crop)
        self.normalize_to = normalize_to
        self.context_root = context_root
        self.degraded_context_root = degraded_context_root or context_root
        self.clean_context_root = clean_context_root or context_root
        self.missing_context = missing_context
        self.token_dim = int(token_dim)
        self.pref_logit_dim = int(pref_logit_dim)
        self.to_tensor = transforms.ToTensor()
        self.samples = build_pairs_for_tasks(root=root, tasks=parse_task_list(tasks), target_per_task=0, balance=False)

    def _load_context_or_zeros(self, image_path: str, root: str) -> torch.Tensor:
        try:
            return self.load_context_from_image_path(image_path, root)
        except FileNotFoundError:
            if self.missing_context == "zeros":
                return torch.zeros(8, self.token_dim + len(self.CANONICAL_ATTRS), dtype=torch.float32)
            raise

    def _load_pair(self, deg_path: str, target_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = Image.open(deg_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        condition, target = self._align_size(condition, target)

        if self.mode == "train" and self.crop_size > 0:
            size = self.crop_size
            condition = self._maybe_resize_if_small(condition, size)
            target = self._maybe_resize_if_small(target, size)
            w, h = condition.size
            crop_x = random.randint(0, w - size)
            crop_y = random.randint(0, h - size)
            condition = TF.crop(condition, crop_y, crop_x, size, size)
            target = TF.crop(target, crop_y, crop_x, size, size)
            if random.random() < 0.5:
                condition = TF.hflip(condition)
                target = TF.hflip(target)
        elif self.eval_center_crop > 0:
            size = self.eval_center_crop
            condition = self._maybe_resize_if_small(condition, size)
            target = self._maybe_resize_if_small(target, size)
            condition = TF.center_crop(condition, [size, size])
            target = TF.center_crop(target, [size, size])

        condition_t = self._normalize_tensor(self.to_tensor(condition), self.normalize_to)
        target_t = self._normalize_tensor(self.to_tensor(target), self.normalize_to)
        deg_context = self._load_context_or_zeros(deg_path, self.degraded_context_root)
        clean_context = self._load_context_or_zeros(target_path, self.clean_context_root)
        return target_t, condition_t, deg_context, clean_context

    def __getitem__(self, idx: int) -> Dict[str, object]:
        deg_path, target_path, task = self.samples[idx]
        target, input_, degraded_context, clean_context = self._load_pair(deg_path, target_path)
        return {
            "target": target,
            "input": input_,
            "degraded_context": degraded_context,
            "clean_context": clean_context,
            "degraded_path": deg_path,
            "target_path": target_path,
            "task": task,
        }

    def __len__(self) -> int:
        return len(self.samples)
