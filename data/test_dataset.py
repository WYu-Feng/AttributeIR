from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from data.paired_dataset import (
    CANONICAL_ATTRS,
    CsvAttributeContextMixin,
    _PairTransformMixin,
    parse_task_list,
)


def _glob_sorted(pattern: str, limit: int = 0) -> List[str]:
    paths = sorted(glob.glob(pattern))
    return paths[:limit] if limit and limit > 0 else paths


def _pair_by_basename(a_paths: Sequence[str], b_paths: Sequence[str]) -> List[Tuple[str, str]]:
    b_map = {os.path.basename(p).lower(): p for p in b_paths}
    pairs: List[Tuple[str, str]] = []
    for a_path in a_paths:
        key = os.path.basename(a_path).lower()
        if key in b_map:
            pairs.append((a_path, b_map[key]))
    return pairs


def _glob_pair(
    root: str,
    input_parts: Sequence[str],
    target_parts: Sequence[str],
    pattern: str = "*.png",
    limit: int = 0,
) -> List[Tuple[str, str]]:
    a_paths = _glob_sorted(os.path.join(root, *input_parts, pattern), limit=limit)
    b_paths = _glob_sorted(os.path.join(root, *target_parts, pattern), limit=limit)
    return _pair_by_basename(a_paths, b_paths)


def build_test_pairs(root: str, task: str) -> List[Tuple[str, str, str]]:
    """Return explicit test pairs.

    Keep this function intentionally simple: testing data should be selected by
    clear glob paths instead of sharing the training dataset's task balancing.
    """
    task = task.lower()

    if task in {"gopro", "gopro_test", "deblur_gopro"}:
        A_paths = glob.glob(os.path.join(root, "Image deblurring", "GoPro", "test", "input", "*.png"))[:200]
        B_paths = glob.glob(os.path.join(root, "Image deblurring", "GoPro", "test", "target", "*.png"))[:200]
    elif task in {"dpdd", "defocus_dpdd"}:
        A_paths = sorted(glob.glob(os.path.join(root, "Defocus deblurring", "DPDD", "input", "*.png")))
        B_paths = sorted(glob.glob(os.path.join(root, "Defocus deblurring", "DPDD", "target", "*.png")))
        if not A_paths or not B_paths:
            A_paths = sorted(glob.glob(os.path.join(root, "Image deblurring", "DPDD", "input", "*.png")))
            B_paths = sorted(glob.glob(os.path.join(root, "Image deblurring", "DPDD", "target", "*.png")))
    elif task in {"uieb", "underwater_uieb"}:
        A_paths = sorted(glob.glob(os.path.join(root, "Underwater", "UIEB", "input", "*.png")))[:200]
        B_paths = sorted(glob.glob(os.path.join(root, "Underwater", "UIEB", "target", "*.png")))[:200]
    elif task in {"jpeg_bsd500", "bsd500_jpeg", "bsd500_qf10"}:
        A_paths = sorted(glob.glob(os.path.join(root, "JPEG", "BSD500", "test", "input", "*.png")))[:200]
        B_paths = sorted(glob.glob(os.path.join(root, "JPEG", "BSD500", "test", "target", "*.png")))[:200]
    elif task in {"hide", "hide_test"}:
        A_paths = sorted(glob.glob(os.path.join(root, "Image deblurring", "HIDE", "input", "*.png")))[:200]
        B_paths = sorted(glob.glob(os.path.join(root, "Image deblurring", "HIDE", "target", "*.png")))[:200]
    elif task in {"rain_mist", "real_rain_mist"}:
        A_paths = sorted(glob.glob(os.path.join(root, "Real_world", "Rain Mist", "*.jpg")))
        B_paths = sorted(glob.glob(os.path.join(root, "Real_world", "Rain Mist", "*.jpg")))
    elif task in {"udc_ir", "udc-ir"}:
        A_paths = sorted(glob.glob(os.path.join(root, "Real_world", "UDC-IR", "input", "*.png")))
        B_paths = sorted(glob.glob(os.path.join(root, "Real_world", "UDC-IR", "target", "*.png")))
        if not A_paths or not B_paths:
            A_paths = sorted(glob.glob(os.path.join(root, "UnderDisplayCamera", "UDC-IR", "input", "*.png")))
            B_paths = sorted(glob.glob(os.path.join(root, "UnderDisplayCamera", "UDC-IR", "target", "*.png")))
    elif task in {"rain100l", "rain100l_test"}:
        A_paths = sorted(glob.glob(os.path.join(root, "Image deraining", "Rain100L", "input", "*.png")))
        B_paths = sorted(glob.glob(os.path.join(root, "Image deraining", "Rain100L", "target", "*.png")))
    else:
        raise ValueError(
            f"Unknown test task '{task}'. Available: "
            "gopro_test, dpdd, uieb, jpeg_bsd500, bsd500_qf10, hide, rain_mist, udc_ir, rain100l"
        )
    pairs = _pair_by_basename(A_paths, B_paths)

    return [(a, b, task) for a, b in pairs]


class AttributeTestDataset(data.Dataset, CsvAttributeContextMixin, _PairTransformMixin):
    """Paired testing dataset with explicit glob-based data selection."""

    CANONICAL_ATTRS = CANONICAL_ATTRS

    def __init__(
        self,
        root: str = "./datasets",
        tasks: Sequence[str] = ("gopro_test",),
        normalize_to: str = "minus1_1",
        test_size: int = 1280,
        eval_center_crop: int = 0,
        context_root: str = "./q_instruct_results",
        degraded_context_root: Optional[str] = None,
        clean_context_root: Optional[str] = None,
        missing_context: str = "error",
        load_contexts: bool = True,
        token_dim: int = 4096,
        pref_logit_dim: int = 9,
    ):
        super().__init__()
        if missing_context not in {"error", "zeros"}:
            raise ValueError("missing_context must be 'error' or 'zeros'")
        self.root = root
        self.normalize_to = normalize_to
        self.test_size = int(test_size)
        self.eval_center_crop = int(eval_center_crop)
        self.context_root = context_root
        self.degraded_context_root = degraded_context_root or context_root
        self.clean_context_root = clean_context_root or context_root
        self.missing_context = missing_context
        self.load_contexts = load_contexts
        self.token_dim = int(token_dim)
        self.pref_logit_dim = int(pref_logit_dim)
        self.to_tensor = transforms.ToTensor()

        self.samples: List[Tuple[str, str, str]] = []
        for task in parse_task_list(tasks):
            self.samples.extend(build_test_pairs(self.root, task))
        if len(self.samples) == 0:
            raise RuntimeError(f"No test pairs found for tasks={tuple(tasks)} under root={root}")

    def _load_context_or_zeros(self, image_path: str, root: str) -> torch.Tensor:
        if not self.load_contexts:
            return torch.zeros(8, self.token_dim + len(self.CANONICAL_ATTRS), dtype=torch.float32)
        try:
            return self.load_context_from_image_path(image_path, root)
        except FileNotFoundError:
            if self.missing_context == "zeros":
                return torch.zeros(8, self.token_dim + len(self.CANONICAL_ATTRS), dtype=torch.float32)
            raise

    @staticmethod
    def _center_crop_max_size(condition: Image.Image, target: Image.Image, size: int) -> Tuple[Image.Image, Image.Image]:
        if size <= 0:
            return condition, target
        w, h = condition.size
        crop_h = min(h, size)
        crop_w = min(w, size)
        if crop_h != h or crop_w != w:
            condition = TF.center_crop(condition, [crop_h, crop_w])
            target = TF.center_crop(target, [crop_h, crop_w])
        return condition, target

    def _load_pair(self, deg_path: str, target_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition = Image.open(deg_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")
        condition, target = self._align_size(condition, target)

        if self.test_size > 0:
            condition, target = self._center_crop_max_size(condition, target, self.test_size)
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


__all__ = ["AttributeTestDataset", "build_test_pairs"]
