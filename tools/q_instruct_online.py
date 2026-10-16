from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, Union

import torch
import torch.nn as nn
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mplug_owl2.constants import IMAGE_TOKEN_INDEX
from mplug_owl2.mm_utils import tokenizer_image_token
from mplug_owl2.model.builder import load_pretrained_model


def _patch_llama_decoder_layer_init() -> None:
    try:
        import mplug_owl2.model.modeling_llama2 as mplug_llama
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer,
            LlamaDynamicNTKScalingRotaryEmbedding,
            LlamaLinearScalingRotaryEmbedding,
            LlamaModel,
            LlamaRotaryEmbedding,
            rotate_half,
        )
        from mplug_owl2.model.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    except Exception:
        return
    if getattr(LlamaDecoderLayer.__init__, "_q_instruct_patched", False):
        layer_patched = True
    else:
        layer_patched = False
        original_init = LlamaDecoderLayer.__init__

        def patched_init(self, config, layer_idx=None):  # type: ignore[override]
            del layer_idx
            if not hasattr(config, "mlp_bias"):
                config.mlp_bias = False
            return original_init(self, config)

        patched_init._q_instruct_patched = True  # type: ignore[attr-defined]
        LlamaDecoderLayer.__init__ = patched_init  # type: ignore[assignment]

    if not hasattr(LlamaModel, "_prepare_decoder_attention_mask"):
        def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
            return _prepare_4d_causal_attention_mask(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )

        LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask  # type: ignore[attr-defined]

    def _patch_rotary_forward(cls) -> None:
        if getattr(cls.forward, "_q_instruct_seq_len_patched", False):
            return
        original_forward = cls.forward

        def patched_forward(self, x, position_ids=None, seq_len=None):  # type: ignore[override]
            del position_ids
            if seq_len is None:
                seq_len = x.shape[-2]
            position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0)
            cos, sin = original_forward(self, x, position_ids)
            return cos, sin

        patched_forward._q_instruct_seq_len_patched = True  # type: ignore[attr-defined]
        cls.forward = patched_forward  # type: ignore[assignment]

    for rotary_cls in (
        LlamaRotaryEmbedding,
        LlamaLinearScalingRotaryEmbedding,
        LlamaDynamicNTKScalingRotaryEmbedding,
    ):
        _patch_rotary_forward(rotary_cls)

    if not getattr(mplug_llama.apply_rotary_pos_emb, "_q_instruct_position_ids_patched", False):
        def old_style_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
            del unsqueeze_dim
            if cos.dim() == 4:
                cos = cos.squeeze(1)
                sin = sin.squeeze(1)
            if cos.dim() == 3 and cos.size(0) == 1:
                cos = cos.squeeze(0)
                sin = sin.squeeze(0)
            if position_ids is None:
                position_ids = torch.arange(q.shape[-2], device=q.device, dtype=torch.long).unsqueeze(0)
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        old_style_apply_rotary_pos_emb._q_instruct_position_ids_patched = True  # type: ignore[attr-defined]
        mplug_llama.apply_rotary_pos_emb = old_style_apply_rotary_pos_emb


ATTRIBUTES = [
    "Brightness",
    "Contrast",
    "Color",
    "Sharpness",
    "Structure",
    "Texture",
    "Noise",
    "Artifact",
]

PREFERENCE_WORDS = [
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

POS_IDXS = (0, 6)
MID_IDXS = (1, 7)
NEG_IDXS = (2, 8)


def longest_common_prefix_len(seq_list: Sequence[torch.Tensor]) -> int:
    limit = min(x.shape[1] for x in seq_list)
    idx = 0
    while idx < limit:
        ref = seq_list[0][0, idx].item()
        if all(seq[0, idx].item() == ref for seq in seq_list[1:]):
            idx += 1
        else:
            break
    return idx


def center_crop_min_square(img: Image.Image) -> Image.Image:
    width, height = img.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return img.crop((left, top, left + side, top + side))


ImageInput = Union[str, Path, Image.Image]


def _processor_pad_color(image_processor) -> tuple[int, int, int]:
    means = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    if isinstance(means, (float, int)):
        means = [float(means)] * 3
    scale = 255.0 if max(means) <= 1 else 1.0
    return tuple(int(round(max(0, min(255, float(v) * scale)))) for v in means)


def _letterbox_full_image(image: Image.Image, target_size: int, pad_color: tuple[int, int, int]) -> Image.Image:
    factor = min(target_size / image.width, target_size / image.height)
    size = (max(1, round(image.width * factor)), max(1, round(image.height * factor)))
    resized = image.resize(size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (target_size, target_size), pad_color)
    canvas.paste(resized, ((target_size - size[0]) // 2, (target_size - size[1]) // 2))
    return canvas


class QInstructAttributeExtractor(nn.Module):
    def __init__(
        self,
        model_path: str = "q-future",
        model_base: str | None = None,
        model_name: str = "mplug_owl2",
        device: str = "cuda:0",
        center_crop_square: bool = False,
    ):
        super().__init__()
        _patch_llama_decoder_layer_init()
        print(f"[IQA] loading Q-Instruct from {model_path} on {device}", flush=True)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_base, model_name, device=device)
        print("[IQA] Q-Instruct loaded", flush=True)
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.center_crop_square = center_crop_square
        self.pad_color = _processor_pad_color(image_processor)
        self.model.eval()
        self.pref_logit_dim = len(PREFERENCE_WORDS)
        self.attr_dim = len(ATTRIBUTES)

        self.preference_ids = [ids[1] for ids in tokenizer(PREFERENCE_WORDS)["input_ids"]]
        self.full_prompt_ids = []
        for attr in ATTRIBUTES:
            prompt = f"USER: <|image|> Rate the {attr} quality of the image.\nASSISTANT: "
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            self.full_prompt_ids.append(ids)

        common_len = longest_common_prefix_len(self.full_prompt_ids)
        self.prefix_ids = self.full_prompt_ids[0][:, :common_len]
        self.tail_ids_list = [ids[:, common_len:] for ids in self.full_prompt_ids]

    def _prepare_images(self, image_inputs: Sequence[ImageInput]) -> torch.Tensor:
        images = []
        for item in image_inputs:
            if isinstance(item, Image.Image):
                img = item.convert("RGB")
            else:
                img = Image.open(item).convert("RGB")
            if self.center_crop_square:
                img = center_crop_min_square(img)
            images.append(img)
        if self.center_crop_square:
            tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]
        else:
            # Keep the complete source image visible. The processor must not
            # apply its own square center crop after this letterbox step.
            images = [_letterbox_full_image(img, 448, self.pad_color) for img in images]
            preprocess = getattr(self.image_processor, "preprocess", self.image_processor)
            try:
                encoded = preprocess(images, return_tensors="pt", do_resize=False, do_center_crop=False)
            except TypeError:
                old_resize = getattr(self.image_processor, "do_resize", None)
                old_crop = getattr(self.image_processor, "do_center_crop", None)
                if old_resize is None or old_crop is None:
                    raise RuntimeError("Cannot disable Q-Instruct processor resize/center-crop")
                try:
                    self.image_processor.do_resize = False
                    self.image_processor.do_center_crop = False
                    encoded = preprocess(images, return_tensors="pt")
                finally:
                    self.image_processor.do_resize = old_resize
                    self.image_processor.do_center_crop = old_crop
            tensor = encoded["pixel_values"]
        return tensor.to(device=self.model.device, dtype=self.model.dtype)

    def _build_prefix_cache(self, image_tensor: torch.Tensor, batch_size: int):
        outputs = self.model(
            input_ids=self.prefix_ids.repeat(batch_size, 1),
            images=image_tensor,
            use_cache=True,
            return_dict=True,
            output_hidden_states=False,
        )
        return outputs.past_key_values

    @staticmethod
    def _extract_quality_state(outputs) -> torch.Tensor:
        hidden = outputs.hidden_states
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[-1]
        return hidden[:, -2, :]

    @staticmethod
    def _pref_logits_to_scores(logits_9: torch.Tensor) -> torch.Tensor:
        sel = logits_9[..., list(POS_IDXS + MID_IDXS + NEG_IDXS)]
        prob = torch.softmax(sel, dim=-1)
        pos = prob[..., 0:2].sum(dim=-1)
        mid = prob[..., 2:4].sum(dim=-1)
        return pos + 0.5 * mid

    def _format_context(self, raw_combined: torch.Tensor) -> torch.Tensor:
        attr_tokens = raw_combined[:, :, :-self.pref_logit_dim]
        pref_logits = raw_combined[:, :, -self.pref_logit_dim :]
        attr_scores = self._pref_logits_to_scores(pref_logits)
        attr_scores = attr_scores.unsqueeze(1).expand(-1, self.attr_dim, -1)
        return torch.cat([attr_tokens, attr_scores], dim=-1)

    @torch.inference_mode()
    def forward_raw(self, image_inputs: Sequence[ImageInput]) -> torch.Tensor:
        image_tensor = self._prepare_images(image_inputs)
        batch_size = image_tensor.shape[0]
        prefix_cache = self._build_prefix_cache(image_tensor, batch_size)

        combined = []
        for tail_ids in self.tail_ids_list:
            outputs = self.model(
                input_ids=tail_ids.repeat(batch_size, 1),
                past_key_values=prefix_cache,
                images=None,
                return_dict=True,
                output_hidden_states=True,
                use_cache=False,
            )
            quality_state = self._extract_quality_state(outputs)
            pref_logits = outputs.logits[:, -1, self.preference_ids]
            combined.append(torch.cat([quality_state, pref_logits], dim=-1))

        return torch.stack(combined, dim=1)

    @torch.inference_mode()
    def forward(self, image_inputs: Sequence[ImageInput]) -> torch.Tensor:
        return self._format_context(self.forward_raw(image_inputs))

    @torch.inference_mode()
    def infer_batch_combined(self, image_inputs: Sequence[ImageInput]) -> torch.Tensor:
        return self.forward_raw(image_inputs)
