from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from clipuie.config import MultimodalSection

try:
    from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
except ModuleNotFoundError as exc:  # pragma: no cover - import guard for optional dependency
    AutoModel = AutoTokenizer = CLIPModel = CLIPProcessor = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class MultimodalConditionAdapter(nn.Module):
    def __init__(self, config: MultimodalSection) -> None:
        super().__init__()
        if CLIPModel is None or AutoModel is None or AutoTokenizer is None or CLIPProcessor is None:
            raise RuntimeError(
                "Multimodal support requires `transformers`. Install project dependencies again after updating pyproject.toml."
            ) from _TRANSFORMERS_IMPORT_ERROR

        self.config = config
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.clip_model = self._safe_from_pretrained(CLIPModel, config.clip_model_name, "CLIP")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name, use_fast=False)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = self._safe_from_pretrained(AutoModel, config.llm_model_name, "LLM")
        self.clip_model = self.clip_model.float()
        self.llm_model = self.llm_model.float()

        if config.freeze_clip:
            self.clip_model.requires_grad_(False)
            self.clip_model.eval()
        if config.freeze_llm:
            self.llm_model.requires_grad_(False)
            self.llm_model.eval()

        clip_dim = int(self.clip_model.config.projection_dim)
        llm_dim = int(self.llm_model.config.hidden_size)
        input_dim = clip_dim * 2 + llm_dim + 12
        self.clip_image_norm = nn.LayerNorm(clip_dim)
        self.clip_text_norm = nn.LayerNorm(clip_dim)
        self.llm_norm = nn.LayerNorm(llm_dim)
        self.stats_norm = nn.LayerNorm(12)
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, config.adapter_hidden_dim),
            nn.GELU(),
            nn.Linear(config.adapter_hidden_dim, config.adapter_hidden_dim),
        )
        self.output_norm = nn.LayerNorm(config.adapter_hidden_dim)
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    @staticmethod
    def _safe_from_pretrained(model_cls, model_name_or_path: str, model_label: str):
        model_path = Path(model_name_or_path)
        has_safetensors = False
        if model_path.exists() and model_path.is_dir():
            has_safetensors = any(model_path.glob("*.safetensors"))
        try:
            if has_safetensors:
                return model_cls.from_pretrained(model_name_or_path, use_safetensors=True, torch_dtype=torch.float32)
            return model_cls.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
        except Exception as exc:
            message = str(exc)
            if "Due to a serious vulnerability issue in `torch.load`" in message:
                raise RuntimeError(
                    f"Failed to load {model_label} from `{model_name_or_path}` because the checkpoint is not being loaded "
                    "through safetensors under the current environment. Please use a model directory that contains "
                    "`model.safetensors` (or equivalent safetensors shards), or upgrade PyTorch to >= 2.6. "
                    "If this is your local CLIP directory, re-download the safetensors version from Hugging Face."
                ) from exc
            raise

    def _clip_image_inputs(self, images: torch.Tensor) -> torch.Tensor:
        image_size = self.clip_processor.image_processor.crop_size["height"]
        pixel_values = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor(self.clip_processor.image_processor.image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.clip_processor.image_processor.image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (pixel_values - mean) / std

    @staticmethod
    def _masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    @staticmethod
    def _ensure_tensor(output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            hidden_states = output.last_hidden_state
            if hidden_states.ndim == 3:
                return hidden_states.mean(dim=1)
            return hidden_states
        if isinstance(output, (tuple, list)) and output:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first
        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    def forward(self, images: torch.Tensor, masks: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        device = images.device
        clip_context = torch.no_grad() if self.config.freeze_clip else nullcontext()
        llm_context = torch.no_grad() if self.config.freeze_llm else nullcontext()

        clip_text_inputs = self.clip_processor(
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        clip_text_inputs = {key: value.to(device) for key, value in clip_text_inputs.items() if key in {"input_ids", "attention_mask"}}
        with clip_context:
            clip_image_features = self._ensure_tensor(self.clip_model.get_image_features(pixel_values=self._clip_image_inputs(images)))
            clip_text_features = self._ensure_tensor(self.clip_model.get_text_features(**clip_text_inputs))
        clip_image_features = self.clip_image_norm(F.normalize(clip_image_features, dim=1))
        clip_text_features = self.clip_text_norm(F.normalize(clip_text_features, dim=1))

        llm_inputs = self.llm_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.prompt_max_length,
            return_tensors="pt",
        )
        llm_inputs = {key: value.to(device) for key, value in llm_inputs.items()}
        with llm_context:
            llm_outputs = self.llm_model(**llm_inputs)
            llm_hidden_states = self._ensure_tensor(llm_outputs)
            if llm_hidden_states.ndim == 2:
                llm_features = llm_hidden_states
            else:
                llm_features = self._masked_mean_pool(llm_hidden_states, llm_inputs["attention_mask"])
        llm_features = self.llm_norm(llm_features)

        channel_mean = images.mean(dim=(2, 3), keepdim=False)
        channel_std = images.std(dim=(2, 3), keepdim=False)
        image_mean = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        image_std = images.std(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        mask_mean = masks.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        mask_std = masks.std(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        red_mean = channel_mean[:, 0:1]
        green_mean = channel_mean[:, 1:2]
        blue_mean = channel_mean[:, 2:3]
        color_offsets = torch.cat([blue_mean - red_mean, green_mean - red_mean], dim=1)
        stats = torch.cat([channel_mean, channel_std, image_mean, image_std, mask_mean, mask_std, color_offsets], dim=1)
        stats = self.stats_norm(stats)

        fused = torch.cat([clip_image_features, clip_text_features, llm_features, stats], dim=1)
        return self.output_norm(self.fusion(fused))
