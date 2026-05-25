from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space.

    Architecture: LayerNorm -> Linear -> GELU -> AdaptivePool -> Linear
    """

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.num_image_tokens = num_image_tokens
        self.norm = nn.LayerNorm(vision_hidden_size)
        self.proj1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(num_image_tokens)
        self.proj2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        x = self.norm(vision_hidden_states)          # [B, N, vision_hidden_size]
        x = self.proj1(x)                            # [B, N, text_hidden_size]
        x = self.act(x)
        x = x.transpose(1, 2)                        # [B, text_hidden_size, N]
        x = self.pool(x)                             # [B, text_hidden_size, num_image_tokens]
        x = x.transpose(1, 2)                        # [B, num_image_tokens, text_hidden_size]
        x = self.proj2(x)
        return x


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings."""
    result = input_embeds.clone()
    for b in range(input_ids.shape[0]):
        positions = (input_ids[b] == image_token_id).nonzero(as_tuple=True)[0]
        result[b, positions] = visual_embeds[b, : len(positions)]
    return result


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model."""

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode [B, T, 3, H, W] pixel values → [B, num_image_tokens, text_hidden_size]."""
        B, T, C, H, W = pixel_values.shape
        flat = pixel_values.view(B * T, C, H, W)
        vision_out = self.vision_encoder(pixel_values=flat)
        vh = vision_out.last_hidden_state if hasattr(vision_out, "last_hidden_state") else vision_out
        vh = vh.view(B, T * vh.shape[1], vh.shape[2])
        return self.adapter(vh)

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass returning a dict/object with a .loss field."""
        visual_embeds = self._encode_images(batch["pixel_values"])
        input_ids = batch["input_ids"]
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        merged = merge_visual_embeddings(text_embeds, input_ids, visual_embeds, self.config.image_token_id)
        return self.language_model(
            inputs_embeds=merged,
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels"),
        )

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        visual_embeds = self._encode_images(batch["pixel_values"])
        input_ids = batch["input_ids"]
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        merged = merge_visual_embeddings(text_embeds, input_ids, visual_embeds, self.config.image_token_id)
        return self.language_model.generate(
            inputs_embeds=merged,
            attention_mask=batch.get("attention_mask"),
            **generation_kwargs,
        )
