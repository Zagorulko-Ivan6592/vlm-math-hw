from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample."""

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Return [num_tiles, 3, H, W] float tensor normalized to [0, 1]."""
        image = image.convert("RGB")
        H = W = self.config.image_size
        n = self.config.num_tiles

        if n == 1:
            image = image.resize((W, H), Image.LANCZOS)
            arr = np.array(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            return tensor.unsqueeze(0)  # [1, 3, H, W]

        grid = math.isqrt(n)
        image = image.resize((grid * W, grid * H), Image.LANCZOS)
        arr = np.array(image, dtype=np.float32) / 255.0
        tiles = []
        for r in range(grid):
            for c in range(grid):
                tile = arr[r * H:(r + 1) * H, c * W:(c + 1) * W]
                tiles.append(torch.from_numpy(tile.copy()).permute(2, 0, 1))
        return torch.stack(tiles)  # [n, 3, H, W]

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build text prompt with visual special tokens."""
        img_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        img_part = f"{IMAGE_START_TOKEN} {img_tokens} {IMAGE_END_TOKEN}"
        options_text = "\n".join(sample.options) if sample.options else ""
        prompt = f"{img_part}\nВопрос: {sample.question}\nВарианты:\n{options_text}\nОтвет:"
        if include_answer:
            prompt += f" {sample.answer}"
        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample."""
        prompt = self.build_prompt(sample, include_answer=False)
        full = self.build_prompt(sample, include_answer=True)

        prompt_ids = self.tokenizer.encode(prompt)
        encoded = self.tokenizer(
            full,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        input_ids: list[int] = encoded["input_ids"]
        attention_mask: list[int] = encoded["attention_mask"]

        n_prompt = min(len(prompt_ids), len(input_ids))
        labels = [self.config.ignore_index] * n_prompt + input_ids[n_prompt:]
        labels = labels[:len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values into [B, T, 3, H, W]."""
        max_len = max(item["input_ids"].shape[0] for item in batch)

        input_ids_list, mask_list, labels_list, pv_list = [], [], [], []
        for item in batch:
            L = item["input_ids"].shape[0]
            pad = max_len - L
            input_ids_list.append(
                torch.cat([item["input_ids"], torch.full((pad,), self.tokenizer.pad_token_id, dtype=torch.long)])
            )
            mask_list.append(
                torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([item["labels"], torch.full((pad,), self.config.ignore_index, dtype=torch.long)])
            )
            pv_list.append(item["pixel_values"])

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(mask_list),
            "labels": torch.stack(labels_list),
            "pixel_values": torch.stack(pv_list),  # [B, T, 3, H, W]
        }
