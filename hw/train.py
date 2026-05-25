from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_step(model: torch.nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
    """Run one optimization step and return scalar loss."""
    model.train()
    out = model(batch)
    loss = out["loss"] if isinstance(out, dict) else out.loss
    assert torch.isfinite(loss), f"Non-finite loss: {loss}"
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def run_training(config: dict[str, Any], fast_train: bool = False) -> None:
    """Main training entry point."""
    from torch.utils.data import DataLoader
    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor, ProcessorConfig
    from hw.model import MathVLM, ModelConfig

    trainer_cfg = config.get("trainer", {})
    data_cfg = config.get("data", {})
    proc_cfg = config.get("processor", {})
    model_cfg = config.get("model", {})

    device = torch.device(trainer_cfg.get("device", "cpu"))
    dtype_str = trainer_cfg.get("dtype", "float32")
    dtype = getattr(torch, dtype_str, torch.float32)

    dataset = MathVQADataset(
        manifest_path=data_cfg["train_manifest"],
        split=data_cfg.get("split", "train"),
        max_samples=data_cfg.get("max_samples"),
    )

    proc_config = ProcessorConfig(
        image_size=proc_cfg.get("image_size", 224),
        num_tiles=proc_cfg.get("num_tiles", 1),
        tile_overlap=proc_cfg.get("tile_overlap", 0.0),
        num_image_tokens=proc_cfg.get("num_image_tokens", 49),
        max_length=proc_cfg.get("max_length", 512),
        ignore_index=proc_cfg.get("ignore_index", -100),
    )

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"])
    processor = MathVLMProcessor(tokenizer, proc_config)

    vision_encoder = AutoModel.from_pretrained(model_cfg["vision_encoder"]).to(device=device, dtype=dtype)
    language_model = AutoModel.from_pretrained(model_cfg["language_model"]).to(device=device, dtype=dtype)

    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    m_config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.hidden_size,
        num_image_tokens=proc_config.num_image_tokens,
        image_token_id=image_token_id,
    )
    model = MathVLM(vision_encoder, language_model, m_config).to(device=device, dtype=dtype)
    if model_cfg.get("freeze_vision", True):
        model.freeze_backbones()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=trainer_cfg.get("learning_rate", 5e-4),
        weight_decay=trainer_cfg.get("weight_decay", 0.0),
    )

    loader = DataLoader(
        dataset,
        batch_size=trainer_cfg.get("local_batch_size", 1),
        shuffle=True,
        collate_fn=processor.collate,
        num_workers=trainer_cfg.get("num_workers", 0),
    )

    max_steps = 1 if fast_train else trainer_cfg.get("max_steps", None)
    step = 0
    for epoch in range(trainer_cfg.get("num_train_epochs", 1)):
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss = train_one_step(model, batch, optimizer)
            step += 1
            print(f"step {step} loss {loss:.4f}")
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    ckpt_path = trainer_cfg.get("save_checkpoint_path")
    if ckpt_path:
        torch.save(model.adapter.state_dict(), ckpt_path)
        print(f"Saved adapter to {ckpt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_training(config, fast_train=args.fast_train)


if __name__ == "__main__":
    main()
