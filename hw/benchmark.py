from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

from hw.constants import CHOICES


def normalize_text(text: str) -> str:
    """Simple normalization for free-form answers."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def parse_mc_answer(text: str, choices: tuple[str, ...] = CHOICES) -> str | None:
    """Extract multiple-choice answer letter from model output."""
    text = text.strip()
    if text in choices:
        return text
    pattern = r"\b(" + "|".join(re.escape(c) for c in choices) + r")\b"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


def build_benchmark_prompt(question: str, options: list[str]) -> str:
    """Build prompt for multiple-choice visual math evaluation."""
    options_text = "\n".join(options)
    return (
        "Реши визуально-математическую задачу. "
        "Выбери один вариант ответа и в конце напиши только букву.\n\n"
        f"Вопрос: {question}\n"
        f"Варианты:\n{options_text}\n"
        "Ответ:"
    )


def compute_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute overall and per-subject accuracy from prediction rows."""
    if not rows:
        return {"overall": 0.0}

    total = len(rows)
    correct = sum(int(r.get("prediction") == r.get("answer")) for r in rows)
    metrics = {"overall": correct / total}

    subjects = sorted({r.get("subject", "unknown") for r in rows})
    for subject in subjects:
        sub_rows = [r for r in rows if r.get("subject", "unknown") == subject]
        sub_correct = sum(int(r.get("prediction") == r.get("answer")) for r in sub_rows)
        metrics[f"subject/{subject}"] = sub_correct / max(1, len(sub_rows))
    return metrics


def run_benchmark(config: dict[str, Any], toy: bool = False) -> dict[str, float]:
    """Run evaluation loop."""
    import torch
    from torch.utils.data import DataLoader
    from hw.dataset import MathVQADataset
    from hw.processor import MathVLMProcessor, ProcessorConfig
    from hw.model import MathVLM, ModelConfig
    from transformers import AutoTokenizer, AutoModel

    data_cfg = config.get("data", {})
    proc_cfg = config.get("processor", {})
    model_cfg = config.get("model", {})
    trainer_cfg = config.get("trainer", {})

    manifest = data_cfg.get("eval_manifest") or data_cfg.get("train_manifest")
    split = "dev" if not toy else "train"
    max_samples = 10 if toy else data_cfg.get("max_eval_samples")

    dataset = MathVQADataset(manifest_path=manifest, split=split, max_samples=max_samples)

    proc_config = ProcessorConfig(
        image_size=proc_cfg.get("image_size", 224),
        num_tiles=proc_cfg.get("num_tiles", 1),
        num_image_tokens=proc_cfg.get("num_image_tokens", 49),
        max_length=proc_cfg.get("max_length", 512),
    )

    device = torch.device(trainer_cfg.get("device", "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["language_model"])
    processor = MathVLMProcessor(tokenizer, proc_config)

    vision_encoder = AutoModel.from_pretrained(model_cfg["vision_encoder"]).to(device)
    language_model = AutoModel.from_pretrained(model_cfg["language_model"]).to(device)
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    m_config = ModelConfig(
        vision_hidden_size=vision_encoder.config.hidden_size,
        text_hidden_size=language_model.config.hidden_size,
        num_image_tokens=proc_config.num_image_tokens,
        image_token_id=image_token_id,
    )
    model = MathVLM(vision_encoder, language_model, m_config).to(device)

    ckpt_path = trainer_cfg.get("save_checkpoint_path") or config.get("checkpoint_path")
    if ckpt_path and Path(ckpt_path).exists():
        model.adapter.load_state_dict(torch.load(ckpt_path, map_location=device))

    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=1, collate_fn=processor.collate)
    for i, (batch, sample) in enumerate(zip(loader, dataset)):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad():
            generated = model.generate(batch, max_new_tokens=16)
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
        prediction = parse_mc_answer(decoded) or normalize_text(decoded)
        rows.append({"id": sample.id, "prediction": prediction, "answer": sample.answer, "subject": sample.subject})

    output_path = config.get("output_path")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return compute_accuracy(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--toy", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    metrics = run_benchmark(config, toy=args.toy)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
