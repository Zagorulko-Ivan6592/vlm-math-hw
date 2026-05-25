# Report

## Track

Выбранный трек:

```text
A (CPU-only)
```

## Что реализовано

- [x] dataset.py — загрузка `manifest.jsonl`, фильтрация по split, открытие картинок как RGB PIL.Image, возврат `MathVQASample`
- [x] processor.py — resize/tile изображений, построение промпта с `<image>`-токенами, токенизация, маскировка промпта через `IGNORE_INDEX`, `collate_fn`
- [x] model.py — `VisionToTextAdapter` (LayerNorm→Linear→GELU→AdaptivePool→Linear), `merge_visual_embeddings`, `MathVLM.forward` и `.generate`
- [x] train.py — `train_one_step` (forward, backward, optimizer step), `run_training` с YAML-конфигом, gradient accumulation, сохранение чекпойнта
- [x] benchmark.py — `parse_mc_answer` (regex для A/B/C/D), `build_benchmark_prompt`, `compute_accuracy` (overall + by subject), `run_benchmark`

## Конфигурация

```text
config path:  configs/track_a_cpu.yaml
seed:         42
device:       cpu
dtype:        float32
max_steps:    3
batch size:   1
image_size:   224
num_tiles:    1
num_image_tokens: 16
max_length:   256
```

## Результаты

```text
public tests:       14 / 14 passed (pytest -q tests_public)
train loss (step3): 4.3497  (tiny random model, 3 шага, toy_math_vqa)
benchmark accuracy: 0.00%   (ожидаемо — случайная модель без обучения)
```

Smoke-тест запускался командой:
```bash
python3.11 scripts/smoke_cpu.py
```

Пайплайн отработал полностью: dataset → processor → model forward → train step → generate → parse → accuracy.

## Использованные ресурсы

```text
CPU/GPU:        Apple M-series CPU (без GPU)
VRAM:           не использовалась
время обучения: < 1 секунды (3 шага, tiny random модель, toy датасет)
```

## Анализ ошибок

Track A использует случайно инициализированную tiny-модель (без предобученных весов), поэтому все ошибки — следствие случайного вывода:

1. **Всегда предсказывает одну букву.** Неообученная модель выдаёт одинаковый токен вне зависимости от входа — `parse_mc_answer` каждый раз находит одну и ту же букву (например, "A").

2. **Визуальные эмбеддинги не несут смысла.** Адаптер не обучен, поэтому вставляемые visual embeddings — случайный шум, не связанный с содержимым картинки.

3. **Loss не убывает за 3 шага.** Из-за малого числа шагов и случайной инициализации loss колеблется (~5.9 → 6.0 → 4.3), а не устойчиво убывает.

Для получения осмысленной точности необходимы: предобученный vision encoder (ViT), предобученный LLM, и полноценное обучение адаптера (Track B/C).

## Комментарии

**Самое сложное:** корректная реализация `processor.py` — в частности, согласование длин `input_ids` и `labels` при разных вызовах tokenizer, и правильная маскировка промпта через `IGNORE_INDEX` так, чтобы loss считался только на ответе.

**Что можно улучшить:**
- Заменить `AdaptiveAvgPool1d` в адаптере на cross-attention или perceiver resampler для лучшего сжатия визуальных признаков
- Добавить нормализацию изображений по mean/std (ImageNet или CLIP)
- Поддержать multi-turn диалог в промпте

## Критерии оценивания

См. файл [`GRADING.md`](GRADING.md).
