# HPML Course Project: Dynamic KV Cache Compression

## Overview

This project extends CCM (Compressed Context Memory) from conversation compression to **reasoning compression** by using a learned classifier to dynamically predict where to insert `<COMP>` tokens.

## Structure

```
├── data/
│   └── gsm8k_compressed_train.json      # 1000 GSM8K samples with <COMP> tokens
├── notebooks/
│   └── ccm_reasoning.ipynb              # exploration notebook
├── outputs/                             # training outputs (checkpoints, logs)
├── scripts/
│   ├── main.sh                          # bash script to run main.py
│   └── visualize_XXX.sh
├── src/
│   ├── analysis_module/                 # scripts for analysis, plotting, etc.
│   │   └── eval_compression.py          # TODO: utility functions for inserting COMP tokens, also for eval?
│   ├── configs/
│   │   ├── base_configs.yaml            # base configs
│   │   └── gsm8k_classifier.yaml        # GSM8K classifier config
│   ├── data_module/
│   │   ├── DataCollator.py              # custom data collator w/ padding
│   │   ├── generate_compression_dataset.py  # standalone dataset generation script (w/ gpt 4o-mini)
│   │   ├── load_data.py                 # dataset loading
│   │   └── preprocessing.py             # tokenization + label creation
│   ├── model_module/
│   │   ├── compression_classifier.py    # binary classifier head (MLP)
│   │   ├── compression_probe_model.py   # wrapper: frozen LM + classifier
│   │   └── load_model.py                # model initialization
│   ├── train_module/
│   │   └── train_utils.py               # HuggingFace Trainer setup
│   ├── common_utils.py                  # seed, wandb setup utilities
│   └── main.py                          # main entry point (hydra config)
├── .gitignore
└── requirements.txt
```

## Installation
```bash
git clone git@github.com:Lorenayannnnn/HPML-dynamic-compression.git
cd HPML-dynamic-compression
```

**Using `uv` (recommended):**
```bash
uv sync
```

**Using conda:**
```bash
conda create -n hpml-compress python=3.13
conda activate hpml-compress
pip install -r requirements.txt
```

## Phase 1: Data Generation

Generate GSM8K samples with `<COMP>` tokens using GPT-4o-mini:

```bash
uv run python src/data_module/generate_compression_dataset.py \
    --num_samples 500 \
    --output data/gsm8k_compressed_train.json
```

Script supports resuming (from crashes/rate limiting).

**Data Format** (`gsm8k_compressed_train.json`):
```json
{
  "question": "Natalia sold clips to 48 of her friends in April...",
  "reasoning_with_compression": "In April, Natalia sold 48 / 2 = 24 clips. <COMP>\n\nIn May, she sold 24 + 24 = 48 clips. <COMP>\n\nTotal = 24 + 48 = 72 clips.\n\n#### 72",
  "answer": "72",
  "original_reasoning": "Natalia sold 48/2 = <<48/2=24>>24 clips in April..."
}
```

## Phase 2: Classifier Training

Train the binary classifier to predict `<COMP>` token positions:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python src/main.py --config-name gsm8k_classifier
```

**Architecture:**
- Frozen LLaMA-3.1-8B extracts hidden states (fp16)
- Binary classifier head (MLP) predicts compression points (fp32)
- BCEWithLogitsLoss with balanced positive/negative sampling

**Training Config** (`gsm8k_classifier.yaml`):
- 3 epochs, batch size 64, learning rate 5e-5
- 80/10/10 train/val/test split
- Outputs saved to `outputs/classifier/`

## Phase 3: CCM Compression Training

Train CCM's conditional LoRA to learn compression at `<COMP>` positions:

```bash
cd Context-Memory && uv sync

CUDA_VISIBLE_DEVICES=0 uv run python run.py \
    --model llama-3.1-8b-instruct \
    --dataset gsm8k \
    --train \
    --run_id run1 \
    --no_wandb  # Remove for actual runs with logging
```

**Key Options:**
| Flag | Description |
|------|-------------|
| `--train` | **Required** - enables training (without it, only evaluation runs) |
| `--run_id <id>` | **Required** - Run ID for checkpoint + wandb. Same ID = resume, new ID = start fresh |
| `--no_wandb` | Disable wandb logging (recommended for testing) |
| `--model <name>` | Model: `llama-3.1-8b-instruct` (default), `llama-3.1-8b`, `llama-2-7b-chat`, `llama-2-7b` |

**Training Config** (`Context-Memory/src/config/gsm8k/llama-8b.yaml`):
- 1000 steps, batch size 4, gradient accumulation 16 (effective batch = 64)
- BF16 + SDPA attention (PyTorch 2.0+ built-in FlashAttention-2)
- Conditional LoRA (r=8) on q/k/v/o projections
- Checkpoints saved every 250 steps to `Context-Memory/result/gsm8k/`

**Resumption:** Use the same `--run_id` to resume both checkpoints and wandb. Use a new `--run_id` to start fresh.

## Workflow

1. **Entry point:** [main.py](src/main.py)
2. **Configuration:** [gsm8k_classifier.yaml](src/configs/gsm8k_classifier.yaml) or [base_configs.yaml](src/configs/base_configs.yaml)
3. **Load data:** [load_data.py](src/data_module/load_data.py) - `load_gsm8k_compressed()` for JSON with train/val/test splits
4. **Preprocessing:** [preprocessing.py](src/data_module/preprocessing.py) - `tokenize_gsm8k()` splits by `<COMP>` and creates labels
5. **Model:**
   - [compression_classifier.py](src/model_module/compression_classifier.py) - Binary head
   - [compression_probe_model.py](src/model_module/compression_probe_model.py) - Frozen LM + classifier wrapper
6. **Training:** [train_utils.py](src/train_module/train_utils.py) - HuggingFace Trainer

## TODOs

### Phase 2: Classifier Training Fixes
- [x] Fix `common_utils.py`: `from random import random` → `import random`
- [x] Fix `common_utils.py`: Add `OmegaConf.set_struct(configs, False)` before setting `wandb_run_name`
- [x] Fix `compression_probe_model.py`: Import path → `from src.model_module.compression_classifier`
- [x] Fix `compression_probe_model.py`: `next_is_COMP_label` undefined → assign from `labels`
- [x] Fix `compression_probe_model.py`: Cast hidden states to float32 (LM outputs fp16)
- [x] Fix `load_model.py`: Remove `freeze_lm=True` (LM frozen internally in `__init__`)
- [x] Fix `train_utils.py`: `configs.use_wandb` → `training_args.use_wandb`
- [x] Fix `train_utils.py`: `evaluation_strategy` → `eval_strategy` (transformers API)

### Phase 3: CCM Compression Training
- [x] Add LLaMA-3.1 support to CCM code (`context-memory/`)
- [x] Create GSM8K data format for CCM
- [x] Train conditional LoRA

### Phase 4: Integration & Inference
- [ ] Create unified inference pipeline (classifier + CCM)
- [ ] Implement baseline methods (no compression, fixed interval, random)

### Phase 5: Evaluation
- [ ] Run comprehensive evaluation on GSM8K test set
- [ ] Generate comparison artifacts (accuracy vs. compression trade-off)

### Phase 6: Meta-Analysis
- [ ] Categorize compression points (after calculations, conclusions, etc.)
- [ ] Analyze what's compressible vs. what must be retained
