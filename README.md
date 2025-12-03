# HPML Course Project: Dynamic KV Cache Compression


## Overview

This project extends CCM (Compressed Context Memory) from conversation compression to **reasoning compression** by using a learned classifier to dynamically predict where to insert `<COMP>` tokens.


## Structure
```
├── data/
│   └── gsm8k_compressed_train.json      # GSM8K samples with <COMP> tokens
├── notebooks/
│   └── ccm_reasoning.ipynb              # Exploration notebook
├── outputs/
├── scripts/
│   └── main.sh                          # bash script to run main.py
├── src/
│   ├── analysis_module/                 # scripts for analysis, plotting, etc.
│   ├── configs/
│   │   ├── base_configs.yaml            # base configs
│   │   └── gsm8k_classifier.yaml        # GSM8K classifier config
│   ├── data_module/
│   │   ├── DataCollator.py
│   │   ├── generate_compression_dataset.py  # standalone data generation script
│   │   ├── load_data.py                 # data loading
│   │   └── preprocessing.py             # tokenization and preprocessing
│   ├── model_module/
│   │   ├── compression_classifier.py   # binary classifier head
│   │   ├── compression_probe_model.py  # wrapper: frozen LM + classifier
│   │   └── load_model.py               # model initialization
│   ├── train_module/
│   │   └── train_utils.py              # HuggingFace trainer setup
│   ├── common_utils.py                 # common utility functions
│   └── main.py                         # main entry point (hydra config)
├── .gitignore
└── requirements.txt
```

## Phase 1: Data Generation

Generate GSM8K samples with `<COMP>` tokens using GPT-4o-mini:

```bash
# From COMS6998-HPML directory
uv run python Project/HPML-dynamic-compression/src/data_module/generate_compression_dataset.py \
    --num_samples 500 \
    --output Project/HPML-dynamic-compression/data/gsm8k_compressed_train.json
```

The script supports resume (skips existing samples), rate limiting, and checkpointing.

## Workflow
1. Entry point: [main.py](src/main.py)
2. Configuration file: [gsm8k_classifier.yaml](src/configs/gsm8k_classifier.yaml) for GSM8K, or [base_configs.yaml](src/configs/base_configs.yaml). Change compression_token under model_args and other configs accordingly.
3. Load data: [load_data.py](src/data_module/load_data.py). Includes `load_gsm8k_compressed()` for loading the generated dataset with train/val/test splits.
4. Data preprocessing: Refer to `tokenize_gsm8k()` and `tokenize()` functions in [preprocessing.py](src/data_module/preprocessing.py), which handle splitting CoT by \<COMP\> token and creating labels.
5. Initialize model:
    - Compression Classifier: [compression_classifier.py](src/model_module/compression_classifier.py); takes in hidden states and outputs binary predictions
    - Wrapper Model: [compression_probe_model.py](src/model_module/compression_probe_model.py); wraps frozen language model and classifier
6. Create Trainer: [train_utils.py](src/train_module/train_utils.py)

## Usage

### Install dependencies
`uv` is preferred: 
```bash
cd COMS6998-HPML
uv sync
```

But can use `conda` if more convenient: 
```
conda create -n hpml-compress python=3.9
conda activate hpml-compress
pip install -r requirements.txt
```

### Run
- Go to [main.sh](scripts/main.sh) and modify the configurations as needed
- Run:
```bash
bash scripts/main.sh
```

Or directly with hydra:
```bash
PYTHONPATH=. uv run python src/main.py --config-name gsm8k_classifier
```

## TODOs

### Phase 2: Classifier Training Fixes
- [ ] Fix `common_utils.py`: `from random import random` should be `import random`
- [ ] Fix `common_utils.py`: Add `OmegaConf.set_struct(configs, False)` before setting `wandb_run_name`
- [ ] Fix `compression_probe_model.py`: Import path should be `from src.model_module.compression_classifier`
- [ ] Fix `compression_probe_model.py`: `next_is_COMP_label` is undefined, should be assigned from `labels`
- [ ] Fix `compression_probe_model.py`: Cast hidden states to float32 for classifier (LM outputs fp16)
- [ ] Fix `load_model.py`: Remove `freeze_lm=True` parameter (LM is frozen internally in `__init__`)
- [ ] Fix `train_utils.py`: `configs.use_wandb` should be `training_args.use_wandb`
- [ ] Fix `train_utils.py`: `evaluation_strategy` renamed to `eval_strategy` in newer transformers

### Phase 3: CCM Compression Training
- [ ] Add LLaMA-3.1 support to CCM code
- [ ] Create GSM8K data format for CCM
- [ ] Train conditional LoRA

### Phase 4: Integration & Inference
- [ ] Create unified inference pipeline
- [ ] Implement baseline methods for comparison

### Phase 5: Evaluation
- [ ] Run comprehensive evaluation
- [ ] Generate comparison artifacts