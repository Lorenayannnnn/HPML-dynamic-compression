# HPML Course Project: Dynamic KV Cache Compression for Reasoning

## Team Information
- **Team Name**: Dynamic Compression
- **Members**:
  - Lorena Yan (ty2575)
  - Shayan Chowdhury (sc4040)
  - Chaitya Shah (cs4621)
  - Can Kerem Akbulut (cka2115)

---

## 1. Problem Statement

**Problem:** KV cache memory grows linearly during LLM generation, becoming a bottleneck for long-context reasoning. Existing compression methods use fixed heuristics that don't adapt to content importance.

**Our Solution:** Dynamic compression via a learned binary classifier that predicts optimal `<COMP>` token insertion points, enabling content-aware KV cache compression during chain-of-thought generation. 

---

## 2. Model Description

### Framework
- **PyTorch** with HuggingFace Transformers (v4.40+)
- **PEFT** for LoRA adapter training

### Architecture Components

| Component | Description |
|-----------|-------------|
| **Base LLM** | LLaMA-3.1-8B-Instruct (frozen during classifier training) |
| **CCM LoRA Adapter** | Conditional LoRA (rank=16) on Q/K/V/O projections, ~8.4M params |
| **Binary Classifier** | 2-layer MLP (4096 hidden dim), 16.8M params |

### Key Modifications
- Extended CCM from conversation compression to **reasoning compression** (intra-turn vs inter-turn)
- Implemented **dynamic `<COMP>` token insertion** via learned classifier (vs fixed positions)
- Added LLaMA-3.1 support to CCM codebase (originally LLaMA-7B)
- Knowledge distillation from GPT-4o-mini for compression annotations

---

## 3. Final Results Summary

### Main Results on GSM8K (200 test samples)

| Method | Accuracy | Avg COMP Tokens | KV Cache (MB) | Compression Ratio | Latency (s) | Throughput |
|--------|----------|-----------------|---------------|-------------------|-------------|------------|
| No Compression (vanilla) | 42.0% | 0.00 | 37.45 | 1.00x | 8.54 | 27.8 tok/s |
| Newline `<COMP>` (baseline) | 26.0% | 0.42 | 30.55 | 1.23x | 16.58 | 13.4 tok/s |
| **Classifier (τ=0.9) [Ours]** | **32.0%** | 0.52 | 26.63 | **1.41x** | 18.44 | 11.1 tok/s |

### Classifier Overhead

| Metric | Value |
|--------|-------|
| Classifier Parameters | 16.8M |
| Classifier Memory | 32 MB |
| Latency Overhead | 0.39% |
| Avg Classifier Time | 75 ms/sample |

### Key Observations
- **Our dynamic classifier outperforms the static newline baseline on both accuracy (32% vs 26%) and compression ratio (1.41x vs 1.23x)**
- The threshold τ controls the accuracy-compression trade-off: lower τ = more aggressive compression
- Classifier overhead is minimal (<0.4% of total inference time)
- Gap from vanilla LLM (42%) reflects the difficulty of reasoning compression where exact values matter

---

## 4. Reproducibility Instructions

### A. Requirements

**System Requirements:**
- NVIDIA GPU with 24+ GB VRAM (tested on A100)
- CUDA 11.8+
- Python 3.10+

**Install dependencies:**

Using `uv` (recommended):
```bash
uv sync
```

Using conda/pip:
```bash
conda create -n hpml-compress python=3.13
conda activate hpml-compress
pip install -r requirements.txt
```

---

### B. WandB Dashboard

View experiment metrics on our public WandB dashboards:
- **CCM LoRA Training Metrics (from C.3 below)**: [WandB Training Dashboard](https://wandb.ai/lorena-yantianyi1020/hpml-dynamic-compression?nw=nwuserlorenayantianyi1020)
- **Evaluation Metrics (from D below)**: [WandB Evaluation Dashboard](https://wandb.ai/cankeremakbulut-personal/hpml-dynamic-compression/workspace?nw=nwusercankeremakbulut)

---

### C. Training

#### Stage 1: Data Generation
Generate GSM8K samples with `<COMP>` tokens using GPT-4o-mini:

```bash
# Step 1: Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Step 2: Generate compression-annotated data
uv run python src/data_module/generate_compression_dataset.py \
    --num_samples 1000 \
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

#### Stage 2: Classifier Training
Train the binary classifier to predict `<COMP>` token positions:
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python src/main.py \
    --config-name gsm8k_classifier
```

**Architecture:**
- Frozen LLaMA-3.1-8B extracts hidden states (fp16)
- Binary classifier head (MLP) predicts compression points (fp32)
- BCEWithLogitsLoss with balanced positive/negative sampling

**Training Config** (`gsm8k_classifier.yaml`):
- 3 epochs, batch size 64, learning rate 5e-5
- 80/10/10 train/val/test split
- Outputs saved to `outputs/classifier/`

#### Stage 3: CCM LoRA Compression Training
Train CCM's conditional LoRA to learn compression at `<COMP>` positions:

```bash
# Step 1: Navigate to Context-Memory
cd Context-Memory && uv sync

# Step 2: Run CCM training
CUDA_VISIBLE_DEVICES=0 uv run python run.py \
    --model llama-3.1-8b-instruct \
    --dataset gsm8k \
    --train \
    --run_id ccm_gsm8k_run1
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

---

### D. Evaluation

Run evaluation on the GSM8K test set:

```bash
# No compression baseline (vanilla LLM)
uv run python src/analysis_module/eval_compression.py \
    --method compression_none

# Static baseline (newline-based COMP insertion)
uv run python src/analysis_module/eval_compression.py \
    --method compression_newline

# Our dynamic classifier
uv run python src/analysis_module/eval_compression.py \
    --method compression_classifier \
    --threshold 0.9
```

---

### E. Quickstart: Minimum Reproducible Result

To reproduce our main result (32% accuracy with 1.41x compression):

```bash
# Step 1: Clone and setup environment
git clone https://github.com/Lorenayannnnn/HPML-dynamic-compression.git
cd HPML-dynamic-compression
uv sync

# Step 2: Download pre-trained checkpoints (if available)
# Or train from scratch following Section C above

# Step 3: Run evaluation with our classifier
uv run python src/analysis_module/eval_compression.py \
    --method compression_classifier \
    --threshold 0.9 \
    --test_dataset data/gsm8k-test-200.json

# Step 4: Compare with baseline
uv run python src/analysis_module/eval_compression.py \
    --method compression_newline \
    --test_dataset data/gsm8k-test-200.json
```

Expected output:
```
Accuracy:            32.00%
Avg COMP tokens:     0.52 (±0.70)
Avg KV Cache (MB):   26.63 (±15.96)
```

---

## 5. Repository Structure

```
├── data/
│   ├── gsm8k_compressed_train.json           # 1000 GSM8K samples with <COMP> tokens
│   └── gsm8k-test-200.json                   # Test set for evaluation
├── notebooks/
│   └── ccm_reasoning.ipynb                   # Exploration notebook
├── outputs/                                  # Training outputs (checkpoints, logs)
│   ├── classifier/                           # Binary classifier checkpoints
│   ├── OURS_llama-3.1.../                    # Our CCM+classifier model
│   └── baseline_.../                         # Baseline CCM model
├── scripts/
│   └── main.sh                               # Bash script to run main.py
├── src/
│   ├── analysis_module/                      # Evaluation and analysis scripts
│   │   ├── eval_compression.py               # Main evaluation script
│   │   ├── gsm8k_utils.py                    # GSM8K answer extraction utilities
│   │   └── profiled_eval.py                  # PyTorch profiler integration
│   ├── configs/
│   │   ├── base_configs.yaml                 # Base configuration
│   │   └── gsm8k_classifier.yaml             # GSM8K classifier config
│   ├── data_module/
│   │   ├── DataCollator.py                   # Custom data collator with padding
│   │   ├── generate_compression_dataset.py   # Dataset generation (GPT-4o-mini)
│   │   ├── load_data.py                      # Dataset loading utilities
│   │   └── preprocessing.py                  # Tokenization and label creation
│   ├── model_module/
│   │   ├── ccm_llama.py                      # CCM-adapted LLaMA model
│   │   ├── compression_classifier.py         # Binary classifier head (MLP)
│   │   ├── compression_probe_model.py        # Wrapper: frozen LM + classifier
│   │   ├── load_model.py                     # Model initialization
│   │   └── peft_loader.py                    # PEFT adapter loading utilities
│   ├── train_module/
│   │   └── train_utils.py                    # HuggingFace Trainer setup
│   ├── common_utils.py                       # Seed, WandB setup utilities
│   └── main.py                               # Main entry point (Hydra config)
├── Context-Memory/                           # CCM framework (submodule)
│   ├── run.py                                # CCM training entry point
│   └── src/                                  # CCM source code
├── requirements.txt
└── README.md
```

---

## 6. Workflow

1. **Entry point:** [main.py](src/main.py) - Hydra-based configuration
2. **Configuration:** [gsm8k_classifier.yaml](src/configs/gsm8k_classifier.yaml)
3. **Load data:** [load_data.py](src/data_module/load_data.py) - `load_gsm8k_compressed()` for JSON with train/val/test splits
4. **Preprocessing:** [preprocessing.py](src/data_module/preprocessing.py) - `tokenize_gsm8k()` splits by `<COMP>` and creates labels
5. **Model:**
   - [compression_classifier.py](src/model_module/compression_classifier.py) - Binary classifier head (2-layer MLP)
   - [compression_probe_model.py](src/model_module/compression_probe_model.py) - Frozen LM + classifier wrapper
6. **Training:** [train_utils.py](src/train_module/train_utils.py) - HuggingFace Trainer
7. **Evaluation:** [eval_compression.py](src/analysis_module/eval_compression.py) - Online generation with KV compression

---

## 7. Notes

- All scripts are located in `src/` with modular organization
- Trained models are saved in `outputs/`
- Requires HuggingFace token for LLaMA-3.1 access: `huggingface-cli login`
- GPU memory: ~24GB for training, ~16GB for inference

### Contact
- Lorena Yan: ty2575@columbia.edu
- Shayan Chowdhury: sc4040@columbia.edu
- Chaitya Shah: cs4621@columbia.edu
- Can Kerem Akbulut: cka2115@columbia.edu

---

## Acknowledgments

This project was a final project for COMS 6998: High Performance Machine Learning with Prof. Kaoutar El Maghraoui in Fall 2025; thank you to her and the TAs for their guidance! 

This project builds on the [Compressed Context Memory (CCM)](https://github.com/snu-mllab/Context-Memory) framework by Kim et al. (2023).
