# Compressed Context Memory (Modified Fork)

> **Note**: This is a modified fork of [CCM (Compressed Context Memory)](https://github.com/snu-mllab/Context-Memory) by Kim et al. (ICLR 2024), adapted for **reasoning compression on GSM8K** with **LLaMA 3.1** support.

![main](image/main.png)

**Original Work:** [Paper](https://janghyun1230.github.io/paper/ccm23.pdf) | [arXiv](https://arxiv.org/abs/2312.03414) | [Project Page](https://janghyun1230.github.io/memory/) | [Original Repo](https://github.com/snu-mllab/Context-Memory)

---

## What's Different in This Fork

This fork extends CCM from **conversation/dialogue compression** to **reasoning compression** for math problem solving (GSM8K). Key modifications:

### 1. LLaMA 3.1 Support
- Added **Grouped Query Attention (GQA)** support (LLaMA 3.1 uses 8 KV heads vs 32 query heads)
- Updated for **transformers 4.40+** API changes
- Supports `llama-3.1-8b-instruct`, `llama-3.1-8b`, and backward-compatible with LLaMA 2
- **SDPA/Flash Attention 2**: Configurable attention implementation (`sdpa`, `flash_attention_2`, or `eager`)

### 2. GSM8K Reasoning Dataset
- New data loader for GSM8K samples with `<COMP>` tokens at reasoning checkpoints
- Custom collator that masks `<COMP>` tokens in labels (compression markers, not prediction targets)

### 3. Dynamic Compression (vs Fixed Interval)
- Original CCM: `<COMP>` tokens at **fixed intervals** (every N tokens)
- This fork: `<COMP>` tokens at **semantically meaningful positions** (after calculations, conclusions)
- Works with our learned classifier that predicts WHERE to place `<COMP>` tokens

---

## Changes from Original Repository

### Transformers 4.40+ Compatibility

| Component | Original | Modified |
|-----------|----------|----------|
| Attention masks | `_make_causal_mask`, `_expand_mask` | `_prepare_4d_causal_attention_mask` |
| Rotary embedding init | `LlamaRotaryEmbedding(dim, max_pos, base)` | `LlamaRotaryEmbedding(config=config)` |
| Rotary embedding forward | `rotary_emb(x, seq_len=...)` | `rotary_emb(x, position_ids)` |
| MLP init | `LlamaMLP(hidden_size, intermediate_size, act)` | `LlamaMLP(config)` |
| Checkpointing | `from transformers.modeling_utils import checkpoint` | `from torch.utils.checkpoint import checkpoint` |
| TPU utils | `is_torch_tpu_available` | Fallback to `is_torch_xla_available` |
| Auth token | `use_auth_token=` | `token=` |

### GQA (Grouped Query Attention) Support

```python
# Added to LlamaAttention for LLaMA 3.1
self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)
self.num_key_value_groups = self.num_heads // self.num_key_value_heads

# K/V projections use fewer heads
self.k_proj = Linear(..., num_key_value_heads * head_dim)
self.v_proj = Linear(..., num_key_value_heads * head_dim)

# Expand before attention
key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)
```

### Files Modified

| Category | Files |
|----------|-------|
| LLaMA Architecture | `src/arch/ccm_llama.py`, `ccm_llama_stream.py`, `gist_llama.py` |
| T5 Architecture | `src/arch/ccm_t5.py`, `gist_t5.py` |
| Model Loading | `src/model.py`, `path_config.py` |
| Training | `src/trainer_seq2seq.py`, `run.py` |
| GSM8K Support (new) | `src/config/gsm8k/*`, `src/data/gsm8k/*` |

---

## Setup

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

**Supported Models:**
- `llama-3.1-8b-instruct` (default) - LLaMA 3.1 8B Instruct
- `llama-3.1-8b` - LLaMA 3.1 8B base
- `llama-2-7b-chat` - LLaMA 2 7B Chat
- `llama-2-7b` - LLaMA 2 7B base
- `llama-7b`, `llama-13b` - Original LLaMA
- `mistral-7b`, `mistral-7b-inst` - Mistral

---

## Training on GSM8K

```bash
CUDA_VISIBLE_DEVICES=0 uv run python run.py \
    --model llama-3.1-8b-instruct \  # Model to use
    --dataset gsm8k \                 # Dataset (gsm8k, metaicl, soda, etc.)
    --train \                         # Enable training mode
    --run_id my_run_v1 \              # Run ID for checkpoint/wandb resume (same ID = resume)
    --no_wandb                        # Disable wandb logging (remove for wandb)
```

**Override config via CLI** (Hydra): `training.max_steps=500`, `data.train_ratio=0.8`, etc.

**Training Config** (`src/config/gsm8k/data.yaml` + `llama-8b.yaml`):
- Learning rate 3e-4 with cosine scheduler
- LoRA r=8 on q_proj, k_proj, v_proj, o_proj
- BF16 training with gradient checkpointing
- Early stopping (patience=3) with best model checkpointing
- Outputs: `result/gsm8k/`

---

## Original CCM Features

The original CCM paper introduced:
- Dynamic updates of **compressed key/value memory** during LLM interactions
- **Conditional LoRA** that activates only at compression positions
- **Fully parallelized training** for recurrent compression procedures
- Evaluations on conversation, multi-task ICL, and personalization

For the original usage (MetaICL, SODA, dialogue), see the [original repository](https://github.com/snu-mllab/Context-Memory).

---

## Citation

If you use this code, please cite the original CCM paper:

```bibtex
@inproceedings{
    kim2024compressed,
    title={Compressed Context Memory for Online Language Model Interaction},
    author={Jang-Hyun Kim and Junyoung Yeom and Sangdoo Yun and Hyun Oh Song},
    booktitle={ICLR},
    year={2024},
}
```

## Acknowledgments

- Original CCM implementation: [snu-mllab/Context-Memory](https://github.com/snu-mllab/Context-Memory)
- Based on the [Gisting repository](https://github.com/jayelm/gisting)
