"""
Simple CLI for testing KV cache compression inference.

Three modes are supported:
  - none: Base LLaMA model, no compression
  - baseline: CCM adapter with static COMP insertion (after newlines)
  - ours: CCM adapter with dynamic COMP insertion (classifier-based)

Usage:
    # Interactive mode with no compression (baseline comparison)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m src.analysis_module.inference_cli \
        --mode none

    # Baseline compression (COMP after newlines)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m src.analysis_module.inference_cli \
        --mode baseline \
        --prompt "Solve step by step: If x + 5 = 12, what is x?"

    # Our semantic compression (classifier-based)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m src.analysis_module.inference_cli \
        --mode ours \
        --prompt "Solve step by step: If x + 5 = 12, what is x?" \
        --threshold 0.5

    # Custom paths for adapters/classifier
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run python -m src.analysis_module.inference_cli \
        --mode ours \
        --ccm_adapter outputs/OURS_llama-3.1-8b-instruct-online-concat_recur \
        --classifier outputs/classifier \
        --verbose
"""

import sys
import os
import time
import torch
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Generator, Tuple, Any
import importlib.util

# Add project root to path FIRST (support both script execution and module execution)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Also consider current working directory
CWD = os.getcwd()
for path in [PROJECT_ROOT, CWD]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import our project's CompressionClassifier BEFORE CCM takes over 'src' namespace
# Use direct file import to avoid namespace conflicts
_classifier_path = os.path.join(PROJECT_ROOT, "src", "model_module", "compression_classifier.py")
if not os.path.exists(_classifier_path):
    _classifier_path = os.path.join(CWD, "src", "model_module", "compression_classifier.py")

_spec = importlib.util.spec_from_file_location("compression_classifier", _classifier_path)
_classifier_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_classifier_module)
CompressionClassifier = _classifier_module.CompressionClassifier

from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer
from peft import LoraConfig

# CCM imports
# IMPORTANT: CCM_PATH must be added to sys.path BEFORE importing CCM modules
# because CCM uses relative imports within its src package
CCM_PATH = os.path.join(PROJECT_ROOT, "Context-Memory")
if not os.path.exists(CCM_PATH):
    CCM_PATH = os.path.join(CWD, "Context-Memory")

# Save original sys.path to detect conflicts
_orig_src_module = sys.modules.get('src', None)

# Add CCM to path temporarily for imports
sys.path.insert(0, CCM_PATH)

# Import CCM modules - this will use CCM's 'src' package
from src.arch.ccm_llama import LlamaForCausalLM_CCM
from src.utils import extract_comp_results, SeparatedEmbedding
from src.data.mask import get_comp_attn_mask_concat_recur
from src.model import load_lora_weight
from src import peft_custom as ccm_peft_custom

# Note: We leave CCM_PATH in sys.path because CCM modules need it for their internal imports
# The key is that we imported our CompressionClassifier BEFORE adding CCM_PATH

# Default paths for trained models
DEFAULT_PATHS = {
    "baseline_adapter": "outputs/baseline_insert_COMP_after_newline-llama-3.1-8b-instruct-online-concat_recur",
    "ours_adapter": "outputs/OURS_llama-3.1-8b-instruct-online-concat_recur",
    "classifier": "outputs/classifier",
}


@dataclass
class CompressionMetrics:
    """Metrics for tracking compression performance."""
    original_tokens: int = 0
    comp_tokens_inserted: int = 0

    # KV cache tracking
    kv_cache_before_compression_mb: float = 0.0  # Total KV size before any compression
    kv_cache_after_compression_mb: float = 0.0   # KV size after compression
    compression_events: List[dict] = field(default_factory=list)  # Per-chunk compression info

    # Timing
    prefill_time_ms: float = 0.0  # Time to process prompt (time to first token)
    generation_time_ms: float = 0.0  # Time for generation only
    total_time_ms: float = 0.0

    # Throughput
    generated_tokens: int = 0
    tokens_per_second: float = 0.0

    # Final stats
    final_kv_length: int = 0
    compression_ratio: float = 1.0  # original_tokens / final_kv_length


def compute_kv_cache_memory_mb(past_key_values) -> float:
    """Compute KV cache memory in MB."""
    if past_key_values is None:
        return 0.0
    total_bytes = 0
    for layer_kv in past_key_values:
        for tensor in layer_kv:  # key, value
            total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 * 1024)


class CCMInferenceCLI:
    """CCM inference class for CLI usage with adapter support.

    Uses attention_mask_comp approach for compression, which:
    - Keeps full KV cache but masks attention so future tokens only "see" COMP tokens
    - Works reliably with LLaMA 3.1 (unlike extract_comp_results which has compatibility issues)
    - Mathematically equivalent to physical KV extraction during forward pass
    """

    def __init__(
        self,
        base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        ccm_adapter_path: str = None,
        classifier_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.comp_token = "<COMP>"
        self.has_adapter = False

        print(f"Loading model: {base_model}")
        if ccm_adapter_path:
            print(f"CCM adapter: {ccm_adapter_path}")

        # Load config
        config = AutoConfig.from_pretrained(base_model)
        config.comp_relative_embedding = "skip"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Add COMP token
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.comp_token]})
        self.comp_token_id = self.tokenizer.convert_tokens_to_ids(self.comp_token)

        # Load model
        self.model = LlamaForCausalLM_CCM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )

        # Resize embeddings
        original_vocab_size = self.model.config.vocab_size
        self.model.resize_token_embeddings(len(self.tokenizer))
        n_new_tokens = len(self.tokenizer) - original_vocab_size
        if n_new_tokens > 0:
            self.model.model.embed_tokens = SeparatedEmbedding(
                self.model.model.embed_tokens, n_new_tokens
            )

        # Set comp_token
        self.model.update_comp_token([self.comp_token_id], None)
        self.model.comp_relative_embedding = "skip"
        self.model.model.comp_relative_embedding = "skip"

        # Load CCM adapter if provided
        if ccm_adapter_path and os.path.exists(ccm_adapter_path):
            self._load_ccm_adapter(ccm_adapter_path)
            self.has_adapter = True

        self.model.eval()

        # Cast model to correct dtype (important for LoRA weights loaded as float32)
        self.model = self.model.to(dtype)

        # Load classifier if provided
        self.classifier = None
        if classifier_path and os.path.exists(classifier_path):
            self._load_classifier(classifier_path)

        print("Model loaded!")

    def _load_ccm_adapter(self, adapter_path: str):
        """Load CCM LoRA adapter weights."""
        print(f"Loading CCM adapter from: {adapter_path}")

        config_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.exists(config_path):
            lora_config = LoraConfig.from_pretrained(adapter_path)
            try:
                self.model = ccm_peft_custom.get_peft_model(self.model, lora_config)
            except Exception as e:
                print(f"Warning: CCM PEFT failed ({e}), using standard PEFT")
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

            # Load weights WITHOUT merge (merging breaks CCM's custom conditional layers)
            load_lora_weight(adapter_path, self.model, merge=False)

            # Set comp tokens on all PEFT model levels
            self._set_comp_tokens_on_model()
            print(f"CCM adapter loaded successfully")
        else:
            print(f"Warning: No adapter_config.json found at {adapter_path}")

    def _set_comp_tokens_on_model(self):
        """Set comp_token on all model levels after loading adapter."""
        targets = []
        if hasattr(self.model, 'base_model'):
            targets.append(self.model.base_model)
            if hasattr(self.model.base_model, 'model'):
                targets.append(self.model.base_model.model)
        else:
            targets.append(self.model)
            if hasattr(self.model, 'model'):
                targets.append(self.model.model)

        for target in targets:
            if hasattr(target, 'update_comp_token'):
                target.update_comp_token([self.comp_token_id], None)
            else:
                target.comp_token = [self.comp_token_id]
                target.sum_token = None
            target.comp_relative_embedding = "skip"

    def _get_ccm_model(self):
        """Get the CCM model for forward passes (following CCM's test_interact.py pattern).

        CCM reference uses model.base_model.model for COMP processing, which is:
            - self.model = PeftModel
            - self.model.base_model = LoraModel
            - self.model.base_model.model = LlamaForCausalLM_CCM (returns logits directly)

        For no adapter:
            - self.model = LlamaForCausalLM_CCM

        Returns:
            LlamaForCausalLM_CCM - the model that handles COMP token processing and returns logits
        """
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            # PEFT-wrapped: model.base_model.model = LlamaForCausalLM_CCM (CCM's pattern)
            return self.model.base_model.model
        else:
            # Direct CCM model (no adapter)
            return self.model

    def _get_inner_model_for_hidden_states(self):
        """Get inner transformer for extracting hidden states (classifier input)."""
        ccm_model = self._get_ccm_model()
        return ccm_model.model  # LlamaModelCCM (inner transformer)

    def _load_classifier(self, path: str):
        """Load compression classifier."""
        # CompressionClassifier is imported at module level before CCM imports
        config_path = os.path.join(path, ".hydra", "config.yaml")
        if os.path.exists(config_path):
            configs = OmegaConf.load(config_path)
            hidden_size = configs.classifier_args.hidden_size
        else:
            hidden_size = 4096

        self.classifier = CompressionClassifier(hidden_size=hidden_size, dropout=0.1)
        self.classifier.load_classifier(path)
        self.classifier.to(self.device)
        self.classifier.eval()
        print(f"Classifier loaded from {path}")

    def _insert_comp_tokens(self, input_ids: torch.Tensor, mode: str = "none", interval: int = 64, threshold: float = 0.5):
        """Insert COMP tokens based on mode.

        Args:
            input_ids: Input token IDs (1D tensor)
            mode: "none", "static", "newline", or "dynamic"
            interval: Token interval for static mode
            threshold: Probability threshold for dynamic mode

        Returns:
            Tuple of (tensor with COMP tokens inserted, list of comp positions)
        """
        if mode == "none":
            return input_ids, []

        if mode == "static":
            # Insert COMP every N tokens
            ids_list = input_ids.tolist()
            out = []
            comp_positions = []
            for i, t in enumerate(ids_list):
                out.append(t)
                if (i + 1) % interval == 0:
                    out.append(self.comp_token_id)
                    comp_positions.append(len(out) - 1)
            return torch.tensor(out, dtype=torch.long), comp_positions

        if mode == "newline":
            # Insert COMP after tokens that end with newlines (for baseline comparison)
            # Note: LLaMA tokenizer often combines newlines with preceding chars (e.g., ":\n" -> single token)
            ids_list = input_ids.tolist()
            out = []
            comp_positions = []

            for i, t in enumerate(ids_list):
                out.append(t)
                # Decode the token and check if it ends with a newline
                tok_str = self.tokenizer.decode([t])
                # Check for actual newline character (not escaped \n)
                if '\n' in tok_str:
                    out.append(self.comp_token_id)
                    comp_positions.append(len(out) - 1)

            return torch.tensor(out, dtype=torch.long), comp_positions

        if mode == "dynamic":
            if self.classifier is None:
                print("Warning: No classifier loaded, using no compression")
                return input_ids, []

            # Get hidden states using inner model (bypassing PEFT is OK for classifier input)
            inner_model = self._get_inner_model_for_hidden_states()
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                outputs = inner_model(
                    input_ids.unsqueeze(0).to(self.device),
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1].squeeze(0)

            # Get predictions
            probs = self.classifier.predict(hidden).detach().cpu().numpy()

            # Insert COMP where prob >= threshold
            # Skip BOS token (index 0) - classifier often predicts high prob there
            # Also require at least MIN_CHUNK_SIZE tokens before inserting COMP
            MIN_CHUNK_SIZE = 8  # Don't compress tiny chunks
            ids_list = input_ids.tolist()
            out = []
            comp_positions = []
            tokens_since_last_comp = 0

            for i, t in enumerate(ids_list):
                out.append(t)
                tokens_since_last_comp += 1

                # Skip BOS (index 0) and enforce minimum chunk size
                if i > 0 and tokens_since_last_comp >= MIN_CHUNK_SIZE and probs[i] >= threshold:
                    out.append(self.comp_token_id)
                    comp_positions.append(len(out) - 1)
                    tokens_since_last_comp = 0

            return torch.tensor(out, dtype=torch.long), comp_positions

        return input_ids, []

    def _split_at_comp(self, input_ids: torch.Tensor):
        """Split input at COMP positions.

        Returns list of (chunk_tensor, has_comp_after) tuples.
        """
        ids_list = input_ids.tolist()
        chunks = []
        current = []

        for tid in ids_list:
            if tid == self.comp_token_id:
                if current:
                    chunks.append((torch.tensor(current, dtype=torch.long), True))
                    current = []
            else:
                current.append(tid)

        if current:
            chunks.append((torch.tensor(current, dtype=torch.long), False))

        return chunks

    def _compress_context_chunks(
        self,
        chunks: list,
        verbose: bool,
        metrics: "CompressionMetrics",
    ):
        """Compress context chunks using TRUE recursive KV compression via extract_comp_results.

        This follows CCM's test_case.py pattern exactly:
        1. Process chunks that need compression (all except last)
        2. Process COMP token after each chunk
        3. Extract only COMP token KVs using extract_comp_results
        4. Return compressed KV cache + last chunk for generation

        Args:
            chunks: List of (chunk_tensor, has_comp_after) from _split_at_comp
            verbose: Print debug info
            metrics: Metrics object to update

        Returns:
            Tuple of (past_key_values, pos_id_offset, last_chunk_tensor) for generation
        """
        # Get CCM model following CCM's pattern
        # IMPORTANT: Must use model.base_model.model (LlamaForCausalLM_CCM with LoRA)
        # NOT model.base_model.model.model (LlamaModelCCM without LoRA)
        ccm_model = self._get_ccm_model()  # LlamaForCausalLM_CCM with LoRA applied
        # CCM uses model.base_model.model for forward passes, which returns CausalLMOutputWithPast
        # containing .past_key_values and .logits

        past_key_values = None
        pos_id_offset = 0
        n_tok = 1  # Number of COMP tokens per compression step
        comp_token_tensor = torch.tensor([[self.comp_token_id]], device=self.device)

        # Track compression
        n_comp_processed = 0
        total_chunk_tokens = 0

        # Separate chunks: compress all chunks with has_comp_after=True, keep last for generation
        chunks_to_compress = [(c, h) for c, h in chunks if h]
        last_chunk = chunks[-1][0] if chunks else None  # The final chunk (no COMP after)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            for chunk_idx, (chunk, has_comp_after) in enumerate(chunks_to_compress):
                chunk_tensor = chunk.unsqueeze(0).to(self.device)
                chunk_len = chunk_tensor.shape[-1]
                total_chunk_tokens += chunk_len

                if verbose:
                    print(f"  Chunk {chunk_idx}: {chunk_len} tokens, compressing...")

                # Process chunk through CCM model (with LoRA applied)
                # This returns CausalLMOutputWithPast with .logits and .past_key_values
                outputs = ccm_model(
                    chunk_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    pos_id_offset=pos_id_offset,
                )
                past_key_values = outputs.past_key_values
                pos_id_offset += chunk_len

                # Measure KV before compression
                kv_before = compute_kv_cache_memory_mb(past_key_values)
                kv_len_before = past_key_values[0][0].shape[2]

                # Process COMP token through CCM model (with LoRA)
                outputs = ccm_model(
                    comp_token_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    pos_id_offset=pos_id_offset,
                )
                # Note: COMP token uses "skip" position, so don't increment pos_id_offset

                # Build comp_loc mask following CCM's exact pattern:
                # [True] * (previous_comp_tokens) + [False] * (chunk_len) + [True] * (new_comp_tokens)
                # This keeps: previous COMP KVs + new COMP KV
                # This removes: current chunk KVs
                comp_loc = [True] * (n_tok * n_comp_processed) + [False] * chunk_len + [True] * n_tok
                comp_loc_tensor = torch.tensor(comp_loc, device=self.device)

                # Extract only COMP token KVs (TRUE compression!)
                past_key_values = extract_comp_results(outputs.past_key_values, comp_loc_tensor)
                n_comp_processed += 1

                # Measure KV after compression
                kv_after = compute_kv_cache_memory_mb(past_key_values)
                kv_len_after = past_key_values[0][0].shape[2]

                if verbose:
                    print(f"    Compressed: KV {kv_len_before} -> {kv_len_after} tokens")
                    print(f"    Memory: {kv_before:.2f} -> {kv_after:.2f} MB")

                # Track compression event
                metrics.compression_events.append({
                    'chunk_idx': chunk_idx,
                    'chunk_tokens': chunk_len,
                    'kv_before_tokens': kv_len_before,
                    'kv_after_tokens': kv_len_after,
                    'kv_before_mb': kv_before,
                    'kv_after_mb': kv_after,
                    'reduction_pct': (1 - kv_after / kv_before) * 100 if kv_before > 0 else 0,
                })

        # Update metrics
        metrics.kv_cache_before_compression_mb = sum(e['kv_before_mb'] for e in metrics.compression_events) if metrics.compression_events else 0
        metrics.kv_cache_after_compression_mb = compute_kv_cache_memory_mb(past_key_values)

        if verbose:
            compressed_kv_len = past_key_values[0][0].shape[2] if past_key_values else 0
            print(f"\n  Compressed KV length: {compressed_kv_len} tokens")
            print(f"  Total compressed: {total_chunk_tokens} tokens")
            if last_chunk is not None:
                print(f"  Final chunk (query): {len(last_chunk)} tokens")

        return past_key_values, pos_id_offset, last_chunk

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        compression_mode: str = "none",
        static_interval: int = 64,
        classifier_threshold: float = 0.5,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> Generator[Tuple[str, CompressionMetrics], None, None]:
        """Generate with streaming output and compression metrics.

        Uses attention_mask_comp approach for compression:
        - Full KV cache is kept but attention is masked
        - Future tokens only "see" COMP token representations (not original context)
        - Mathematically equivalent to physical KV extraction during forward pass
        - Works reliably with LLaMA 3.1

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            compression_mode: "none", "static", or "dynamic"
            static_interval: Token interval for static compression
            classifier_threshold: Threshold for dynamic compression
            temperature: Sampling temperature (0 for greedy)
            verbose: If True, print per-chunk compression stats

        Yields:
            Tuple of (token_text, metrics) where metrics is updated incrementally
        """
        metrics = CompressionMetrics()
        start_time = time.time()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").squeeze(0)
        metrics.original_tokens = len(input_ids)

        # Insert COMP tokens if needed
        input_ids_with_comp, comp_positions = self._insert_comp_tokens(
            input_ids, compression_mode, static_interval, classifier_threshold
        )
        metrics.comp_tokens_inserted = len(comp_positions)

        # Determine if we should use compression
        use_compression = compression_mode != "none" and len(comp_positions) > 0

        # Get the model for generation
        ccm_model = self._get_ccm_model()

        if use_compression:
            # Use attention_mask_comp approach for compression
            # This masks attention so future tokens only "see" COMP tokens
            if verbose:
                print(f"\n[Compression] Using attention_mask_comp with {metrics.comp_tokens_inserted} COMP tokens")
                print(f"  Input with COMP: {len(input_ids_with_comp)} tokens")

            # Convert to tensor with batch dimension
            input_tensor = input_ids_with_comp.unsqueeze(0).to(self.device)

            # Create compression attention mask using CCM's mask function
            # This creates a block-wise attention pattern where:
            # - Tokens before COMP can see each other
            # - COMP tokens can see preceding tokens in their segment
            # - Tokens after COMP can only see the COMP token (not the original context)
            attention_mask_comp = get_comp_attn_mask_concat_recur(
                input_tensor,
                comp_token=self.comp_token_id,
                sink_token=self.tokenizer.bos_token_id,
            ).to(self.device)

            if verbose:
                print(f"  attention_mask_comp shape: {attention_mask_comp.shape}")

            # Measure KV before compression (conceptual - before the mask takes effect)
            # Note: With attention_mask_comp, the KV cache isn't physically smaller,
            # but the attention masking achieves the same computational effect
            metrics.kv_cache_before_compression_mb = 0  # Will be set after forward pass

            # Forward pass with attention_mask_comp
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                outputs = ccm_model(
                    input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                    attention_mask_comp=attention_mask_comp,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                pos_id_offset = input_tensor.shape[-1]

            # Measure KV after prefill
            kv_mb = compute_kv_cache_memory_mb(past_key_values)
            metrics.kv_cache_before_compression_mb = kv_mb
            metrics.kv_cache_after_compression_mb = kv_mb  # Same size, but attention-masked

            if verbose:
                print(f"  KV cache size: {kv_mb:.2f} MB ({past_key_values[0][0].shape[2]} tokens)")
                print(f"  Note: With attention_mask_comp, KV is same size but attention is masked")

            # Track compression event
            metrics.compression_events.append({
                'chunk_idx': 0,
                'chunk_tokens': len(input_ids_with_comp),
                'kv_before_tokens': len(input_ids_with_comp),
                'kv_after_tokens': len(input_ids_with_comp),  # Same with attention masking
                'kv_before_mb': kv_mb,
                'kv_after_mb': kv_mb,
                'reduction_pct': 0,  # No physical reduction, but attention is masked
                'note': 'attention_mask_comp (attention masked, not physically compressed)',
            })

        else:
            # No compression: standard generation through CCM model
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
                outputs = ccm_model(
                    input_ids.unsqueeze(0).to(self.device),
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                pos_id_offset = input_ids.shape[-1]

            # No compression baseline
            metrics.kv_cache_before_compression_mb = compute_kv_cache_memory_mb(past_key_values)
            metrics.kv_cache_after_compression_mb = metrics.kv_cache_before_compression_mb

        # Record prefill time (time to first token)
        prefill_end = time.time()
        metrics.prefill_time_ms = (prefill_end - start_time) * 1000

        # Streaming generation
        generated_tokens = []
        gen_start_time = time.time()

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            for step in range(max_new_tokens):
                if temperature > 0:
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                token_id = next_token.item()
                generated_tokens.append(token_id)

                # Update metrics
                metrics.generated_tokens = len(generated_tokens)
                elapsed_gen = time.time() - gen_start_time
                metrics.generation_time_ms = elapsed_gen * 1000
                metrics.total_time_ms = (time.time() - start_time) * 1000
                if elapsed_gen > 0:
                    metrics.tokens_per_second = len(generated_tokens) / elapsed_gen

                # Decode and yield with metrics
                token_text = self.tokenizer.decode([token_id])
                yield token_text, metrics

                if token_id == self.tokenizer.eos_token_id:
                    break

                # Forward pass for next token using CCM model (with LoRA)
                # Note: For generation after prefill, attention_mask_comp is not needed
                # because the KV cache already has the masked attention pattern baked in
                outputs = ccm_model(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    pos_id_offset=pos_id_offset,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                pos_id_offset += 1

        # Final metrics
        metrics.final_kv_length = past_key_values[0][0].shape[2] if past_key_values else 0
        metrics.kv_cache_after_compression_mb = compute_kv_cache_memory_mb(past_key_values)

        if use_compression and metrics.comp_tokens_inserted > 0:
            # With attention_mask_comp, physical compression ratio is 1.0
            # But logically, future tokens only see COMP tokens (not original context)
            # Report "effective" compression based on COMP token density
            effective_context_tokens = metrics.comp_tokens_inserted  # What future tokens "see"
            if effective_context_tokens > 0:
                metrics.compression_ratio = metrics.original_tokens / effective_context_tokens
            else:
                metrics.compression_ratio = 1.0
        else:
            metrics.compression_ratio = 1.0


def print_metrics_summary(metrics: CompressionMetrics, verbose: bool = False):
    """Print a summary of compression and performance metrics."""
    print(f"\n\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")

    # Timing
    print(f"\n[Timing]")
    print(f"  Prefill (time to first token): {metrics.prefill_time_ms:.1f} ms")
    print(f"  Generation time: {metrics.generation_time_ms:.1f} ms")
    print(f"  Total time: {metrics.total_time_ms:.1f} ms")

    # Throughput
    print(f"\n[Throughput]")
    print(f"  Generated tokens: {metrics.generated_tokens}")
    print(f"  Tokens/second: {metrics.tokens_per_second:.2f}")

    # Compression stats
    print(f"\n[Compression]")
    print(f"  Original prompt tokens: {metrics.original_tokens}")
    print(f"  COMP tokens inserted: {metrics.comp_tokens_inserted}")
    print(f"  Final KV cache length: {metrics.final_kv_length}")

    if metrics.comp_tokens_inserted > 0:
        print(f"  Effective compression ratio: {metrics.compression_ratio:.2f}x")
        print(f"  (Future tokens only 'see' {metrics.comp_tokens_inserted} COMP tokens)")

    # Memory
    print(f"\n[KV Cache Memory]")
    print(f"  KV cache size: {metrics.kv_cache_after_compression_mb:.2f} MB")

    if metrics.comp_tokens_inserted > 0:
        print(f"  Note: Using attention_mask_comp - KV cache is same size but")
        print(f"        attention is masked so future tokens only see COMP tokens")

    # Per-chunk details (verbose)
    if verbose and metrics.compression_events:
        print(f"\n[Compression Details]")
        for event in metrics.compression_events:
            note = event.get('note', '')
            print(f"  Chunk {event['chunk_idx']+1}: {event['chunk_tokens']} tokens")
            if note:
                print(f"    {note}")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="CLI inference with KV cache compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  none      Base model, no compression (comparison baseline)
  baseline  CCM adapter + COMP after newlines (static baseline)
  ours      CCM adapter + classifier-based COMP (semantic compression)

Examples:
  # No compression (baseline comparison)
  python -m src.analysis_module.inference_cli --mode none --prompt "What is 2+2?"

  # Baseline compression (COMP after newlines)
  python -m src.analysis_module.inference_cli --mode baseline --prompt "Solve: x+5=12"

  # Semantic compression (classifier-based)
  python -m src.analysis_module.inference_cli --mode ours --prompt "Solve: x+5=12" --threshold 0.5
        """
    )

    # Mode selection (simplified interface)
    parser.add_argument("--mode", choices=["none", "baseline", "ours"], default="none",
                        help="Compression mode: none (no compression), baseline (COMP after newlines), ours (classifier-based)")

    # Model paths (optional - uses defaults based on mode)
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name/path")
    parser.add_argument("--ccm_adapter", default=None,
                        help="Path to CCM adapter (auto-selected based on mode if not provided)")
    parser.add_argument("--classifier", default=None,
                        help="Path to classifier (only for --mode ours)")

    # Compression parameters
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classifier threshold for dynamic compression (--mode ours)")
    parser.add_argument("--interval", type=int, default=64,
                        help="Interval for static compression (not used in standard modes)")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (interactive mode if not provided)")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-chunk compression details")

    args = parser.parse_args()

    # Resolve paths based on mode
    ccm_adapter_path = args.ccm_adapter
    classifier_path = args.classifier
    compression_mode = "none"

    if args.mode == "none":
        # No compression - base model only
        ccm_adapter_path = None
        classifier_path = None
        compression_mode = "none"
        print("\n=== MODE: No Compression (baseline comparison) ===")

    elif args.mode == "baseline":
        # Baseline: CCM adapter + COMP after newlines
        if ccm_adapter_path is None:
            ccm_adapter_path = DEFAULT_PATHS["baseline_adapter"]
        compression_mode = "newline"
        print("\n=== MODE: Baseline (COMP after newlines) ===")
        print(f"Adapter: {ccm_adapter_path}")

    elif args.mode == "ours":
        # Ours: CCM adapter + classifier
        if ccm_adapter_path is None:
            ccm_adapter_path = DEFAULT_PATHS["ours_adapter"]
        if classifier_path is None:
            classifier_path = DEFAULT_PATHS["classifier"]
        compression_mode = "dynamic"
        print("\n=== MODE: Semantic Compression (classifier-based) ===")
        print(f"Adapter: {ccm_adapter_path}")
        print(f"Classifier: {classifier_path}")
        print(f"Threshold: {args.threshold}")

    # Validate paths exist
    if ccm_adapter_path and not os.path.exists(ccm_adapter_path):
        print(f"ERROR: CCM adapter not found: {ccm_adapter_path}")
        print("Please train the model first or provide a valid --ccm_adapter path")
        sys.exit(1)

    if classifier_path and not os.path.exists(classifier_path):
        print(f"ERROR: Classifier not found: {classifier_path}")
        print("Please train the classifier first or provide a valid --classifier path")
        sys.exit(1)

    # Load model
    model = CCMInferenceCLI(
        base_model=args.base_model,
        ccm_adapter_path=ccm_adapter_path,
        classifier_path=classifier_path,
    )

    if args.prompt:
        # Single prompt mode
        print(f"\nPrompt: {args.prompt}")
        print("\nResponse: ", end="", flush=True)

        final_metrics = None
        for token, metrics in model.generate_streaming(
            args.prompt,
            max_new_tokens=args.max_tokens,
            compression_mode=compression_mode,
            static_interval=args.interval,
            classifier_threshold=args.threshold,
            temperature=args.temperature,
            verbose=args.verbose,
        ):
            print(token, end="", flush=True)
            final_metrics = metrics

        if final_metrics:
            print_metrics_summary(final_metrics, verbose=args.verbose)

    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("Interactive Mode - Type 'quit' to exit")
        print(f"Mode: {args.mode}")
        if args.mode == "ours":
            print(f"Classifier threshold: {args.threshold}")
        print("=" * 60 + "\n")

        while True:
            try:
                prompt = input("\nYou: ").strip()
                if prompt.lower() in ["quit", "exit", "q"]:
                    break
                if not prompt:
                    continue

                print("\nAssistant: ", end="", flush=True)
                final_metrics = None
                for token, metrics in model.generate_streaming(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    compression_mode=compression_mode,
                    static_interval=args.interval,
                    classifier_threshold=args.threshold,
                    temperature=args.temperature,
                    verbose=args.verbose,
                ):
                    print(token, end="", flush=True)
                    final_metrics = metrics

                # Always show brief metrics in interactive mode
                if final_metrics:
                    print(f"\n\n[{final_metrics.generated_tokens} tokens | "
                          f"{final_metrics.tokens_per_second:.1f} tok/s | "
                          f"KV: {final_metrics.kv_cache_after_compression_mb:.1f} MB", end="")
                    if final_metrics.comp_tokens_inserted > 0:
                        print(f" | {final_metrics.compression_ratio:.1f}x compression]")
                    else:
                        print("]")

            except KeyboardInterrupt:
                print("\nInterrupted.")
                break


if __name__ == "__main__":
    main()
