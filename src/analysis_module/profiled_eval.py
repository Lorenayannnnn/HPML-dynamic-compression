"""
Profiled evaluation script for CCM compression comparison.

Builds on eval_compression.py and adds:
- PyTorch profiler integration (per HPML guidelines)
- Detailed classifier overhead analysis
- "No compression" baseline for comparison
- Comprehensive metrics table

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. uv run python src/analysis_module/profiled_eval.py
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports (so 'src.module' and 'module' both work)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.profiler
import numpy as np
import json
import time
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer

# Import peft_loader BEFORE eval_compression (eval_compression messes with sys.modules)
from model_module.peft_loader import load_peft_model
from model_module.compression_classifier import CompressionClassifier
from analysis_module.gsm8k_utils import extract_gsm8k_answer, verify_gsm8k_answer

# Import from existing eval_compression module (this imports CCM and modifies sys.modules)
from analysis_module.eval_compression import (
    generate_with_compression,
    get_comp_and_newline_tokens,
    extract_comp_results,
)

# import wanbd logging for eval
from src.common_utils import init_wandb_eval, log_profiled_eval_results


@dataclass
class ProfiledMetrics:
    """Detailed metrics for a single sample with profiling data."""

    sample_id: int
    method: str
    # Core metrics
    accuracy: float
    comp_tokens: int
    tokens_generated: int
    kv_cache_mb: float
    latency_seconds: float
    throughput_tokens_per_sec: float
    # Classifier overhead (dynamic method only)
    classifier_calls: int = 0
    classifier_time_ms: float = 0.0
    classifier_overhead_pct: float = 0.0
    # Memory metrics
    peak_memory_mb: float = 0.0
    # Profiler metrics (if enabled)
    cuda_time_ms: float = 0.0
    cpu_time_ms: float = 0.0


@dataclass
class AggregatedResults:
    """Aggregated results across all samples for a method."""

    method: str
    num_samples: int
    # Accuracy
    accuracy_pct: float
    # Compression
    avg_comp_tokens: float
    std_comp_tokens: float
    avg_tokens_generated: float
    avg_kv_cache_mb: float
    std_kv_cache_mb: float
    compression_ratio: float
    # Performance
    avg_latency_sec: float
    std_latency_sec: float
    avg_throughput_tps: float
    total_time_sec: float
    # Classifier overhead
    avg_classifier_time_ms: float = 0.0
    classifier_overhead_pct: float = 0.0
    # Memory
    avg_peak_memory_mb: float = 0.0
    classifier_memory_mb: float = 0.0


def generate_no_compression(
    model, tokenizer, input_ids, max_new_tokens=256, device="cuda"
):
    """
    Generate without any COMP token insertion (vanilla LLM baseline).

    Returns:
        Tuple of (generated_ids, 0, kv_size_bytes, latency)
    """
    generated_ids = input_ids.clone().to(device)
    past_key_values = None

    t_start = time.perf_counter()

    for step in range(max_new_tokens):
        if past_key_values is None:
            curr_input = generated_ids
        else:
            curr_input = generated_ids[:, -1:]

        with torch.no_grad():
            outputs = model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    # Calculate KV cache size
    if past_key_values is not None:
        kv_len = past_key_values[0][0].shape[2]
        n_layers = len(past_key_values)
        n_kv_heads = past_key_values[0][0].shape[1]
        head_dim = past_key_values[0][0].shape[3]
        dtype_size = past_key_values[0][0].element_size()
        kv_size = 2 * n_layers * kv_len * n_kv_heads * head_dim * dtype_size
    else:
        kv_size = 0

    return generated_ids, 0, kv_size, t_end - t_start


def generate_with_classifier_profiled(
    model,
    tokenizer,
    input_ids,
    classifier,
    comp_token_id,
    threshold=0.85,
    max_new_tokens=256,
    device="cuda",
    compress_kv=True,
):
    """
    Generate with classifier-based COMP insertion + detailed timing breakdown.

    Returns dict with generation results and classifier profiling data.
    """
    from analysis_module.eval_compression import (
        _extract_sink_and_comp,
        extract_comp_results,
    )

    generated_ids = input_ids.clone().to(device)
    past_key_values = None
    pos_id_offset = 0
    comp_count = 0
    comp_positions_in_kv = []
    sink_positions = [0]
    comp_token_tensor = torch.tensor([[comp_token_id]], device=device)

    # Timing breakdown
    classifier_time_total = 0.0
    classifier_calls = 0
    model_forward_time = 0.0
    comp_forward_time = 0.0

    t_start = time.perf_counter()

    for step in range(max_new_tokens):
        if past_key_values is None:
            curr_input = generated_ids
        else:
            curr_input = generated_ids[:, -1:]

        # Model forward pass timing
        t_model_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                input_ids=curr_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                pos_id_offset=pos_id_offset,
            )
        torch.cuda.synchronize()
        model_forward_time += time.perf_counter() - t_model_start

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        pos_id_offset += curr_input.shape[-1]

        if next_token.item() == tokenizer.eos_token_id:
            break

        # Classifier timing
        hidden_states = outputs.hidden_states[-1][:, -1:, :]

        t_cls_start = time.perf_counter()
        with torch.no_grad():
            comp_prob = classifier.predict(hidden_states)[0, 0].item()
        torch.cuda.synchronize()
        t_cls_end = time.perf_counter()

        classifier_calls += 1
        classifier_time_total += t_cls_end - t_cls_start

        should_insert_comp = comp_prob >= threshold

        if should_insert_comp:
            comp_count += 1

            # Add COMP token to generated sequence (so it appears in output text)
            generated_ids = torch.cat([generated_ids, comp_token_tensor], dim=1)

            t_comp_start = time.perf_counter()
            with torch.no_grad():
                comp_outputs = model(
                    input_ids=comp_token_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    pos_id_offset=pos_id_offset,
                )
            torch.cuda.synchronize()
            comp_forward_time += time.perf_counter() - t_comp_start

            past_key_values = comp_outputs.past_key_values

            if compress_kv and extract_comp_results is not None:
                kv_len = past_key_values[0][0].shape[2]
                current_comp_pos = kv_len - 1
                all_comp_positions = comp_positions_in_kv + [current_comp_pos]

                past_key_values = _extract_sink_and_comp(
                    past_key_values, sink_positions, all_comp_positions, kv_len, device
                )
                comp_positions_in_kv = list(range(1, comp_count + 1))
            else:
                kv_len = past_key_values[0][0].shape[2]
                comp_positions_in_kv.append(kv_len - 1)

    t_end = time.perf_counter()
    total_latency = t_end - t_start

    # Calculate KV size
    if past_key_values is not None:
        kv_len = past_key_values[0][0].shape[2]
        n_layers = len(past_key_values)
        n_kv_heads = past_key_values[0][0].shape[1]
        head_dim = past_key_values[0][0].shape[3]
        dtype_size = past_key_values[0][0].element_size()
        kv_size = 2 * n_layers * kv_len * n_kv_heads * head_dim * dtype_size
    else:
        kv_size = 0

    return {
        "generated_ids": generated_ids,
        "comp_count": comp_count,
        "kv_size_bytes": kv_size,
        "total_latency": total_latency,
        "classifier_calls": classifier_calls,
        "classifier_time_total": classifier_time_total,
        "model_forward_time": model_forward_time,
        "comp_forward_time": comp_forward_time,
    }


class ProfiledEvaluator:
    """Evaluator with PyTorch profiler integration."""

    def __init__(
        self,
        baseline_adapter_path: str,
        dynamic_adapter_path: str,
        classifier_path: str,
        base_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.baseline_adapter_path = baseline_adapter_path
        self.dynamic_adapter_path = dynamic_adapter_path
        self.classifier_path = classifier_path
        self.base_model_id = base_model_id
        self.device = device
        self.dtype = dtype

        # Models and tokenizer
        self.tokenizer = None
        self.comp_token_id = None
        self.newline_token_ids = None
        self.vanilla_model = None  # Original Llama (no CCM fine-tuning)
        self.baseline_model = None
        self.dynamic_model = None
        self.classifier = None

        # Profiling info
        self.classifier_memory_mb = 0.0
        self.classifier_params = 0

    def load_tokenizer(self):
        """Load tokenizer only."""
        print(f"Loading tokenizer from {self.baseline_adapter_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.baseline_adapter_path,
            trust_remote_code=True,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.comp_token_id, self.newline_token_ids = get_comp_and_newline_tokens(
            self.tokenizer
        )

    def load_vanilla_model(self):
        """Load vanilla Llama model (no CCM fine-tuning)."""
        from transformers import AutoModelForCausalLM

        print(f"\nLoading {self.base_model_id} (no CCM fine-tuning)...")
        self.vanilla_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=self.dtype,
            device_map=self.device,
        )

    def unload_vanilla_model(self):
        """Free vanilla model from GPU."""
        if self.vanilla_model is not None:
            del self.vanilla_model
            self.vanilla_model = None
            torch.cuda.empty_cache()
            print("Vanilla model unloaded.")

    def load_baseline_model(self):
        """Load baseline CCM model."""
        print(f"\nLoading baseline model...")
        self.baseline_model, _ = load_peft_model(
            base_model_id=self.base_model_id,
            adapter_path=self.baseline_adapter_path,
            device=self.device,
            dtype=self.dtype,
        )

    def unload_baseline_model(self):
        """Free baseline model from GPU."""
        if self.baseline_model is not None:
            del self.baseline_model
            self.baseline_model = None
            torch.cuda.empty_cache()
            print("Baseline model unloaded.")

    def load_dynamic_model_and_classifier(self):
        """Load dynamic CCM model and classifier."""
        print(f"\nLoading dynamic model...")
        self.dynamic_model, _ = load_peft_model(
            base_model_id=self.base_model_id,
            adapter_path=self.dynamic_adapter_path,
            device=self.device,
            dtype=self.dtype,
        )

        # Load classifier with memory measurement
        print(f"\nLoading classifier from {self.classifier_path}...")
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

        self.classifier = CompressionClassifier(
            hidden_size=self.dynamic_model.config.hidden_size, dropout=0.1
        )
        self.classifier.load_state_dict(torch.load(self.classifier_path))
        self.classifier.eval()
        self.classifier = self.classifier.to(self.device).to(self.dtype)

        mem_after = torch.cuda.memory_allocated()
        self.classifier_memory_mb = (mem_after - mem_before) / (1024 * 1024)
        self.classifier_params = sum(p.numel() for p in self.classifier.parameters())

        print(f"  Classifier memory: {self.classifier_memory_mb:.2f} MB")
        print(f"  Classifier parameters: {self.classifier_params:,}")

    def unload_dynamic_model(self):
        """Free dynamic model and classifier from GPU."""
        if self.dynamic_model is not None:
            del self.dynamic_model
            self.dynamic_model = None
        if self.classifier is not None:
            del self.classifier
            self.classifier = None
        torch.cuda.empty_cache()
        print("Dynamic model and classifier unloaded.")

    def profile_classifier_standalone(
        self, num_iterations: int = 100
    ) -> Dict[str, float]:
        """Profile the classifier in isolation using PyTorch profiler."""
        print("\n" + "=" * 60)
        print("CLASSIFIER STANDALONE PROFILING")
        print("=" * 60)

        # Create dummy input matching real usage
        hidden_size = self.dynamic_model.config.hidden_size
        dummy_hidden = torch.randn(
            1, 1, hidden_size, device=self.device, dtype=self.dtype
        )

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.classifier.predict(dummy_hidden)
        torch.cuda.synchronize()

        # Profile with PyTorch profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = self.classifier.predict(dummy_hidden)
                torch.cuda.synchronize()

        # Extract profiler stats
        key_averages = prof.key_averages()

        # Get total device (CUDA) and CPU time
        # Note: PyTorch renamed cuda_time -> device_time for attribute access
        total_device_time = (
            sum(item.self_device_time_total for item in key_averages) / 1000
        )  # ms
        total_cpu_time = (
            sum(item.self_cpu_time_total for item in key_averages) / 1000
        )  # ms

        # Manual timing for comparison
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = self.classifier.predict(dummy_hidden)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

        avg_time_ms = np.mean(times)
        std_time_ms = np.std(times)

        results = {
            "avg_latency_ms": avg_time_ms,
            "std_latency_ms": std_time_ms,
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "profiler_cuda_time_ms": total_device_time / num_iterations,
            "profiler_cpu_time_ms": total_cpu_time / num_iterations,
            "memory_mb": self.classifier_memory_mb,
            "num_parameters": self.classifier_params,
        }

        print(f"\nClassifier Latency (over {num_iterations} iterations):")
        print(f"  Average: {avg_time_ms:.4f} ms")
        print(f"  Std Dev: {std_time_ms:.4f} ms")
        print(f"  Min:     {np.min(times):.4f} ms")
        print(f"  Max:     {np.max(times):.4f} ms")
        print(f"\nPyTorch Profiler Stats (per call):")
        print(f"  CUDA time: {total_device_time / num_iterations:.4f} ms")
        print(f"  CPU time:  {total_cpu_time / num_iterations:.4f} ms")
        print(f"\nMemory:")
        print(f"  Classifier size: {self.classifier_memory_mb:.2f} MB")
        print(f"  Parameters: {self.classifier_params:,}")

        # Print detailed profiler table
        print(f"\nDetailed Profiler Breakdown (top 10 ops):")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

        return results

    def evaluate_method_with_checkpoints(
        self,
        method: str,
        test_data: List[Dict],
        threshold: float,
        max_new_tokens: int,
        output_path: str,
        all_results: Dict[str, List[ProfiledMetrics]],
        classifier_profile: Optional[Dict],
        existing_results: List[ProfiledMetrics],
    ) -> List[ProfiledMetrics]:
        """Evaluate a method with checkpointing after each sample."""
        results = list(existing_results)  # Start with existing results
        start_idx = len(results)

        for i, sample in tqdm(
            enumerate(test_data), total=len(test_data), desc=f"Eval {method}"
        ):
            # Skip already completed samples
            if i < start_idx:
                continue

            gt_answer = extract_gsm8k_answer(sample)
            question = sample.get("question", "")
            input_ids = self.tokenizer.encode(question, return_tensors="pt").to(
                self.device
            )

            torch.cuda.reset_peak_memory_stats()

            if method == "compression_none":
                gen_ids, comp_count, kv_bytes, latency = generate_no_compression(
                    self.vanilla_model,
                    self.tokenizer,
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    device=self.device,
                )
                classifier_calls = 0
                classifier_time = 0.0

            elif method == "compression_newline":
                t0 = time.perf_counter()
                gen_ids, comp_count, kv_bytes = generate_with_compression(
                    self.baseline_model,
                    self.tokenizer,
                    input_ids,
                    classifier=None,
                    comp_token_id=self.comp_token_id,
                    newline_token_id_list=self.newline_token_ids,
                    use_classifier=False,
                    max_new_tokens=max_new_tokens,
                    device=self.device,
                    compress_kv=True,
                )
                latency = time.perf_counter() - t0
                classifier_calls = 0
                classifier_time = 0.0

            elif method.startswith("compression_classifier"):
                # Handle compression_classifier_{threshold} methods
                result = generate_with_classifier_profiled(
                    self.dynamic_model,
                    self.tokenizer,
                    input_ids,
                    self.classifier,
                    self.comp_token_id,
                    threshold=threshold,
                    max_new_tokens=max_new_tokens,
                    device=self.device,
                    compress_kv=True,
                )
                gen_ids = result["generated_ids"]
                comp_count = result["comp_count"]
                kv_bytes = result["kv_size_bytes"]
                latency = result["total_latency"]
                classifier_calls = result["classifier_calls"]
                classifier_time = result["classifier_time_total"]
            else:
                raise ValueError(f"Unknown method: {method}")

            # Check accuracy
            gen_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=False)
            is_correct = verify_gsm8k_answer(gt_answer, gen_text)

            # Calculate metrics
            tokens_generated = gen_ids.shape[1] - input_ids.shape[1]
            kv_cache_mb = kv_bytes / (1024 * 1024)
            throughput = tokens_generated / latency if latency > 0 else 0
            classifier_overhead = (
                (classifier_time / latency * 100)
                if latency > 0 and classifier_time > 0
                else 0
            )
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

            metrics = ProfiledMetrics(
                sample_id=i,
                method=method,
                accuracy=1.0 if is_correct else 0.0,
                comp_tokens=comp_count,
                tokens_generated=tokens_generated,
                kv_cache_mb=kv_cache_mb,
                latency_seconds=latency,
                throughput_tokens_per_sec=throughput,
                classifier_calls=classifier_calls,
                classifier_time_ms=classifier_time * 1000,
                classifier_overhead_pct=classifier_overhead,
                peak_memory_mb=peak_memory,
            )
            results.append(metrics)

            # Save checkpoint after each sample
            all_results[method] = results
            self._save_checkpoint(
                output_path,
                all_results,
                classifier_profile,
                threshold,
                max_new_tokens,
                len(test_data),
            )

            if (i + 1) % 10 == 0:
                print(
                    f"Checkpoint saved: {i + 1}/{len(test_data)} samples for {method}"
                )

        return results

    def _save_checkpoint(
        self,
        output_path: str,
        all_results: Dict[str, List[ProfiledMetrics]],
        classifier_profile: Optional[Dict],
        threshold: float,
        max_new_tokens: int,
        num_samples: int,
    ):
        """Save current progress to JSON file."""
        checkpoint = {
            "config": {
                "threshold": threshold,
                "max_new_tokens": max_new_tokens,
                "num_samples": num_samples,
            },
            "classifier_profile": classifier_profile,
            "per_sample": {k: [asdict(s) for s in v] for k, v in all_results.items()},
            # Don't include aggregated - will be computed at the end
        }
        with open(output_path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def _print_method_stats(
        self,
        method: str,
        results: List[ProfiledMetrics],
        baseline_kv_mb: Optional[float] = None,
    ):
        """Print statistics for a completed method (consistent with eval_compression.py style)."""
        n = len(results)
        if n == 0:
            print(f"\nNo results for {method}")
            return

        accuracy = np.mean([r.accuracy for r in results]) * 100
        avg_comp_tokens = np.mean([r.comp_tokens for r in results])
        std_comp_tokens = np.std([r.comp_tokens for r in results])
        avg_kv_cache_mb = np.mean([r.kv_cache_mb for r in results])
        std_kv_cache_mb = np.std([r.kv_cache_mb for r in results])
        avg_latency = np.mean([r.latency_seconds for r in results])
        std_latency = np.std([r.latency_seconds for r in results])
        avg_throughput = np.mean([r.throughput_tokens_per_sec for r in results])
        avg_tokens_gen = np.mean([r.tokens_generated for r in results])
        total_time = sum(r.latency_seconds for r in results)

        # Compression ratio (vs baseline if available)
        if baseline_kv_mb and avg_kv_cache_mb > 0:
            compression_ratio = baseline_kv_mb / avg_kv_cache_mb
        else:
            compression_ratio = 1.0

        print(f"\n{'=' * 60}")
        print(f"Results: {method}")
        print(f"{'=' * 60}")
        print(f"Accuracy:            {accuracy:.1f}%")
        print(f"Avg COMP tokens:     {avg_comp_tokens:.2f} ± {std_comp_tokens:.2f}")
        print(f"Avg KV Cache (MB):   {avg_kv_cache_mb:.2f} ± {std_kv_cache_mb:.2f}")
        if baseline_kv_mb:
            print(f"Compression Ratio:   {compression_ratio:.2f}x")
        print(f"Avg Latency (s):     {avg_latency:.3f} ± {std_latency:.3f}")
        print(f"Avg Throughput:      {avg_throughput:.1f} tokens/sec")
        print(f"Avg Tokens Gen:      {avg_tokens_gen:.1f}")
        print(f"Total Time:          {total_time:.1f}s")
        print(f"Examples:            {n}")

        # Additional classifier stats for classifier methods
        if method.startswith("compression_classifier"):
            avg_classifier_time = np.mean([r.classifier_time_ms for r in results])
            avg_classifier_overhead = np.mean(
                [r.classifier_overhead_pct for r in results]
            )
            print(f"Avg Classifier Time: {avg_classifier_time:.3f} ms")
            print(f"Classifier Overhead: {avg_classifier_overhead:.2f}%")

        print(f"{'=' * 60}")

    def _build_final_results(
        self,
        all_results: Dict[str, List[ProfiledMetrics]],
        classifier_profile: Dict,
        thresholds: List[float],
        max_new_tokens: int,
        test_data: List[Dict],
    ) -> Dict[str, Any]:
        """Build final results dict with aggregated metrics."""
        # Calculate baseline KV for compression ratio
        no_comp_kv = np.mean([r.kv_cache_mb for r in all_results["compression_none"]])

        # Aggregate results
        aggregated = {}
        for method, samples in all_results.items():
            n = len(samples)
            avg_kv = np.mean([r.kv_cache_mb for r in samples])
            total_time = sum(r.latency_seconds for r in samples)

            aggregated[method] = AggregatedResults(
                method=method,
                num_samples=n,
                accuracy_pct=np.mean([r.accuracy for r in samples]) * 100,
                avg_comp_tokens=np.mean([r.comp_tokens for r in samples]),
                std_comp_tokens=np.std([r.comp_tokens for r in samples]),
                avg_tokens_generated=np.mean([r.tokens_generated for r in samples]),
                avg_kv_cache_mb=avg_kv,
                std_kv_cache_mb=np.std([r.kv_cache_mb for r in samples]),
                compression_ratio=no_comp_kv / avg_kv if avg_kv > 0 else 1.0,
                avg_latency_sec=np.mean([r.latency_seconds for r in samples]),
                std_latency_sec=np.std([r.latency_seconds for r in samples]),
                avg_throughput_tps=np.mean(
                    [r.throughput_tokens_per_sec for r in samples]
                ),
                total_time_sec=total_time,
                avg_classifier_time_ms=np.mean([r.classifier_time_ms for r in samples]),
                classifier_overhead_pct=np.mean(
                    [r.classifier_overhead_pct for r in samples]
                ),
                avg_peak_memory_mb=np.mean([r.peak_memory_mb for r in samples]),
                classifier_memory_mb=self.classifier_memory_mb
                if method.startswith("compression_classifier")
                else 0,
            )

        return {
            "config": {
                "thresholds": thresholds,
                "max_new_tokens": max_new_tokens,
                "num_samples": len(test_data),
            },
            "classifier_profile": classifier_profile,
            "aggregated": {k: asdict(v) for k, v in aggregated.items()},
            "per_sample": {k: [asdict(s) for s in v] for k, v in all_results.items()},
        }

    def run_full_evaluation(
        self,
        test_data: List[Dict],
        thresholds: List[float] = [0.5, 0.7, 0.9],
        max_new_tokens: int = 256,
        output_path: str = "outputs/profiled_eval_results.json",
    ) -> Dict[str, Any]:
        """Run full evaluation across all methods, with checkpoint/resume support.

        Methods evaluated:
        - compression_none: Vanilla LLM without any compression
        - compression_newline: Insert COMP after newlines (static baseline)
        - compression_classifier_{threshold}: Classifier-based COMP insertion for each threshold
        """

        # Build list of all methods to evaluate
        classifier_methods = [f"compression_classifier_{t}" for t in thresholds]
        all_methods = ["compression_none", "compression_newline"] + classifier_methods

        # Load existing results if available (for resume)
        all_results = {}
        classifier_profile = None

        if os.path.exists(output_path):
            print(f"\nFound existing results at {output_path}, loading for resume...")
            with open(output_path, "r") as f:
                existing = json.load(f)

            # Restore previous results
            if "per_sample" in existing:
                for method, samples in existing["per_sample"].items():
                    all_results[method] = [ProfiledMetrics(**s) for s in samples]
                    print(f"  Loaded {len(samples)} samples for {method}")

            if "classifier_profile" in existing and existing["classifier_profile"]:
                classifier_profile = existing["classifier_profile"]
                # Restore classifier memory info for final results
                self.classifier_memory_mb = classifier_profile.get("memory_mb", 0.0)
                self.classifier_params = classifier_profile.get("num_parameters", 0)
                print(f"  Loaded classifier profile")

        # Check if all methods are complete
        all_complete = all(
            method in all_results and len(all_results[method]) == len(test_data)
            for method in all_methods
        )

        if all_complete and classifier_profile is not None:
            print("\nAll evaluations already complete! Nothing to do.")
            return self._build_final_results(
                all_results, classifier_profile, thresholds, max_new_tokens, test_data
            )

        # Load tokenizer first (shared across all methods)
        self.load_tokenizer()

        # === 1. No Compression (vanilla Llama) ===
        if "compression_none" not in all_results or len(
            all_results["compression_none"]
        ) < len(test_data):
            print(f"\n{'=' * 60}")
            print(f"Evaluating: compression_none")
            print(f"{'=' * 60}")
            self.load_vanilla_model()

            # Get starting index for resume
            start_idx = len(all_results.get("compression_none", []))
            if start_idx > 0:
                print(f"Resuming from sample {start_idx}/{len(test_data)}")

            all_results["compression_none"] = self.evaluate_method_with_checkpoints(
                method="compression_none",
                test_data=test_data,
                threshold=0.0,  # Not used for this method
                max_new_tokens=max_new_tokens,
                output_path=output_path,
                all_results=all_results,
                classifier_profile=classifier_profile,
                existing_results=all_results.get("compression_none", []),
            )
            self.unload_vanilla_model()
            self._print_method_stats(
                "compression_none", all_results["compression_none"]
            )
        else:
            print(
                f"\ncompression_none already complete ({len(all_results['compression_none'])} samples), skipping."
            )
            self._print_method_stats(
                "compression_none", all_results["compression_none"]
            )

        # Get baseline KV for compression ratio calculation
        baseline_kv_mb = (
            np.mean([r.kv_cache_mb for r in all_results["compression_none"]])
            if all_results.get("compression_none")
            else None
        )

        # === 2. Newline-based Compression ===
        if "compression_newline" not in all_results or len(
            all_results["compression_newline"]
        ) < len(test_data):
            print(f"\n{'=' * 60}")
            print(f"Evaluating: compression_newline")
            print(f"{'=' * 60}")
            self.load_baseline_model()

            start_idx = len(all_results.get("compression_newline", []))
            if start_idx > 0:
                print(f"Resuming from sample {start_idx}/{len(test_data)}")

            all_results["compression_newline"] = self.evaluate_method_with_checkpoints(
                method="compression_newline",
                test_data=test_data,
                threshold=0.0,  # Not used for this method
                max_new_tokens=max_new_tokens,
                output_path=output_path,
                all_results=all_results,
                classifier_profile=classifier_profile,
                existing_results=all_results.get("compression_newline", []),
            )
            self.unload_baseline_model()
            self._print_method_stats(
                "compression_newline",
                all_results["compression_newline"],
                baseline_kv_mb,
            )
        else:
            print(
                f"\ncompression_newline already complete ({len(all_results['compression_newline'])} samples), skipping."
            )
            self._print_method_stats(
                "compression_newline",
                all_results["compression_newline"],
                baseline_kv_mb,
            )

        # === 3. Classifier-based Compression (multiple thresholds) ===
        # Check if any classifier method needs to be run
        classifier_methods_needed = [
            m
            for m in classifier_methods
            if m not in all_results or len(all_results[m]) < len(test_data)
        ]

        if classifier_methods_needed:
            # Load model and classifier once for all thresholds
            self.load_dynamic_model_and_classifier()

            # Profile classifier while it's loaded (if not already done)
            if classifier_profile is None:
                classifier_profile = self.profile_classifier_standalone()
            else:
                print("Classifier profile already exists, skipping profiling.")

            # Run each threshold
            for threshold in thresholds:
                method_name = f"compression_classifier_{threshold}"

                if method_name not in all_results or len(
                    all_results[method_name]
                ) < len(test_data):
                    print(f"\n{'=' * 60}")
                    print(f"Evaluating: {method_name}")
                    print(f"{'=' * 60}")

                    start_idx = len(all_results.get(method_name, []))
                    if start_idx > 0:
                        print(f"Resuming from sample {start_idx}/{len(test_data)}")

                    all_results[method_name] = self.evaluate_method_with_checkpoints(
                        method=method_name,
                        test_data=test_data,
                        threshold=threshold,
                        max_new_tokens=max_new_tokens,
                        output_path=output_path,
                        all_results=all_results,
                        classifier_profile=classifier_profile,
                        existing_results=all_results.get(method_name, []),
                    )
                    self._print_method_stats(
                        method_name, all_results[method_name], baseline_kv_mb
                    )
                else:
                    print(
                        f"\n{method_name} already complete ({len(all_results[method_name])} samples), skipping."
                    )
                    self._print_method_stats(
                        method_name, all_results[method_name], baseline_kv_mb
                    )

            self.unload_dynamic_model()
        else:
            # All classifier methods complete, just print stats
            for threshold in thresholds:
                method_name = f"compression_classifier_{threshold}"
                print(
                    f"\n{method_name} already complete ({len(all_results[method_name])} samples), skipping."
                )
                self._print_method_stats(
                    method_name, all_results[method_name], baseline_kv_mb
                )

        # Build and return final results
        return self._build_final_results(
            all_results, classifier_profile, thresholds, max_new_tokens, test_data
        )


def print_results_table(results: Dict[str, Any]):
    """Print formatted results table (consistent with eval_compression.py style)."""
    agg = results["aggregated"]
    cls_prof = results["classifier_profile"]
    thresholds = results["config"].get("thresholds", [])

    def print_method_results(title, r, show_classifier_stats=False):
        """Print results for a single method."""
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Accuracy:            {r['accuracy_pct']:.1f}%")
        print(
            f"Avg COMP tokens:     {r['avg_comp_tokens']:.2f} ± {r['std_comp_tokens']:.2f}"
        )
        print(
            f"Avg KV Cache (MB):   {r['avg_kv_cache_mb']:.2f} ± {r['std_kv_cache_mb']:.2f}"
        )
        print(f"Compression Ratio:   {r['compression_ratio']:.2f}x")
        print(
            f"Avg Latency (s):     {r['avg_latency_sec']:.3f} ± {r['std_latency_sec']:.3f}"
        )
        print(f"Avg Throughput:      {r['avg_throughput_tps']:.1f} tokens/sec")
        print(f"Avg Tokens Gen:      {r['avg_tokens_generated']:.1f}")
        print(f"Total Time:          {r['total_time_sec']:.1f}s")
        print(f"Examples:            {r['num_samples']}")
        if show_classifier_stats:
            print(f"Avg Classifier Time: {r['avg_classifier_time_ms']:.3f} ms")
            print(f"Classifier Overhead: {r['classifier_overhead_pct']:.2f}%")

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    # Print individual method results
    print_method_results(
        "compression_none (vanilla LLM, no COMP tokens)", agg["compression_none"]
    )
    print_method_results(
        "compression_newline (insert COMP after newlines) [recursive KV compression]",
        agg["compression_newline"],
    )

    # Print results for each threshold
    for t in thresholds:
        method_name = f"compression_classifier_{t}"
        if method_name in agg:
            print_method_results(
                f"{method_name} [recursive KV compression]",
                agg[method_name],
                show_classifier_stats=True,
            )

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    # Build dynamic header based on thresholds
    classifier_headers = [f"Cls_{t}" for t in thresholds]
    header_parts = ["Metric".ljust(22), "None".rjust(10), "Newline".rjust(10)] + [
        h.rjust(10) for h in classifier_headers
    ]
    header = " ".join(header_parts)
    print(f"\n{header}")
    print("-" * len(header))

    # Helper to build row
    def print_row(metric_name, key, format_str="{:>10.2f}", suffix=""):
        parts = [metric_name.ljust(22)]
        parts.append(format_str.format(agg["compression_none"][key]) + suffix)
        parts.append(format_str.format(agg["compression_newline"][key]) + suffix)
        for t in thresholds:
            method = f"compression_classifier_{t}"
            if method in agg:
                parts.append(format_str.format(agg[method][key]) + suffix)
        print(" ".join(parts))

    print_row("Accuracy (%)", "accuracy_pct", "{:>9.1f}", "%")
    print_row("Avg COMP Tokens", "avg_comp_tokens", "{:>10.2f}")
    print_row("Avg KV Cache (MB)", "avg_kv_cache_mb", "{:>10.2f}")
    print_row("Compression Ratio", "compression_ratio", "{:>9.2f}", "x")
    print_row("Avg Latency (s)", "avg_latency_sec", "{:>10.3f}")
    print_row("Throughput (tok/s)", "avg_throughput_tps", "{:>10.1f}")

    print("-" * len(header))

    # Classifier overhead analysis
    print("\n" + "=" * 70)
    print("CLASSIFIER OVERHEAD ANALYSIS")
    print("=" * 70)

    print(f"Classifier Memory:           {cls_prof['memory_mb']:.2f} MB")
    print(f"Classifier Parameters:       {cls_prof['num_parameters']:,}")
    print(
        f"Avg Classifier Latency:      {cls_prof['avg_latency_ms']:.4f} ms (standalone)"
    )

    # Show overhead for each threshold
    for t in thresholds:
        method = f"compression_classifier_{t}"
        if method in agg:
            dyn = agg[method]
            latency_overhead = (
                (dyn["avg_latency_sec"] - agg["compression_none"]["avg_latency_sec"])
                / agg["compression_none"]["avg_latency_sec"]
                * 100
            )
            print(f"\n  Threshold {t}:")
            print(
                f"    Avg Classifier Time/Sample:  {dyn['avg_classifier_time_ms']:.3f} ms"
            )
            print(
                f"    Classifier Overhead:         {dyn['classifier_overhead_pct']:.2f}% of total latency"
            )
            print(f"    Latency vs compression_none: {latency_overhead:+.1f}%")

    print("\n" + "=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profiled CCM compression evaluation")
    parser.add_argument("--test_dataset", type=str, default="data/gsm8k-test-20.json")
    parser.add_argument(
        "--baseline_model",
        type=str,
        default="outputs/baseline_insert_COMP_after_newline-llama-3.1-8b-instruct-online-concat_recur",
    )
    parser.add_argument(
        "--dynamic_model",
        type=str,
        default="outputs/OURS_llama-3.1-8b-instruct-online-concat_recur",
    )
    parser.add_argument(
        "--classifier_path",
        type=str,
        default="outputs/classifier/compression_classifier.pt",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.5, 0.7, 0.9],
        help="List of classifier thresholds to evaluate (default: 0.5 0.7 0.9)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--output", type=str, default="outputs/profiled_eval_results.json"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="hpml-dynamic-compression")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    parser.add_argument(
        "--wandb_no_table", action="store_true", help="Disable per-sample table logging"
    )

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_dataset}...")
    with open(args.test_dataset) as f:
        test_data = json.load(f)
    print(f"  Loaded {len(test_data)} samples")
    print(f"  Thresholds: {args.thresholds}")

    # initialize WandB
    run = init_wandb_eval(
        enabled=args.wandb,
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.wandb_name,
        tags=args.wandb_tags,
        config={
            "test_dataset": args.test_dataset,
            "num_samples": len(test_data),
            "thresholds": args.thresholds,
            "max_new_tokens": args.max_new_tokens,
            "baseline_model": args.baseline_model,
            "dynamic_model": args.dynamic_model,
            "classifier_path": args.classifier_path,
            "device": args.device,
            "script": "profiled_eval.py",
        },
    )

    # Initialize evaluator
    evaluator = ProfiledEvaluator(
        baseline_adapter_path=args.baseline_model,
        dynamic_adapter_path=args.dynamic_model,
        classifier_path=args.classifier_path,
        device=args.device,
    )

    # Run evaluation (loads/unloads models one at a time to save memory)
    # Checkpoints are saved after each sample for resume capability
    results = evaluator.run_full_evaluation(
        test_data=test_data,
        thresholds=args.thresholds,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )

    # Print results
    print_results_table(results)

    # Save final results (with aggregated metrics)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # log the eval results to json
    log_profiled_eval_results(
        run,
        results=results,
        output_json_path=args.output,
        log_per_sample_table=(not args.wandb_no_table),
        artifact_name=f"profiled_eval_{Path(args.test_dataset).stem}",
    )

    if run is not None:
        run.finish()

    num_methods = 2 + len(
        args.thresholds
    )  # compression_none, compression_newline, + classifiers
    print(f"\nFinal results saved to: {args.output}")
    print(f"  - Config and classifier profile")
    print(f"  - Aggregated metrics for {num_methods} methods")
    print(f"  - {len(test_data)} per-sample results per method")


if __name__ == "__main__":
    main()
