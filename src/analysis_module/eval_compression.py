"""
Online evaluation script for GSM8K compression.
Compares static baseline vs. dynamic classifier-based compression during generation.
Follows CCM inference pattern with streaming COMP token insertion.

Supports two compression modes:
- Attention masking: Full KV cache with masked attention (no memory savings)
- Recursive KV compression: Actual KV extraction at COMP positions (real memory savings)
"""

import sys
import os
from pathlib import Path

from src.model_module.peft_loader import load_peft_model

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_module.ccm_llama import LlamaForCausalLM_CCM
from model_module.compression_classifier import CompressionClassifier
from analysis_module.gsm8k_utils import extract_gsm8k_answer, verify_gsm8k_answer
import json
import time
from peft import PeftModel
from tqdm import tqdm

# Import extract_comp_results from CCM for actual KV compression
CCM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Context-Memory"))
extract_comp_results = None

def _import_ccm_extract():
    """Import extract_comp_results from CCM."""
    global extract_comp_results

    # Save main project's src modules
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == 'src' or key.startswith('src.'):
            saved_modules[key] = sys.modules.pop(key)

    sys.path.insert(0, CCM_PATH)

    try:
        from src.utils import extract_comp_results as _extract_comp_results
        extract_comp_results = _extract_comp_results
        print("Successfully imported extract_comp_results from CCM")
        return True
    except ImportError as e:
        print(f"Warning: Could not import extract_comp_results: {e}")
        return False
    finally:
        # We keep CCM's src in sys.modules since we need it for extraction
        pass

# Try to import CCM's extract function
_ccm_import_success = _import_ccm_extract()


def _extract_sink_and_comp(past_kv, sink_positions, comp_positions, kv_len, device):
    """Extract sink token(s) and COMP token(s) from KV cache.

    This preserves the BOS (sink) token which is critical for generation quality.
    Based on StreamingLLM's attention sink finding.

    Args:
        past_kv: KV cache tuple of (key, value) for each layer
        sink_positions: List of positions to keep as sink (usually [0] for BOS)
        comp_positions: List of COMP token positions in the current KV cache to keep
        kv_len: Total length of current KV cache
        device: torch device

    Returns:
        Extracted KV cache with only sink + COMP tokens
    """
    if extract_comp_results is None:
        raise ImportError("extract_comp_results not available from CCM")

    # Build mask: True for positions to KEEP
    keep_mask = [False] * kv_len
    for pos in sink_positions:
        if 0 <= pos < kv_len:
            keep_mask[pos] = True
    for pos in comp_positions:
        if 0 <= pos < kv_len:
            keep_mask[pos] = True

    keep_tensor = torch.tensor(keep_mask, device=device)
    return extract_comp_results(past_kv, keep_tensor)


def generate_no_compression(model, tokenizer, input_ids, max_new_tokens=256, device='cuda'):
    """
    Generate tokens without any compression (vanilla LLM baseline).
    No COMP tokens are inserted.

    Args:
        model: The model
        tokenizer: Tokenizer
        input_ids: Input token IDs
        max_new_tokens: Maximum tokens to generate
        device: Device to use

    Returns:
        Tuple of (generated_ids, 0, actual_kv_size_bytes)
    """
    generated_ids = input_ids.clone().to(device)
    past_key_values = None

    for step in range(max_new_tokens):
        if past_key_values is None:
            curr_input_ids = generated_ids
        else:
            curr_input_ids = generated_ids[:, -1:]

        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
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

    # Calculate actual KV cache size
    actual_kv_size = 0
    if past_key_values is not None:
        kv_len = past_key_values[0][0].shape[2]
        n_layers = len(past_key_values)
        n_kv_heads = past_key_values[0][0].shape[1]
        head_dim = past_key_values[0][0].shape[3]
        dtype_size = past_key_values[0][0].element_size()
        actual_kv_size = 2 * n_layers * kv_len * n_kv_heads * head_dim * dtype_size

    return generated_ids, 0, actual_kv_size


def generate_with_compression(model, tokenizer, input_ids, classifier, comp_token_id,
                             newline_token_id_list, use_classifier=True, threshold=0.5,
                             max_new_tokens=256, device='cuda', compress_kv=False):
    """
    Generate tokens online, inserting COMP tokens based on strategy.
    Follows CCM inference pattern with proper pos_id_offset tracking.

    Args:
        model: The CCM model
        tokenizer: Tokenizer
        input_ids: Input token IDs
        classifier: CompressionClassifier (or None for baseline)
        comp_token_id: Token ID for COMP token
        newline_token_id_list: List of newline token IDs for baseline
        use_classifier: If True, use classifier to decide COMP placement
                       If False, use baseline (newline-based)
        threshold: Classifier threshold for COMP insertion
        max_new_tokens: Maximum tokens to generate
        device: Device to use
        compress_kv: If True, actually extract KV cache at COMP positions (real memory savings)
                    If False, keep full KV cache (no memory savings, just attention masking)

    Returns:
        Tuple of (generated_ids, comp_count, actual_kv_size_bytes)
    """
    # Ensure input is on correct device
    generated_ids = input_ids.clone().to(device)
    past_key_values = None
    pos_id_offset = 0
    comp_count = 0
    comp_positions_in_kv = []  # Track COMP positions in the KV cache for extraction
    sink_positions = [0]  # BOS token position (attention sink)
    comp_token_tensor = torch.tensor([[comp_token_id]], device=device)
    actual_kv_size = 0

    for step in range(max_new_tokens):
        # Determine input for this step
        if past_key_values is None:
            curr_input_ids = generated_ids
        else:
            curr_input_ids = generated_ids[:, -1:]

        # Forward pass with CCM model (supports pos_id_offset)
        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=use_classifier,
                return_dict=True,
                pos_id_offset=pos_id_offset,
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # Append generated token
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        pos_id_offset += curr_input_ids.shape[-1]  # Increment by current input length only

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # ===== DECIDE IF WE SHOULD INSERT COMP =====
        should_insert_comp = False

        if use_classifier and classifier is not None:
            # OURS: Use classifier to predict compression
            if outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1][:, -1:, :]  # Last layer, last token, shape: (batch_size, 1, hidden_size)
                comp_prob = classifier.predict(hidden_states)[0, 0].item()  # Get probability for first batch, first (and only) token
                should_insert_comp = comp_prob >= threshold
        else:
            # Baseline: Insert after newline token
            # Check if the next token is also newline; if yes, insert COMP after the second newline
            if next_token.item() in newline_token_id_list:
                should_insert_comp = True
                # # Peek at the next token logits
                # with torch.no_grad():
                #     peek_outputs = model(
                #         input_ids=generated_ids[:, -1:],
                #         past_key_values=past_key_values,
                #         use_cache=True,
                #         return_dict=True
                #     )
                # peek_logits = peek_outputs.logits[:, -1, :]
                # peek_next_token = torch.argmax(peek_logits, dim=-1, keepdim=True)
                # if peek_next_token.item() == newline_token_id:
                #     generated_ids = torch.cat([generated_ids, peek_next_token], dim=1)
                #     should_insert_comp = True

        # Insert COMP token if needed
        if should_insert_comp:
            comp_count += 1

            # Add COMP token to generated sequence (so it appears in output text)
            generated_ids = torch.cat([generated_ids, comp_token_tensor], dim=1)

            # Process COMP token
            with torch.no_grad():
                comp_outputs = model(
                    input_ids=comp_token_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    pos_id_offset=pos_id_offset,
                )
            past_key_values = comp_outputs.past_key_values

            # If compress_kv is enabled, extract sink + COMP tokens from KV cache
            if compress_kv and extract_comp_results is not None:
                kv_len = past_key_values[0][0].shape[2]

                # The COMP token we just processed is at the last position
                current_comp_pos = kv_len - 1

                # Extract: keep sink (BOS) + all previous COMP tokens + current COMP
                # After extraction, positions shift, so we track relative to extraction
                all_comp_positions = comp_positions_in_kv + [current_comp_pos]

                past_key_values = _extract_sink_and_comp(
                    past_key_values,
                    sink_positions,
                    all_comp_positions,
                    kv_len,
                    device
                )

                # After extraction, update tracked positions:
                # - Sink stays at position 0
                # - Previous COMPs are at positions 1, 2, ..., len(comp_positions_in_kv)
                # - New COMP is at position len(comp_positions_in_kv) + 1
                comp_positions_in_kv = list(range(1, comp_count + 1))
            else:
                # No extraction, COMP position is at end of KV cache
                kv_len = past_key_values[0][0].shape[2]
                comp_positions_in_kv.append(kv_len - 1)

    # Calculate actual KV cache size
    if past_key_values is not None:
        kv_len = past_key_values[0][0].shape[2]
        n_layers = len(past_key_values)
        n_kv_heads = past_key_values[0][0].shape[1]
        head_dim = past_key_values[0][0].shape[3]
        dtype_size = past_key_values[0][0].element_size()
        actual_kv_size = 2 * n_layers * kv_len * n_kv_heads * head_dim * dtype_size

    return generated_ids, comp_count, actual_kv_size


def estimate_kv_cache(generated_ids, comp_token_id, model):
    """
    Estimate peak KV cache size (MB) based on the longest contiguous
    sequence of non-compression tokens.

    Args:
        generated_ids: torch.Tensor of shape (batch_size, seq_len)
        comp_token_id: int, ID of the <COMP> token
        model: HuggingFace model (with config.num_hidden_layers and config.hidden_size)

    Returns:
        float: Estimated peak KV cache in MB
    """
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    max_seq = 0
    curr_seq = 0

    for token in generated_ids[0].tolist():  # assume batch_size = 1
        if token == comp_token_id:
            max_seq = max(max_seq, curr_seq)
            curr_seq = 0
        else:
            curr_seq += 1

    max_seq = max(max_seq, curr_seq)
    kv_bytes = 2 * num_layers * max_seq * hidden_size * 2
    return kv_bytes / (1024 * 1024) 


def evaluate(test_dataset, model, tokenizer, classifier, comp_token_id, newline_token_id_list, do_baseline,
             classifier_threshold=0.5, max_new_tokens=256, device='cuda', compress_kv=False,
             no_compression=False):
    """
    Online evaluation comparing static baseline vs dynamic classifier.

    Args:
        test_dataset: List of test samples
        model: CCM model
        tokenizer: Tokenizer
        classifier: CompressionClassifier (or None for baseline)
        comp_token_id: COMP token ID
        newline_token_id_list: List of newline token IDs
        do_baseline: If True, run baseline (newline-based); if False, run dynamic (classifier)
        classifier_threshold: Threshold for classifier
        max_new_tokens: Max tokens to generate
        device: Device
        compress_kv: If True, use actual KV extraction (real compression)
        no_compression: If True, run vanilla generation without any COMP tokens

    Returns:
        Tuple of (aggregate_metrics, per_sample_results)
    """

    per_sample_results = []
    for i, sample in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        # Extract question and ground truth answer using GSM8K utilities
        gt_answer = extract_gsm8k_answer(sample)
        question = sample.get('question', '')

        # Tokenize input
        input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
        input_tokens = input_ids.shape[1]

        if no_compression:
            # ===== NO COMPRESSION: Vanilla LLM without any COMP tokens =====
            t0 = time.time()
            gen_ids, comp_count, kv_bytes = generate_no_compression(
                model, tokenizer, input_ids,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            latency = time.time() - t0
        elif do_baseline:
            # ===== STATIC BASELINE: Insert COMP after newline =====
            t0 = time.time()
            gen_ids, comp_count, kv_bytes = generate_with_compression(
                model, tokenizer, input_ids, classifier=None,
                comp_token_id=comp_token_id,
                newline_token_id_list=newline_token_id_list,
                use_classifier=False,
                max_new_tokens=max_new_tokens,
                device=device,
                compress_kv=compress_kv,
            )
            latency = time.time() - t0
        else:
            # ===== DYNAMIC: Use classifier to decide COMP insertion =====
            t0 = time.time()
            gen_ids, comp_count, kv_bytes = generate_with_compression(
                model, tokenizer, input_ids, classifier=classifier,
                comp_token_id=comp_token_id,
                newline_token_id_list=newline_token_id_list,
                use_classifier=True,
                threshold=classifier_threshold,
                max_new_tokens=max_new_tokens,
                device=device,
                compress_kv=compress_kv,
            )
            latency = time.time() - t0

        # Calculate metrics
        if kv_bytes > 0:
            kv_cache_mb = kv_bytes / (1024 * 1024)
        else:
            kv_cache_mb = estimate_kv_cache(gen_ids, comp_token_id, model)

        generated_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
        is_correct = verify_gsm8k_answer(gt_answer, generated_text)

        # Calculate tokens generated and throughput
        tokens_generated = gen_ids.shape[1] - input_tokens
        throughput = tokens_generated / latency if latency > 0 else 0

        # Extract predicted answer from generated text
        predicted_answer = None
        if '####' in generated_text:
            try:
                predicted_answer = generated_text.split('####')[-1].strip().split()[0]
            except:
                pass

        # Save comprehensive per-sample metadata
        per_sample_results.append({
            # Identification
            'sample_id': i,
            # Input
            'question': question,
            'input_tokens': input_tokens,
            # Ground truth
            'gt_answer': gt_answer,
            # Generation output
            'generated_text': generated_text,
            'predicted_answer': predicted_answer,
            'tokens_generated': tokens_generated,
            # Accuracy
            'is_correct': is_correct,
            # Compression metrics
            'comp_tokens_inserted': comp_count,
            'kv_cache_mb': kv_cache_mb,
            # Performance metrics
            'latency_seconds': latency,
            'throughput_tokens_per_sec': throughput,
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(test_dataset)}")

    # Compute aggregate metrics
    aggregate_metrics = {
        'total_examples': len(per_sample_results),
        'accuracy': np.mean([r['is_correct'] for r in per_sample_results]),
        'avg_comp_tokens': np.mean([r['comp_tokens_inserted'] for r in per_sample_results]),
        'std_comp_tokens': np.std([r['comp_tokens_inserted'] for r in per_sample_results]),
        'avg_kv_cache_mb': np.mean([r['kv_cache_mb'] for r in per_sample_results]),
        'std_kv_cache_mb': np.std([r['kv_cache_mb'] for r in per_sample_results]),
        'avg_latency': np.mean([r['latency_seconds'] for r in per_sample_results]),
        'std_latency': np.std([r['latency_seconds'] for r in per_sample_results]),
        'avg_throughput': np.mean([r['throughput_tokens_per_sec'] for r in per_sample_results]),
        'avg_tokens_generated': np.mean([r['tokens_generated'] for r in per_sample_results]),
        'total_time_seconds': sum([r['latency_seconds'] for r in per_sample_results]),
    }

    return aggregate_metrics, per_sample_results


def compare_models(baseline_output_fn, dynamic_output_fn, output_fn="outputs/comparison_results.json"):
    """Compare baseline and dynamic model results."""
    print("\n" + "=" * 70)
    print("COMPRESSION COMPARISON: BASELINE vs DYNAMIC")
    print("=" * 70)

    with open(baseline_output_fn) as f:
        baseline_data = json.load(f)
    with open(dynamic_output_fn) as f:
        dynamic_data = json.load(f)

    # Handle both old and new format
    if 'aggregate_metrics' in baseline_data:
        baseline_metrics = baseline_data['aggregate_metrics']
        baseline_config = baseline_data.get('config', {})
    else:
        baseline_metrics = baseline_data.get('results', baseline_data)
        baseline_config = {}

    if 'aggregate_metrics' in dynamic_data:
        dynamic_metrics = dynamic_data['aggregate_metrics']
        dynamic_config = dynamic_data.get('config', {})
        threshold = dynamic_config.get('threshold', 0.5)
    else:
        dynamic_metrics = dynamic_data.get('results', dynamic_data)
        threshold = dynamic_data.get('threshold', 0.5)

    # Print comparison table
    print(f"\n{'Metric':<30} {'Baseline':>15} {'Dynamic':>15} {'Diff':>15}")
    print("-" * 75)

    # Accuracy
    b_acc = baseline_metrics.get('accuracy', 0)
    d_acc = dynamic_metrics.get('accuracy', 0)
    print(f"{'Accuracy':<30} {b_acc:>14.1%} {d_acc:>14.1%} {(d_acc-b_acc):>+14.1%}")

    # COMP tokens
    b_comp = baseline_metrics.get('avg_comp_tokens', 0)
    d_comp = dynamic_metrics.get('avg_comp_tokens', 0)
    comp_diff = ((d_comp - b_comp) / b_comp * 100) if b_comp > 0 else 0
    print(f"{'Avg COMP Tokens':<30} {b_comp:>15.2f} {d_comp:>15.2f} {comp_diff:>+14.1f}%")

    # KV Cache
    b_kv = baseline_metrics.get('avg_kv_cache_mb', 0)
    d_kv = dynamic_metrics.get('avg_kv_cache_mb', 0)
    kv_diff = ((d_kv - b_kv) / b_kv * 100) if b_kv > 0 else 0
    print(f"{'Avg KV Cache (MB)':<30} {b_kv:>15.2f} {d_kv:>15.2f} {kv_diff:>+14.1f}%")

    # Latency
    b_lat = baseline_metrics.get('avg_latency', 0)
    d_lat = dynamic_metrics.get('avg_latency', 0)
    lat_diff = ((d_lat - b_lat) / b_lat * 100) if b_lat > 0 else 0
    print(f"{'Avg Latency (s)':<30} {b_lat:>15.3f} {d_lat:>15.3f} {lat_diff:>+14.1f}%")

    # Throughput
    b_tput = baseline_metrics.get('avg_throughput', 0)
    d_tput = dynamic_metrics.get('avg_throughput', 0)
    tput_diff = ((d_tput - b_tput) / b_tput * 100) if b_tput > 0 else 0
    print(f"{'Avg Throughput (tok/s)':<30} {b_tput:>15.1f} {d_tput:>15.1f} {tput_diff:>+14.1f}%")

    print("-" * 75)
    print(f"\nDynamic classifier threshold: {threshold}")

    # Save comparison
    comparison_data = {
        "baseline": {
            "config": baseline_config,
            "metrics": baseline_metrics,
        },
        "dynamic": {
            "config": dynamic_config if 'aggregate_metrics' in dynamic_data else {},
            "metrics": dynamic_metrics,
        },
        "comparison": {
            "accuracy_diff": d_acc - b_acc,
            "comp_tokens_diff_pct": comp_diff,
            "kv_cache_diff_pct": kv_diff,
            "latency_diff_pct": lat_diff,
            "throughput_diff_pct": tput_diff,
        }
    }

    with open(output_fn, "w") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nComparison saved to: {output_fn}")


def get_comp_and_newline_tokens(tokenizer) -> tuple:
    """
    Extract COMP token ID and newline token IDs from tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer

    Returns:
        tuple: (comp_token_id, newline_token_id_list)
    """
    # Get COMP0 token
    if '<COMP0>' not in tokenizer.get_vocab():
        raise ValueError('Tokenizer vocabulary does not contain <COMP0>.')
    comp_token_id = tokenizer.convert_tokens_to_ids('<COMP0>')

    # Get newline tokens
    newline_token_id_list = [
        tokenizer.encode('\n', add_special_tokens=False)[0],
        tokenizer.encode('\n\n', add_special_tokens=False)[0]
    ]

    return comp_token_id, newline_token_id_list


def main(args):
    # Load data
    with open(args.test_dataset) as f:
        test_data = json.load(f)

    dtype = torch.float16
    # Load model and tokenizer (PEFT adapter)
    OURS_model_path = Path(args.OURS_model).resolve()
    assert OURS_model_path.exists(), f"Model path does not exist: {OURS_model_path}"
    baseline_model_path = Path(args.baseline_model).resolve()

    print(f"Loading PEFT adapter from {OURS_model_path}...")
    
    # Load tokenizer and get special token IDs
    tokenizer = AutoTokenizer.from_pretrained(str(OURS_model_path), trust_remote_code=True, local_files_only=True,
                                              fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token
    comp_token_id, newline_token_id_list = get_comp_and_newline_tokens(tokenizer)
    
    print(f"COMP token ID: {comp_token_id}")
    print(f"newline_token_id_list: {newline_token_id_list}")

    # Load base model (Llama 3.1 8B Instruct)
    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # dtype = torch.float16 if args.device == 'cuda' else torch.float32

    compress_kv = not args.no_compress_kv

    if args.method == 'compression_none':
        print(f"\nLoading {base_model_id} (no CCM fine-tuning) for NO COMPRESSION evaluation...")
        from transformers import AutoModelForCausalLM
        vanilla_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=dtype,
            device_map=args.device,
        )

        print(f"\nRunning NO COMPRESSION evaluation (vanilla LLM, no COMP tokens)...")
        aggregate_metrics, per_sample_results = evaluate(
            test_data, vanilla_model, tokenizer, classifier=None,
            do_baseline=False,
            comp_token_id=comp_token_id,
            newline_token_id_list=newline_token_id_list,
            classifier_threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            compress_kv=False,
            no_compression=True,
        )
        method_name = "compression_none"
    elif args.method == 'compression_newline':
        print(f"\nLoading baseline model with PEFT adapter...")
        baseline_model, _ = load_peft_model(
            base_model_id=base_model_id,
            adapter_path=str(baseline_model_path),
            device=args.device,
            dtype=dtype
        )

        compression_mode = "recursive KV compression" if compress_kv else "attention masking only"
        print(f"\nRunning STATIC baseline evaluation ({compression_mode})...")
        aggregate_metrics, per_sample_results = evaluate(
            test_data, baseline_model, tokenizer, classifier=None,
            do_baseline=True,
            comp_token_id=comp_token_id,
            newline_token_id_list=newline_token_id_list,
            classifier_threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            compress_kv=compress_kv,
        )
        method_name = "compression_newline"
    elif args.method == 'compression_classifier':
        print(f"\nLoading OURS model with PEFT adapter...")
        model, _ = load_peft_model(
            base_model_id=base_model_id,
            adapter_path=str(OURS_model_path),
            device=args.device,
            # dtype=dtype
        )

        # Load classifier
        print(f"Loading classifier from {args.classifier_path}...")
        classifier = CompressionClassifier(hidden_size=model.config.hidden_size, dropout=0.1)
        classifier.load_state_dict(torch.load(args.classifier_path))
        classifier.eval()
        classifier = classifier.to(args.device).to(dtype)

        compression_mode = "recursive KV compression" if compress_kv else "attention masking only"
        print(f"\nRunning DYNAMIC classifier evaluation ({compression_mode})...")
        aggregate_metrics, per_sample_results = evaluate(
            test_data, model, tokenizer, classifier=classifier,
            do_baseline=False,
            comp_token_id=comp_token_id,
            newline_token_id_list=newline_token_id_list,
            classifier_threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            compress_kv=compress_kv,
        )
        method_name = "compression_classifier"

    def print_results(title, r):
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Accuracy:            {r['accuracy']:.2%}")
        print(f"Avg COMP tokens:     {r['avg_comp_tokens']:.2f} (±{r['std_comp_tokens']:.2f})")
        print(f"Avg KV Cache (MB):   {r['avg_kv_cache_mb']:.2f} (±{r['std_kv_cache_mb']:.2f})")
        print(f"Avg Latency (s):     {r['avg_latency']:.3f} (±{r['std_latency']:.3f})")
        print(f"Avg Throughput:      {r['avg_throughput']:.1f} tokens/sec")
        print(f"Avg Tokens Gen:      {r['avg_tokens_generated']:.1f}")
        print(f"Total Time:          {r['total_time_seconds']:.1f}s")
        print(f"Examples:            {r['total_examples']}")

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    compression_mode = "recursive KV compression" if compress_kv else "attention masking only"
    if args.method == 'compression_none':
        title = "No Compression (vanilla LLM, no COMP tokens)"
    elif args.method == 'compression_newline':
        title = f"Static Baseline (newline-based) [{compression_mode}]"
    else:
        title = f"Dynamic Classifier (threshold={args.threshold}) [{compression_mode}]"

    print_results(title, aggregate_metrics)

    # Save comprehensive results
    suffix = "_compressed" if compress_kv else ""
    if args.method == 'compression_none':
        output_fn = "outputs/compression_none_eval_results.json"
    elif args.method == 'compression_newline':
        output_fn = f"outputs/compression_newline_eval_results{suffix}.json"
    else:
        output_fn = f"outputs/compression_classifier_eval_results{suffix}.json"

    # Build comprehensive output
    output_data = {
        # Configuration
        "config": {
            "method": args.method,
            "threshold": args.threshold if args.method == "compression_classifier" else None,
            "compress_kv": compress_kv,
            "max_new_tokens": args.max_new_tokens,
            "test_dataset": args.test_dataset,
            "model_path": base_model_id if args.method == 'compression_none' else (str(baseline_model_path) if args.method == 'compression_newline' else str(OURS_model_path)),
            "classifier_path": args.classifier_path if args.method == "compression_classifier" else None,
        },
        # Aggregate metrics
        "aggregate_metrics": aggregate_metrics,
        # Per-sample detailed results
        "per_sample_results": per_sample_results,
    }

    with open(output_fn, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_fn}")
    print(f"  - Config and aggregate metrics")
    print(f"  - {len(per_sample_results)} per-sample results with full metadata")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, required=False, default='data/gsm8k-test-200.json',
                        help='Path to test JSON file')
    parser.add_argument('--classifier_path', type=str, default='outputs/classifier/compression_classifier.pt',
                        help='Path to classifier checkpoint')
    parser.add_argument('--baseline_model', type=str,
                        default='outputs/baseline_insert_COMP_after_newline-llama-3.1-8b-instruct-online-concat_recur',
                        help='Path to model (local) or HF model ID')
    parser.add_argument('--OURS_model', type=str, default='outputs/OURS_llama-3.1-8b-instruct-online-concat_recur',
                        help='Path to model (local) or HF model ID')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classifier threshold for COMP insertion')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--method', type=str, required=True,
                        choices=['compression_none', 'compression_newline', 'compression_classifier'],
                        help='Compression method: '
                             'compression_none = vanilla LLM (no COMP tokens), '
                             'compression_newline = insert COMP after newlines, '
                             'compression_classifier = learned COMP placement')
    parser.add_argument('--no_compress_kv', action='store_true', default=False,
                        help='Disable KV extraction, use attention masking only (no real memory savings). '
                             'By default, KV extraction is enabled.')

    main(parser.parse_args())
    # compare_models("outputs/baseline_eval_results.json", "outputs/dynamic_eval_results.json")