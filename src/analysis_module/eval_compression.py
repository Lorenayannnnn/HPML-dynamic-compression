"""
Online evaluation script for GSM8K compression.
Compares static baseline vs. dynamic classifier-based compression during generation.
Follows CCM inference pattern with streaming COMP token insertion.
"""

import sys
from pathlib import Path
from sqlite3 import adapt

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

def generate_with_compression(model, tokenizer, input_ids, classifier, comp_token_id,
                             newline_token_id_list, use_classifier=True, threshold=0.5,
                             max_new_tokens=256, device='cuda'):
    """
    Generate tokens online, inserting COMP tokens based on strategy.
    Follows CCM inference pattern with proper pos_id_offset tracking.

    Args:
        use_classifier: If True, use classifier to decide COMP placement
                       If False, use baseline (newline-based)
    """
    generated_ids = input_ids.clone()
    past_key_values = None
    # pos_id_offset = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=device)
    pos_id_offset = 0
    comp_count = 0
    # comp_positions = []
    comp_token_tensor = torch.tensor([[comp_token_id]], device='cuda')

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
            # generated_ids = torch.cat([generated_ids, comp_token], dim=1)
            # comp_positions.append(generated_ids.shape[1] - 1)
            comp_count += 1

            # Update past_key_values for COMP token (simplified: just process it)
            with torch.no_grad():
                comp_outputs = model(
                    input_ids=comp_token_tensor,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    pos_id_offset=pos_id_offset,
                )
            past_key_values = comp_outputs.past_key_values

    return generated_ids, comp_count


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
             classifier_threshold=0.5, max_new_tokens=256, device='cuda'):
    """
    Online evaluation comparing static baseline vs dynamic classifier.
    """

    return_results = []
    for i, sample in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        # Extract question and ground truth answer using GSM8K utilities
        gt_answer = extract_gsm8k_answer(sample)
        question = sample.get('question', '')

        # Tokenize input
        input_ids = tokenizer.encode(question, return_tensors='pt').to(device)

        if do_baseline:
            # ===== STATIC BASELINE: Insert COMP after newline =====
            t0 = time.time()
            static_gen_ids, static_comp_count = generate_with_compression(
                model, tokenizer, input_ids, classifier=None,
                comp_token_id=comp_token_id,
                newline_token_id_list=newline_token_id_list,
                use_classifier=False,
                max_new_tokens=max_new_tokens,
                device=device
            )
            static_latency = time.time() - t0
            static_kv_cache = estimate_kv_cache(static_gen_ids, comp_token_id, model)

            static_text = tokenizer.decode(static_gen_ids[0], skip_special_tokens=False)
            static_acc = verify_gsm8k_answer(gt_answer, static_text)

            return_results.append({
                'comp_count': static_comp_count,
                'kv_cache_mb': static_kv_cache,
                'latency': static_latency,
                'accuracy': static_acc
            })
        else:

            # ===== DYNAMIC: Use classifier to decide COMP insertion =====
            t0 = time.time()
            dynamic_gen_ids, dynamic_comp_count = generate_with_compression(
                model, tokenizer, input_ids, classifier=classifier,
                comp_token_id=comp_token_id,
                newline_token_id_list=newline_token_id_list,
                use_classifier=True,
                threshold=classifier_threshold,
                max_new_tokens=max_new_tokens,
                device=device
            )
            dynamic_latency = time.time() - t0
            dynamic_kv_cache = estimate_kv_cache(dynamic_gen_ids, comp_token_id, model)

            dynamic_text = tokenizer.decode(dynamic_gen_ids[0], skip_special_tokens=False)
            # print(f"Sample {i+1} Generated Text:\n{dynamic_text}\n")
            dynamic_acc = verify_gsm8k_answer(gt_answer, dynamic_text)

            return_results.append({
                'comp_count': dynamic_comp_count,
                'kv_cache_mb': dynamic_kv_cache,
                'latency': dynamic_latency,
                'accuracy': dynamic_acc
            })

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(test_dataset)}")

    return {
        'avg_comp_tokens': np.mean([r['comp_count'] for r in return_results]),
        'avg_kv_cache_mb': np.mean([r['kv_cache_mb'] for r in return_results]),
        'avg_latency': np.mean([r['latency'] for r in return_results]),
        'accuracy': np.mean([r['accuracy'] for r in return_results]),
        'total_examples': len(return_results)
    }

    # Compute aggregate metrics
    # return {
    #     'static': {
    #         'avg_comp_tokens': np.mean([r['comp_count'] for r in static_results]),
    #         'avg_kv_cache_mb': np.mean([r['kv_cache_mb'] for r in static_results]),
    #         'avg_latency': np.mean([r['latency'] for r in static_results]),
    #         'accuracy': np.mean([r['accuracy'] for r in static_results]),
    #         'total_examples': len(static_results)
    #     },
    #     'dynamic': {
    #         'avg_comp_tokens': np.mean([r['comp_count'] for r in dynamic_results]),
    #         'avg_kv_cache_mb': np.mean([r['kv_cache_mb'] for r in dynamic_results]),
    #         'avg_latency': np.mean([r['latency'] for r in dynamic_results]),
    #         'accuracy': np.mean([r['accuracy'] for r in dynamic_results]),
    #         'total_examples': len(dynamic_results)
    #     }
    # }


def compare_models(baseline_output_fn, dynamic_output_fn):
    print("\n" + "=" * 70)
    print("COMPRESSION GAINS")
    print("=" * 70)

    with open(baseline_output_fn) as f:
        baseline_results = json.load(f)
    with open(dynamic_output_fn) as f:
        dynamic_results = json.load(f)
        threshold = dynamic_results.get('threshold', 0.5)

    if baseline_results['avg_comp_tokens'] > 0:
        comp_reduction = (
                                 baseline_results['avg_comp_tokens']
                                 - dynamic_results['avg_comp_tokens']
                         ) / baseline_results['avg_comp_tokens']

        kv_reduction = (
                               baseline_results['avg_kv_cache_mb']
                               - dynamic_results['avg_kv_cache_mb']
                       ) / baseline_results['avg_kv_cache_mb']

        speedup = (
            baseline_results['avg_latency']
            / dynamic_results['avg_latency']
            if dynamic_results['avg_latency'] > 0 else 1.0
        )

        print(f"COMP token reduction: {comp_reduction:.2%}")
        print(f"KV cache reduction:   {kv_reduction:.2%}")
        print(f"Latency speedup:      {speedup:.2f}×")

    with open("eval_results.json", "w") as f:
        json.dump(
            {
                "baseline": baseline_results,
                "dynamic": dynamic_results,
                "threshold": threshold,
            },
            f,
            indent=2,
        )


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

    if args.do_baseline:
        print(f"\nLoading baseline model with PEFT adapter...")
        baseline_model, _ = load_peft_model(
            base_model_id=base_model_id,
            adapter_path=str(baseline_model_path),
            device=args.device,
            dtype=dtype
        )

        print("\nRunning STATIC baseline evaluation...")
        results = evaluate(
            test_data, baseline_model, tokenizer, classifier=None,
            do_baseline=True,
            comp_token_id=comp_token_id,
            newline_token_id_list=newline_token_id_list,
            classifier_threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    else:
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

        print("\nRunning DYNAMIC classifier evaluation...")
        results = evaluate(
            test_data, model, tokenizer, classifier=classifier,
            do_baseline=False,
            comp_token_id=comp_token_id,
            newline_token_id_list=newline_token_id_list,
            classifier_threshold=args.threshold,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )

    def print_results(title, r):
        print(f"\n{title}")
        print("-" * len(title))
        print(f"Accuracy:        {r['accuracy']:.2%}")
        print(f"Avg COMP tokens: {r['avg_comp_tokens']:.2f}")
        print(f"KV Cache (MB):   {r['avg_kv_cache_mb']:.2f}")
        print(f"Latency (s):     {r['avg_latency']:.3f}")
        print(f"Examples:        {r['total_examples']}")

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print_results(
        "Static Baseline (newline-based)" if args.do_baseline else f"Dynamic Classifier (threshold={args.threshold})",
        results)

    # Save result
    output_fn = "outputs/baseline_eval_results.json" if args.do_baseline else "outputs/dynamic_eval_results.json"
    with open(output_fn, "w") as f:
        json.dump(
            {
                "results": results,
                "threshold": args.threshold if not args.do_baseline else None,
            },
            f,
            indent=2,
        )

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
    parser.add_argument('--do_baseline', action='store_true', help='Whether to run baseline evaluation')

    main(parser.parse_args())
    # compare_models("outputs/baseline_eval_results.json", "outputs/dynamic_eval_results.json")