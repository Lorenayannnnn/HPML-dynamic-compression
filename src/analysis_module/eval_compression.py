"""
Online evaluation script for GSM8K compression.
Compares static baseline vs. dynamic classifier-based compression during generation.
Follows CCM inference pattern with streaming COMP token insertion.
"""

import sys
from pathlib import Path
from sqlite3 import adapt

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
    # pos_id_offset tracks the cumulative offset from COMP token insertions
    # This is needed for CCM models to adjust position embeddings correctly
    pos_id_offset = torch.zeros((input_ids.shape[0], 1), dtype=torch.long, device=device)
    comp_count = 0
    comp_positions = []

    for step in range(max_new_tokens):
        # Determine input for this step
        if past_key_values is None:
            curr_input_ids = generated_ids
            curr_pos_id_offset = None  # First pass, no offset needed
        else:
            curr_input_ids = generated_ids[:, -1:]
            curr_pos_id_offset = pos_id_offset

        # Forward pass with CCM model (supports pos_id_offset)
        with torch.no_grad():
            outputs = model(
                input_ids=curr_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=use_classifier,
                return_dict=True,
                pos_id_offset=curr_pos_id_offset,
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        # Append generated token
        generated_ids = torch.cat([generated_ids, next_token], dim=1)

        # Check for EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

        # ===== DECIDE IF WE SHOULD INSERT COMP =====
        should_insert_comp = False

        if use_classifier and classifier is not None:
            # OURS: Use classifier to predict compression
            if outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1][:, -1, :]  # Last layer, last token
                comp_prob = classifier.predict(hidden_states).item()
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
            comp_token = torch.tensor([[comp_token_id]], device=device)
            generated_ids = torch.cat([generated_ids, comp_token], dim=1)
            comp_positions.append(generated_ids.shape[1] - 1)
            comp_count += 1
            # Increment pos_id_offset by 1 for each COMP token inserted
            pos_id_offset += 1

            # Update past_key_values for COMP token (simplified: just process it)
            with torch.no_grad():
                comp_outputs = model(
                    input_ids=comp_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    pos_id_offset=pos_id_offset,
                )
            past_key_values = comp_outputs.past_key_values

    return generated_ids, comp_count, comp_positions


def estimate_kv_cache(seq_len, model):
    """Estimate KV cache size in MB."""
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    # KV cache: 2 * num_layers * seq_len * hidden_size * 2 bytes (float16)
    kv_bytes = 2 * num_layers * seq_len * hidden_size * 2
    return kv_bytes / (1024 * 1024)  # Convert to MB


def evaluate(test_dataset, model, tokenizer, classifier, comp_token_id, newline_token_id_list, do_baseline,
             classifier_threshold=0.5, max_new_tokens=256, device='cuda'):
    """
    Online evaluation comparing static baseline vs dynamic classifier.
    """

    return_results = []

    for i, sample in tqdm(enumerate(test_dataset)):
        # Extract question and ground truth answer using GSM8K utilities
        gt_answer = extract_gsm8k_answer(sample)
        question = sample.get('question', '')

        # Tokenize input
        input_ids = tokenizer.encode(question, return_tensors='pt').to(device)

        if do_baseline:
            # ===== STATIC BASELINE: Insert COMP after newline =====
            t0 = time.time()
            static_gen_ids, static_comp_count, _ = generate_with_compression(
                model, tokenizer, input_ids, classifier=None,
                comp_token_id=comp_token_id,
                newline_token_id_list=newline_token_id_list,
                use_classifier=False,
                max_new_tokens=max_new_tokens,
                device=device
            )
            static_latency = time.time() - t0
            static_kv_cache = estimate_kv_cache(static_gen_ids.shape[1], model)

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
            dynamic_gen_ids, dynamic_comp_count, _ = generate_with_compression(
                model, tokenizer, input_ids, classifier=classifier,
                comp_token_id=comp_token_id,
                newline_token_id_list=newline_token_id_list,
                use_classifier=True,
                threshold=classifier_threshold,
                max_new_tokens=max_new_tokens,
                device=device
            )
            dynamic_latency = time.time() - t0
            dynamic_kv_cache = estimate_kv_cache(dynamic_gen_ids.shape[1], model)

            dynamic_text = tokenizer.decode(dynamic_gen_ids[0], skip_special_tokens=False)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dataset', type=str, required=False, default='data/gsm8k-test-200.json',help='Path to test JSON file')
    parser.add_argument('--classifier_path', type=str, default='outputs/classifier/compression_classifier.pt', help='Path to classifier checkpoint')
    parser.add_argument('--baseline_model', type=str, default='outputs/baseline_insert_COMP_after_newline-llama-3.1-8b-instruct-online-concat_recur', help='Path to model (local) or HF model ID')
    parser.add_argument('--OURS_model', type=str, default='outputs/OURS_llama-3.1-8b-instruct-online-concat_recur', help='Path to model (local) or HF model ID')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classifier threshold for COMP insertion')
    parser.add_argument('--max_new_tokens', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load data
    with open(args.test_dataset) as f:
        test_data = json.load(f)

    # Load model and tokenizer (PEFT adapter)
    OURS_model_path = Path(args.OURS_model).resolve()
    assert OURS_model_path.exists(), f"Model path does not exist: {OURS_model_path}"
    baseline_model_path = Path(args.baseline_model).resolve()

    print(f"Loading PEFT adapter from {OURS_model_path}...")
    # Load tokenizer from adapter directory
    tokenizer = AutoTokenizer.from_pretrained(str(OURS_model_path), trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model (Llama 3.1 8B Instruct)
    base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading base model {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map='auto',
        trust_remote_code=True
    )

    # Apply PEFT adapter
    print(f"Applying PEFT adapter to load OURS...")
    model = PeftModel.from_pretrained(base_model, str(OURS_model_path))
    model = model.merge_and_unload()  # Merge adapter weights into base model
    model.eval()

    baseline_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map='auto',
        trust_remote_code=True
    )
    # Apply PEFT adapter
    print(f"Applying PEFT adapter to load baseline...")
    baseline_base_model = PeftModel.from_pretrained(baseline_base_model, str(baseline_model_path))
    baseline_base_model = baseline_base_model.merge_and_unload()  # Merge adapter weights into base model
    baseline_base_model.eval()

    # Add COMP0 token if not present
    if '<COMP0>' not in tokenizer.get_vocab():
        raise ValueError('Tokenizer vocabulary does not contain <COMP0>.')
        # tokenizer.add_special_tokens({'additional_special_tokens': ['<COMP0>']})
        # model.resize_token_embeddings(len(tokenizer))
    comp_token_id = tokenizer.convert_tokens_to_ids('<COMP0>')

    # Get newline token ID
    newline_token_id_list = [
        tokenizer.encode('\n', add_special_tokens=False)[0],
        tokenizer.encode('\n\n', add_special_tokens=False)[0]
    ]
    print(f"COMP token ID: {comp_token_id}")
    print(f"newline_token_id_list: {newline_token_id_list}")

    # Load classifier
    print(f"Loading classifier from {args.classifier_path}...")
    classifier = CompressionClassifier(hidden_size=model.config.hidden_size, dropout=0.1)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=args.device))
    classifier.eval()
    classifier = classifier.to(args.device)

    # Run evaluation
    # print(f"Evaluating on {len(test_data)} examples...")
    # results = evaluate(
    #     test_data, model, tokenizer, classifier,
    #     do_baseline=args.do_baseline,
    #     comp_token_id=comp_token_id,
    #     newline_token_id=newline_token_id,
    #     classifier_threshold=args.threshold,
    #     max_new_tokens=args.max_new_tokens,
    #     device=args.device,
    # )

    print("\nRunning STATIC baseline evaluation...")
    baseline_results = evaluate(
        test_data, baseline_base_model, tokenizer, classifier=None,
        do_baseline=True,
        comp_token_id=comp_token_id,
        newline_token_id_list=newline_token_id_list,
        classifier_threshold=args.threshold,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    print("\nRunning DYNAMIC classifier evaluation...")
    dynamic_results = evaluate(
        test_data, model, tokenizer, classifier=classifier,
        do_baseline=False,
        comp_token_id=comp_token_id,
        newline_token_id=newline_token_id,
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

    print_results("Static Baseline (newline-based)", baseline_results)
    print_results(f"Dynamic Classifier (threshold={args.threshold})", dynamic_results)

    print("\n" + "=" * 70)
    print("COMPRESSION GAINS")
    print("=" * 70)

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
                "threshold": args.threshold,
            },
            f,
            indent=2,
        )
