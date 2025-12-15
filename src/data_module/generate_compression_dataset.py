#!/usr/bin/env python3
"""
Generate GSM8K reasoning traces with <COMP> compression tokens.

Uses GPT-4o-mini to generate step-by-step reasoning with <COMP> tokens
placed at natural compression checkpoints (after calculations, key facts, etc.)

This script supports:
- Resuming from existing partial datasets
- Configurable number of samples
- Periodic checkpointing
- Analysis of generated data

Example usage:
    # Generate 500 samples (default)
    python generate_compression_dataset.py

    # Generate 1000 samples with custom output path
    python generate_compression_dataset.py --num_samples 1000 --output data/gsm8k_custom.json

    # Resume generation (automatically resumes from existing file)
    python generate_compression_dataset.py --output data/gsm8k_compressed_train.json
"""

import os
import json
import time
import argparse
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional

from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

from textwrap import dedent

# Prompt for generating compressed reasoning
COMPRESSION_PROMPT = """Solve this math problem step-by-step. After each logical step or calculation, insert a <COMP> token to mark a "checkpoint" in your reasoning.

Rules for <COMP> placement:
- Insert <COMP> after completing a calculation (e.g., "2 × 3 = 6 <COMP>")
- Insert <COMP> after establishing a key fact from the problem
- Insert <COMP> at the end of each reasoning phase
- Do NOT insert <COMP> in the middle of a calculation
- Aim for 3-6 <COMP> tokens per solution

Problem: {question}

Solve with <COMP> tokens at natural checkpoints, then give the final numerical answer after ####:"""

# OpenAI API parameters
DEFAULT_MODEL = "gpt-4o-mini"
API_PARAMS = {
    "temperature": 0.3,
    "max_tokens": 1024,
}


def load_existing_data(save_path: str) -> Tuple[List[Dict], Set[str]]:
    """Load existing dataset if file exists (for resume capability).

    Args:
        save_path: Path to the JSON file

    Returns:
        Tuple of (existing dataset, set of processed questions)
    """
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            dataset = json.load(f)
        existing_questions = {item['question'] for item in dataset}
        print(f"Loaded {len(dataset)} existing samples from {save_path}")
        return dataset, existing_questions
    return [], set()


def save_dataset(dataset: List[Dict], save_path: str) -> None:
    """Save dataset to JSON file.

    Args:
        dataset: List of data items
        save_path: Path to save to
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(dataset, f, indent=2)


def generate_compressed_reasoning(
    questions,
    num_samples: int = 500,
    save_path: str = 'gsm8k_compressed_train.json',
    model: str = DEFAULT_MODEL,
    checkpoint_interval: int = 50,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate reasoning traces with <COMP> tokens using OpenAI API.

    Automatically resumes from where it left off if save_path exists.

    Args:
        questions: Dataset of GSM8K problems (HuggingFace Dataset)
        num_samples: Number of samples to generate
        save_path: Path to save the dataset (appends to existing data)
        model: OpenAI model to use
        checkpoint_interval: Save checkpoint every N samples

    Returns:
        Tuple of (dataset, errors)
    """
    client = OpenAI()
    errors = []

    # Load existing data (resume capability)
    dataset, existing_questions = load_existing_data(save_path)

    # Check if we already have enough samples
    if len(dataset) >= num_samples:
        print(f"Already have {len(dataset)} samples (requested {num_samples}). Skipping generation.")
        return dataset, errors

    # Calculate how many more we need
    num_to_generate = num_samples - len(dataset)
    print(f"Need to generate {num_to_generate} more samples...")
    print(f"Using model: {model}")

    # Get questions to process (skip already processed ones)
    num_to_process = min(num_samples, len(questions))
    subset = questions.select(range(num_to_process))

    generated_count = 0
    samples_since_checkpoint = 0

    for i, item in enumerate(tqdm(subset, desc="Generating")):
        # Skip if already processed
        if item['question'] in existing_questions:
            continue

        # Stop if we have enough
        if len(dataset) >= num_samples:
            break

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": COMPRESSION_PROMPT.format(question=item['question'])
                }],
                **API_PARAMS
            )

            reasoning = response.choices[0].message.content
            original_answer = item['answer'].split("####")[-1].strip()

            dataset.append({
                "question": item['question'],
                "reasoning_with_compression": reasoning,
                "answer": original_answer,
                "original_reasoning": item['answer']
            })
            existing_questions.add(item['question'])
            generated_count += 1
            samples_since_checkpoint += 1

            # Periodic checkpoint
            if samples_since_checkpoint >= checkpoint_interval:
                save_dataset(dataset, save_path)
                print(f"\nCheckpoint: saved {len(dataset)} samples")
                samples_since_checkpoint = 0

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            errors.append({"index": i, "question": item['question'][:100], "error": str(e)})
            print(f"\nError at index {i}: {e}")
            # Save on error to preserve progress
            save_dataset(dataset, save_path)
            print(f"Saved {len(dataset)} samples after error")
            time.sleep(1)  # Back off on errors

    # Final save
    save_dataset(dataset, save_path)

    print(f"\nGeneration complete!")
    print(f"  Generated: {generated_count} new samples")
    print(f"  Total: {len(dataset)} samples")
    print(f"  Errors: {len(errors)}")

    return dataset, errors


def analyze_compression_data(data: List[Dict]) -> Dict:
    """Analyze <COMP> token distribution in generated data.

    Args:
        data: List of generated samples

    Returns:
        Analysis statistics
    """
    comp_counts = [d['reasoning_with_compression'].count('<COMP>') for d in data]

    print("\n" + "=" * 50)
    print("Compression Token Analysis")
    print("=" * 50)
    print(f"Total samples: {len(data)}")
    print(f"Average <COMP> per sample: {sum(comp_counts)/len(comp_counts):.1f}")
    print(f"Min <COMP>: {min(comp_counts)}")
    print(f"Max <COMP>: {max(comp_counts)}")

    # Distribution
    dist = Counter(comp_counts)
    print(f"\nDistribution:")
    for count in sorted(dist.keys()):
        print(f"  {count} <COMP>: {dist[count]} samples")

    # Show example
    print("\n" + "=" * 50)
    print("Example (first sample):")
    print("=" * 50)
    print(f"Question: {data[0]['question'][:100]}...")
    print(f"\nReasoning with <COMP>:")
    print(data[0]['reasoning_with_compression'][:500])
    if len(data[0]['reasoning_with_compression']) > 500:
        print("...")

    return {
        "total_samples": len(data),
        "avg_comp_tokens": sum(comp_counts) / len(comp_counts),
        "min_comp": min(comp_counts),
        "max_comp": max(comp_counts),
        "distribution": dict(dist)
    }

def insert_comp_tokens_at_newlines(json_fn: str = "data/gsm8k_compressed_train.json",
                                   output_path: str = "data/gsm8k_compressed_train_with_comp_at_newline.json") -> None:
    """
    Insert <COMP> tokens before non-consecutive newline char and after non <COMP> words in the reasoning traces.
    """
    new_lines = []
    with open(json_fn, "r") as f:
        data = json.load(f)
    print(f"Number of lines in {json_fn}: {len(data)}")
    for line in data:
        reasoning_with_compression = line["reasoning_with_compression"]
        reasoning_with_compression_split_by_newline = reasoning_with_compression.split("\n")
        tmp_reasoning = []
        for segment in reasoning_with_compression_split_by_newline:
            segment = segment.strip()
            if segment != "":
                if segment.endswith("<COMP>"):
                    tmp_reasoning.append(f"{segment}\n")
                else:
                    tmp_reasoning.append(f"{segment} <COMP>\n")
            else:
                tmp_reasoning.append("\n")
        line["reasoning_with_compression"] = "".join(tmp_reasoning).strip()
        new_lines.append(line)
    with open(output_path, "w") as f:
        json.dump(new_lines, f, indent=2)


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate GSM8K reasoning traces with <COMP> compression tokens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
        Examples:
        # Generate 500 samples (default)
        python generate_compression_dataset.py

        # Generate 1000 samples with custom output
        python generate_compression_dataset.py --num_samples 1000 --output data/custom.json

        # Use a different model
        python generate_compression_dataset.py --model gpt-4

        # Just analyze existing data
        python generate_compression_dataset.py --analyze_only --output data/existing.json
        """)
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/gsm8k_compressed_train.json",
        help="Output path for the generated dataset (default: data/gsm8k_compressed_train.json)"
    )
    parser.add_argument(
        "--num_samples", "-n",
        type=int,
        default=500,
        help="Number of samples to generate (default: 500)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N samples (default: 50)"
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze existing data, don't generate new samples"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Which GSM8K split to use as source questions (default: train)"
    )

    args = parser.parse_args()

    # Check for API key
    if not args.analyze_only and "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return 1

    # Analyze only mode
    if args.analyze_only:
        if not os.path.exists(args.output):
            print(f"Error: File not found: {args.output}")
            return 1
        with open(args.output, 'r') as f:
            data = json.load(f)
        analyze_compression_data(data)
        return 0

    # Load GSM8K dataset
    print(f"Loading GSM8K {args.split} split...")
    gsm8k = load_dataset("openai/gsm8k", "main", split=args.split)
    print(f"Loaded {len(gsm8k)} questions")

    # Generate compressed reasoning traces
    print(f"\nTarget: {args.num_samples} compressed reasoning traces")
    print(f"Output: {args.output}")
    print()

    data, errors = generate_compressed_reasoning(
        gsm8k,
        num_samples=args.num_samples,
        save_path=args.output,
        model=args.model,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Analyze the generated data
    if data:
        analyze_compression_data(data)

    # Save errors if any
    if errors:
        error_path = args.output.replace('.json', '_errors.json')
        with open(error_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f"\nErrors saved to {error_path}")

    return 0


if __name__ == "__main__":
    exit(main())
