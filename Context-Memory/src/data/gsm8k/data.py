"""GSM8K Dataset for CCM compression training.

This module loads GSM8K samples with pre-embedded <COMP> tokens for reasoning compression.
Unlike dialogue/MetaICL where CCM adds <COMP> between examples, here <COMP> tokens are
already placed at natural reasoning checkpoints within each problem's solution.
"""

import os
import json
import torch
import numpy as np
from datasets import Dataset, DatasetDict
from collections import defaultdict


class GSM8KDataset:
    """Dataset class for GSM8K with embedded <COMP> tokens for reasoning compression."""

    def __init__(
        self,
        tokenizer,
        data_path: str = None,
        comp_token=None,
        max_length: int = 2048,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            data_path: Path to gsm8k_compressed_train.json
            comp_token: Compression token ID(s)
            max_length: Maximum sequence length
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (rest is test)
            seed: Random seed for splitting
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = tokenizer.name_or_path.split('/')[-1]

        # Token setup
        self.bos_token = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
        self.eos_token = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        self.comp_token = comp_token if comp_token is not None else []
        if isinstance(self.comp_token, int):
            self.comp_token = [self.comp_token]

        # Load data
        if data_path is None:
            # Default path relative to this file (src/data/gsm8k/data.py)
            # Goes 5 levels up to Project dir where gsm8k_compressed_train.json lives
            data_path = os.path.join(
                os.path.dirname(__file__),
                "../../../../../gsm8k_compressed_train.json"
            )

        self.raw_data = self._load_json(data_path)
        print(f"Loaded {len(self.raw_data)} GSM8K samples from {data_path}")

        # Tokenize and split
        self.tokenized_data = self._tokenize_all(self.raw_data)
        self.train_data, self.val_data, self.test_data = self._split_data(
            self.tokenized_data, train_ratio, val_ratio, seed
        )

        # Create HuggingFace datasets
        self.train_dataset = Dataset.from_dict(self._to_dict(self.train_data), split='train')
        self.eval_dataset = DatasetDict({
            'validation': Dataset.from_dict(self._to_dict(self.val_data), split='validation'),
            'test': Dataset.from_dict(self._to_dict(self.test_data), split='test'),
        })

        print(f"Train: {len(self.train_dataset)}, Val: {len(self.eval_dataset['validation'])}, Test: {len(self.eval_dataset['test'])}")

    def _load_json(self, path: str) -> list:
        """Load JSON data file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _tokenize_all(self, data: list) -> list:
        """Tokenize all samples, keeping <COMP> token positions."""
        tokenized = []
        comp_text = "<COMP>"

        for i, sample in enumerate(data):
            question = sample['question']
            reasoning = sample['reasoning_with_compression']
            answer = sample['answer']

            # Format: Question + Reasoning (with <COMP>) + Final Answer
            # The reasoning already contains <COMP> tokens
            full_text = f"Question: {question}\n\nSolution: {reasoning}"

            # Split by <COMP> to tokenize each segment, then insert comp_token
            segments = reasoning.split(comp_text)

            # Tokenize question prefix
            question_text = f"Question: {question}\n\nSolution: "
            question_tokens = self.tokenizer.encode(question_text, add_special_tokens=False)

            # Build token sequence with <COMP> tokens inserted
            reasoning_tokens = []
            for j, segment in enumerate(segments):
                seg_tokens = self.tokenizer.encode(segment.strip(), add_special_tokens=False)
                reasoning_tokens.extend(seg_tokens)

                # Add <COMP> token after each segment except the last
                if j < len(segments) - 1 and len(self.comp_token) > 0:
                    reasoning_tokens.extend(self.comp_token)

            # Combine: [BOS] + question + reasoning + [EOS]
            input_ids = self.bos_token + question_tokens + reasoning_tokens + self.eos_token

            # Find positions where <COMP> tokens appear (for labels)
            comp_positions = []
            for k, tok in enumerate(input_ids):
                if tok in self.comp_token:
                    comp_positions.append(k)

            # Skip if too long
            if len(input_ids) > self.max_length:
                continue

            tokenized.append({
                'input_ids': input_ids,
                'comp_positions': comp_positions,
                'question': question,
                'answer': answer,
                'n_comp_tokens': len(comp_positions),
            })

            if i % 100 == 0:
                print(f"Tokenizing {i}/{len(data)}", end='\r')

        print(f"\nTokenized {len(tokenized)}/{len(data)} samples (skipped {len(data) - len(tokenized)} too long)")
        return tokenized

    def _split_data(self, data: list, train_ratio: float, val_ratio: float, seed: int):
        """Split data into train/val/test."""
        np.random.seed(seed)
        indices = np.random.permutation(len(data))

        n_train = int(len(data) * train_ratio)
        n_val = int(len(data) * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        test_data = [data[i] for i in test_idx]

        return train_data, val_data, test_data

    def _to_dict(self, data: list) -> dict:
        """Convert list of dicts to dict of lists for HuggingFace Dataset."""
        result = defaultdict(list)
        for item in data:
            for key, value in item.items():
                result[key].append(value)
        return dict(result)

    def get_sample(self, idx: int, split: str = 'train'):
        """Get a formatted sample for debugging."""
        if split == 'train':
            sample = self.train_dataset[idx]
        else:
            sample = self.eval_dataset[split][idx]

        decoded = self.tokenizer.decode(sample['input_ids'])
        return {
            'decoded': decoded,
            'n_comp_tokens': sample['n_comp_tokens'],
            'answer': sample['answer'],
        }


if __name__ == '__main__':
    from transformers import AutoTokenizer

    # Test with LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Meta-Llama-3.1-8B-Instruct')
    tokenizer.pad_token = tokenizer.eos_token

    # Add <COMP> token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<COMP>"]})
    comp_token = tokenizer.convert_tokens_to_ids("<COMP>")

    # Load dataset
    dataset = GSM8KDataset(
        tokenizer=tokenizer,
        comp_token=[comp_token],
        max_length=2048,
    )

    # Show sample
    print("\n=== Sample ===")
    sample = dataset.get_sample(0)
    print(f"Decoded: {sample['decoded'][:500]}...")
    print(f"N comp tokens: {sample['n_comp_tokens']}")
    print(f"Answer: {sample['answer']}")
