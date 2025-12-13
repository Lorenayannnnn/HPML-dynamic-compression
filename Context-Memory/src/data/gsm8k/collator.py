"""Data collator for GSM8K with embedded <COMP> tokens.

This collator handles batching for reasoning compression where <COMP> tokens
are already embedded in the reasoning traces at natural checkpoint positions.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union, List

import torch
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import PaddingStrategy

logger = logging.getLogger(__name__)


def pad_inputs_simple(model_inputs, label_pad_token_id, pad_token):
    """Simple left-padding for decoder models."""
    max_len = max(len(ids) for ids in model_inputs["input_ids"])

    for i in range(len(model_inputs["input_ids"])):
        pad_len = max_len - len(model_inputs["input_ids"][i])
        # Left padding
        model_inputs["input_ids"][i] = [pad_token] * pad_len + model_inputs["input_ids"][i]
        model_inputs["attention_mask"][i] = [0] * pad_len + model_inputs["attention_mask"][i]
        model_inputs["labels"][i] = [label_pad_token_id] * pad_len + model_inputs["labels"][i]

    # Convert to tensors
    model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
    model_inputs["attention_mask"] = torch.tensor(model_inputs["attention_mask"])
    model_inputs["labels"] = torch.tensor(model_inputs["labels"])

    return model_inputs


@dataclass
class DataCollatorForGSM8K_LLAMA:
    """Data collator for GSM8K reasoning compression with LLaMA models.

    Unlike dialogue/MetaICL collators that add <COMP> tokens during collation,
    this collator expects <COMP> tokens to already be present in the input_ids.
    It sets up the proper attention masks for CCM's compression mechanism.
    """
    gsm8k: Optional[Any] = None
    tokenizer: PreTrainedTokenizerBase = None
    comp_args: Optional[Any] = None  # Duck-typed to avoid import issues
    model: Optional[Any] = None
    comp_token: Union[int, List[int]] = 32000
    sum_token: Optional[int] = None
    return_tensors: str = "pt"
    padding: Union[bool, str, PaddingStrategy] = 'left'
    pad_token: int = 0
    label_pad_token_id: int = -100
    max_length: int = 2048

    def __call__(self, batch, return_tensors=None):
        if isinstance(self.comp_token, int):
            self.comp_token = [self.comp_token]

        model_inputs = defaultdict(list)

        for instance in batch:
            input_ids = instance['input_ids']

            # For causal LM training, we predict the next token
            # Labels are shifted input_ids (handled by model)
            full_token = input_ids
            labels = input_ids.copy()

            # Mask COMP tokens in labels with -100 so they're ignored in loss
            # COMP tokens are for compression, not prediction targets
            for i, tok in enumerate(labels):
                if tok in self.comp_token:
                    labels[i] = self.label_pad_token_id

            # For reasoning compression, we train on the full sequence
            # The model learns to compress context at <COMP> positions
            model_inputs["input_ids"].append(full_token)
            model_inputs["labels"].append(labels)
            model_inputs["attention_mask"].append([1] * len(full_token))

        # Simple padding - CCM's special attention masks will be set up by the model
        model_inputs = pad_inputs_simple(
            model_inputs,
            self.label_pad_token_id,
            self.pad_token
        )

        # Convert defaultdict to regular dict for accelerate compatibility
        return dict(model_inputs)


@dataclass
class DataCollatorForGSM8K_Eval:
    """Simpler collator for evaluation - just pads and creates masks."""
    tokenizer: PreTrainedTokenizerBase = None
    comp_token: Union[int, List[int]] = 32000
    pad_token: int = 0
    label_pad_token_id: int = -100
    max_length: int = 2048

    def __call__(self, batch, return_tensors=None):
        if isinstance(self.comp_token, int):
            self.comp_token = [self.comp_token]

        model_inputs = defaultdict(list)

        for instance in batch:
            input_ids = instance['input_ids']
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(input_ids.copy())

            # Store metadata for evaluation
            if 'answer' in instance:
                model_inputs["answers"].append(instance['answer'])
            if 'n_comp_tokens' in instance:
                model_inputs["n_comp_tokens"].append(instance['n_comp_tokens'])

        # Simple padding
        model_inputs = pad_inputs_simple(
            model_inputs,
            self.label_pad_token_id,
            self.pad_token
        )

        return model_inputs
