"""
Custom data collator for compression classifier training

Handles padding of input_ids, attention_mask, and labels (with -100 for ignored tokens)
"""

from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Any

from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch
        
        # Separate features that need padding from those that don't
        skip_pad_features = ['chat_gpt_cot_w_COMP', 'chat_gpt_cot_wo_COMP', "next_is_COMP"]
        
        batch = {}
        
        # Prepare features for standard padding (input_ids, attention_mask)
        features_to_be_padded = []
        labels_to_pad = []
        
        for f in features:
            features_to_be_padded.append({
                'input_ids': f['input_ids'],
                'attention_mask': f['attention_mask']
            })
            labels_to_pad.append(f['labels'])
            
            for feat_name in skip_pad_features:
                if feat_name in f:
                    if feat_name not in batch:
                        batch[feat_name] = []
                    batch[feat_name].append(f[feat_name])
            
        # Pad input_ids and attention_mask
        padding_result = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features_to_be_padded,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        batch.update(padding_result)
        
        # Manually pad labels with -100 (ignore index)
        # Respect the tokenizer's padding_side
        max_length = batch['input_ids'].shape[1]
        padded_labels = []
        for labels in labels_to_pad:
            padding_length = max_length - len(labels)
            if self.tokenizer.padding_side == "left":
                padded = [-100] * padding_length + labels
            else:
                padded = labels + [-100] * padding_length
            padded_labels.append(padded)
        
        # Use standard HuggingFace naming convention
        batch['labels'] = torch.tensor(padded_labels, dtype=torch.long)
        
        return batch