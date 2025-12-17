"""
Utility functions for loading and merging PEFT adapters.
"""

import torch
import warnings
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_peft_model(
    base_model_id: str,
    adapter_path: str,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
    trust_remote_code: bool = True,
) -> tuple:
    """
    Load a base model and apply PEFT adapter, then merge.
    
    Args:
        base_model_id: HuggingFace model ID or local path for base model
        adapter_path: Path to PEFT adapter directory
        device: Device to load model on ('cpu', 'cuda', 'mps', or 'auto')
        dtype: PyTorch dtype for model weights (torch.float32, torch.float16, etc.)
        trust_remote_code: Whether to trust remote code execution
        
    Returns:
        tuple: (merged_model, tokenizer)
    """
    adapter_path = Path(adapter_path).resolve()
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    
    print(f"[PEFT Loader] Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        fix_mistral_regex=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[PEFT Loader] ✓ Tokenizer loaded")
    
    print(f"[PEFT Loader] Loading base model {base_model_id}...")
    # KEY: Use _fast_init=False to avoid meta device issues with PEFT
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map=device if device != 'cuda' else 'auto',
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        local_files_only=False,
        _fast_init=False  # Critical for PEFT compatibility
    )
    print(f"[PEFT Loader] ✓ Base model loaded")
    
    print(f"[PEFT Loader] Applying PEFT adapter from {adapter_path}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False
        )
    print(f"[PEFT Loader] ✓ PEFT adapter applied")
    
    print(f"[PEFT Loader] Merging adapter weights into base model...")
    model = model.merge_and_unload()
    model.eval()
    print(f"[PEFT Loader] ✓ Adapter merged and unloaded")
    
    return model, tokenizer


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
