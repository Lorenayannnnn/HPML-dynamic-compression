"""
Utility functions for loading and merging PEFT adapters.

Uses CCMInferenceCLI for robust CCM adapter loading with proper:
- COMP token handling
- LoRA weight loading without merging (preserves CCM's conditional layers)
- Dtype and device management
"""

import torch
import warnings
from pathlib import Path
from typing import Optional, Tuple

# Import CCMInferenceCLI for robust CCM adapter loading
import sys

# Add project root to path to import inference_cli
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis_module.inference_cli import CCMInferenceCLI


def load_peft_model(
    base_model_id: str,
    adapter_path: str,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float16,
    classifier_path: Optional[str] = None,
) -> Tuple[CCMInferenceCLI, torch.nn.Module]:
    """
    Load a CCM model with PEFT adapter using CCMInferenceCLI.
    
    This function provides a robust loading mechanism that:
    - Properly handles CCM's custom LlamaForCausalLM_CCM architecture
    - Loads LoRA adapters without merging (preserves conditional layers)
    - Sets up COMP token handling correctly
    - Optionally loads compression classifier
    
    Args:
        base_model_id: HuggingFace model ID or local path for base model
        adapter_path: Path to CCM PEFT adapter directory
        device: Device to load model on ('cpu', 'cuda', 'mps')
        dtype: Model dtype (torch.bfloat16, torch.float16, torch.float32)
        classifier_path: Optional path to compression classifier
        
    Returns:
        tuple: (ccm_inference_cli, model)
            - ccm_inference_cli: CCMInferenceCLI instance with full inference capabilities
            - model: The underlying model (for direct access if needed)
            
    Example:
        >>> cli, model = load_peft_model(
        ...     "meta-llama/Llama-3.1-8B-Instruct",
        ...     "outputs/OURS_llama-3.1-8b-instruct-online-concat_recur",
        ...     classifier_path="outputs/classifier"
        ... )
        >>> # Use for generation
        >>> for token, metrics in cli.generate_streaming(prompt, compression_mode="dynamic"):
        ...     print(token, end="", flush=True)
    """
    adapter_path = Path(adapter_path).resolve()
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    
    print(f"[PEFT Loader] Loading CCM model with adapter using CCMInferenceCLI...")
    print(f"  Base model: {base_model_id}")
    print(f"  Adapter: {adapter_path}")
    if classifier_path:
        print(f"  Classifier: {classifier_path}")
    
    # Use CCMInferenceCLI for robust CCM adapter loading
    ccm_cli = CCMInferenceCLI(
        base_model=base_model_id,
        ccm_adapter_path=str(adapter_path),
        classifier_path=classifier_path,
        device=device,
        dtype=dtype,
    )
    
    # Extract the underlying model for direct access
    # CCM model is either:
    # - ccm_cli.model (no adapter)
    # - ccm_cli.model.base_model.model (with adapter - LlamaForCausalLM_CCM)
    model = ccm_cli._get_ccm_model()
    
    print(f"[PEFT Loader] ✓ Model loaded successfully")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Has adapter: {ccm_cli.has_adapter}")
    print(f"  Has classifier: {ccm_cli.classifier is not None}")
    
    return model, ccm_cli


def load_peft_model_legacy(
    base_model_id: str,
    adapter_path: str,
    device: str = 'cpu',
    trust_remote_code: bool = True,
) -> tuple:
    """
    [LEGACY] Load a base model and apply PEFT adapter, then merge.
    
    WARNING: This function merges adapters, which breaks CCM's conditional layers.
    Use load_peft_model() instead for CCM adapters.
    
    This is kept for backwards compatibility with non-CCM adapters only.
    
    Args:
        base_model_id: HuggingFace model ID or local path for base model
        adapter_path: Path to PEFT adapter directory
        device: Device to load model on ('cpu', 'cuda', 'mps', or 'auto')
        trust_remote_code: Whether to trust remote code execution
        
    Returns:
        tuple: (merged_model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    adapter_path = Path(adapter_path).resolve()
    
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    
    print(f"[PEFT Loader - LEGACY] Loading tokenizer from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_path),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[PEFT Loader - LEGACY] ✓ Tokenizer loaded")
    
    print(f"[PEFT Loader - LEGACY] Loading base model {base_model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map=device if device != 'cuda' else 'auto',
        trust_remote_code=trust_remote_code,
        local_files_only=False,
        _fast_init=False  # Critical for PEFT compatibility
    )
    print(f"[PEFT Loader - LEGACY] ✓ Base model loaded")
    
    print(f"[PEFT Loader - LEGACY] Applying PEFT adapter from {adapter_path}...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            is_trainable=False
        )
    print(f"[PEFT Loader - LEGACY] ✓ PEFT adapter applied")
    
    print(f"[PEFT Loader - LEGACY] Merging adapter weights into base model...")
    model = model.merge_and_unload()
    model.eval()
    print(f"[PEFT Loader - LEGACY] ✓ Adapter merged and unloaded")
    
    return model, tokenizer

