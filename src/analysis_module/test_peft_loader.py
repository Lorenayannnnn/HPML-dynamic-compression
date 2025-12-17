"""
Quick validation that the new PEFT loader function works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model_module.peft_loader import load_peft_model, get_comp_and_newline_tokens

# Test paths
OURS_model_path = Path("outputs/OURS_llama-3.1-8b-instruct-online-concat_recur").resolve()
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"

print("="*70)
print("PEFT LOADER FUNCTION TEST")
print("="*70)

try:
    print("\n[TEST 1] Loading OURS model with PEFT adapter...")
    model, tokenizer = load_peft_model(
        base_model_id=base_model_id,
        adapter_path=str(OURS_model_path),
        device='cpu',
        dtype=torch.float32
    )
    print("✓ Model loaded successfully")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Model device: {next(model.parameters()).device}")
    print(f"  - Config hidden_size: {model.config.hidden_size}")
    
    print("\n[TEST 2] Getting special token IDs...")
    comp_token_id, newline_token_ids = get_comp_and_newline_tokens(tokenizer)
    print(f"✓ Tokens extracted successfully")
    print(f"  - COMP0 token ID: {comp_token_id}")
    print(f"  - Newline token IDs: {newline_token_ids}")
    
    print("\n[TEST 3] Verifying model state...")
    assert model.training == False, "Model should be in eval mode"
    print("✓ Model is in eval mode")
    
    # Quick forward pass test
    print("\n[TEST 4] Testing forward pass...")
    test_text = "What is 2+2?"
    input_ids = tokenizer.encode(test_text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
    print(f"✓ Forward pass successful")
    print(f"  - Output logits shape: {outputs.logits.shape}")
    print(f"  - Hidden states layers: {len(outputs.hidden_states)}")
    
    print("\n" + "="*70)
    print("ALL LOADER TESTS PASSED!")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
