import time
import numpy as np
import torch
import os

from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from src.model_module.compression_classifier import CompressionClassifier
from src.model_module.load_model import load_model
from src.data_module.preprocessing import create_tokenizer
from src.common_utils import seed_all


# -----------------------------
# Utility functions for inserting COMP tokens
# -----------------------------
def insert_static_comp(input_ids: torch.LongTensor, interval: int, comp_id: int) -> torch.LongTensor:
    toks = input_ids.tolist()
    out = []
    for i, t in enumerate(toks, 1):
        out.append(t)
        if i % interval == 0:
            out.append(comp_id)
    return torch.tensor(out, dtype=torch.long)


def insert_dynamic_comp(input_ids: torch.LongTensor, hidden_states: torch.FloatTensor,
                        classifier: CompressionClassifier, threshold: float, comp_id: int):
    if hidden_states.dim() == 3:
        hidden_states = hidden_states.squeeze(0)
    # CompressionClassifier.predict() returns probs directly (shape: seq_len)
    probs = classifier.predict(hidden_states.to(input_ids.device)).detach().cpu().numpy()
    toks = input_ids.tolist()
    out = []
    for i, t in enumerate(toks):
        out.append(t)
        if probs[i] >= threshold:
            out.append(comp_id)
    return torch.tensor(out, dtype=torch.long), probs


# -----------------------------
# Core evaluation functions
# NOTE: model_wrapper should be a wrapper/dataclass that provides:
#   - .model: the actual language model (for device access)
#   - .tokenizer: the tokenizer
#   - .generate(input_ids): generates text from input
#   - .forward(input_ids, return_hidden): returns forward pass with hidden states
#   - .estimate_kv_size(): estimates KV cache size


def run_static_baseline(example, model_wrapper, comp_id: int, interval: int):
    """Run static interval-based compression baseline.
    
    Args:
        example: Dict with 'input_ids' key
        model_wrapper: Wrapper providing .generate(), .estimate_kv_size(), .tokenizer
        comp_id: Token ID for <COMP> token
        interval: Number of tokens between <COMP> insertions
    
    Returns:
        Dict with 'gen_text', 'latency', 'kv_est', 'comp_count'
    """
    input_ids = example["input_ids"].unsqueeze(0)
    input_ids_with_comp = insert_static_comp(input_ids.squeeze(0), interval, comp_id).unsqueeze(0)

    t0 = time.time()
    gen_ids = model_wrapper.generate(input_ids_with_comp)
    latency = time.time() - t0

    kv_est = model_wrapper.estimate_kv_size()
    gen_text = model_wrapper.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return {"gen_text": gen_text, "latency": latency, "kv_est": kv_est,
            "comp_count": (input_ids_with_comp == comp_id).sum().item()}


def run_dynamic_classifier(example, model_wrapper, classifier: CompressionClassifier, comp_id: int, threshold: float):
    """Run dynamic classifier-based compression.
    
    Args:
        example: Dict with 'input_ids' key
        model_wrapper: Wrapper providing .generate(), .forward(), .estimate_kv_size(), .tokenizer, .model
        classifier: Trained CompressionClassifier
        comp_id: Token ID for <COMP> token
        threshold: Probability threshold for inserting <COMP> tokens
    
    Returns:
        Dict with 'gen_text', 'latency', 'kv_est', 'comp_count', 'probs'
    """
    input_ids = example["input_ids"].unsqueeze(0).to(model_wrapper.model.device)
    fwd = model_wrapper.forward(input_ids, return_hidden=True)
    hidden = fwd["hidden_states"]
    if hidden is None:
        raise RuntimeError("Model did not return hidden states required for classifier.")

    input_with_comp, probs = insert_dynamic_comp(input_ids.squeeze(0).cpu(), hidden.cpu(),
                                                 classifier, threshold, comp_id)
    input_with_comp = input_with_comp.unsqueeze(0)

    t0 = time.time()
    gen_ids = model_wrapper.generate(input_with_comp.to(model_wrapper.model.device))
    latency = time.time() - t0

    kv_est = model_wrapper.estimate_kv_size()
    gen_text = model_wrapper.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    comp_count = (input_with_comp == comp_id).sum().item()

    return {"gen_text": gen_text, "latency": latency, "kv_est": kv_est,
            "comp_count": comp_count, "probs": probs}


# -----------------------------
# Accuracy function (task-dependent)
# -----------------------------
def accuracy_from_generation(generated_text: str, reference_text: str) -> float:
    return 1.0 if generated_text.strip() == reference_text.strip() else 0.0


# -----------------------------
# Orchestration
# -----------------------------
def evaluate(dataset, model_wrapper, classifier: CompressionClassifier, comp_id: int,
             static_interval: int, cls_threshold: float, max_examples: int = 100):
    results = {"static": [], "dynamic": []}

    for i, item in enumerate(dataset):
        if i >= max_examples:
            break
        ex = {"input_ids": item["input_ids"], "reference": item.get("reference", "")}

        # Static baseline
        res_static = run_static_baseline(ex, model_wrapper, comp_id, static_interval)
        res_static["acc"] = accuracy_from_generation(res_static["gen_text"], ex["reference"])
        results["static"].append(res_static)

        # Dynamic classifier
        res_dyn = run_dynamic_classifier(ex, model_wrapper, classifier, comp_id, cls_threshold)
        res_dyn["acc"] = accuracy_from_generation(res_dyn["gen_text"], ex["reference"])
        results["dynamic"].append(res_dyn)

        print(f"[{i+1}] static_acc={res_static['acc']} dyn_acc={res_dyn['acc']} "
              f"static_kv={res_static['kv_est']} dyn_kv={res_dyn['kv_est']} "
              f"comps: static={res_static['comp_count']} dyn={res_dyn['comp_count']}")

    # Aggregate metrics
    def summarize(arr):
        if len(arr) == 0:
            return {}
        return {
            "accuracy_mean": float(np.mean([x["acc"] for x in arr])),
            "latency_mean": float(np.mean([x["latency"] for x in arr])),
            "kv_mean": float(np.mean([x["kv_est"] for x in arr])),
            "comp_mean": float(np.mean([x["comp_count"] for x in arr]))
        }

    summary = {"static": summarize(results["static"]), "dynamic": summarize(results["dynamic"])}
    return {"per_example": results, "summary": summary}

def main(CCM_model_path: str, classifier_model_path: str, data_path: str = None, seed: int = 42):
    """Evaluate compression performance with static and dynamic classifiers.
    
    Args:
        CCM_model_path: Path to the trained CCM model (e.g., outputs/baseline)
        classifier_model_path: Path to the trained classifier (e.g., outputs/classifier)
        data_path: Path to evaluation dataset JSON file
        seed: Random seed for reproducibility
    
    TODO:
        - Load dataset from data_path or use provided dataset
        - Initialize CCM_model using appropriate wrapper
        - Define COMP_TOKEN_ID based on tokenizer
        - Implement model_wrapper with required interface (.generate, .forward, etc.)
    """
    # Set random seed
    seed_all(seed)
    
    # Load classifier config and model
    config_path = os.path.join(classifier_model_path, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. "
            f"Expected Hydra config in {classifier_model_path}/.hydra/"
        )
    
    configs = OmegaConf.load(config_path)
    
    # Initialize tokenizer
    tokenizer = create_tokenizer(configs)
    
    # Load classifier model
    hidden_size = AutoConfig.from_pretrained(
        configs.model_args.model_name_or_path
    ).hidden_size
    classifier = CompressionClassifier(
        hidden_size=hidden_size,
        dropout=configs.classifier_args.dropout if hasattr(configs, 'classifier_args') else 0.1
    )
    classifier.load_classifier(classifier_model_path)
    classifier.eval()
    
    # TODO: Load dataset
    # If data_path is provided, load from file:
    # from src.data_module.load_data import load_gsm8k_compressed
    # dataset = load_gsm8k_compressed(data_path, train_dev_test_split_ratio=[0.8, 0.1, 0.1], seed=seed)
    # dataset = dataset['test']  # Use test split for evaluation
    dataset = None  # Placeholder
    
    # TODO: Initialize CCM model and model_wrapper
    # CCM_model = load_model(configs)  # or appropriate loading mechanism
    # model_wrapper = ModelWrapper(model=CCM_model, tokenizer=tokenizer)
    CCM_model = None
    model_wrapper = None
    COMP_TOKEN_ID = None  # Define based on tokenizer or config
    
    if dataset is None or model_wrapper is None or COMP_TOKEN_ID is None:
        raise NotImplementedError(
            "Please implement: dataset loading, model initialization, and COMP_TOKEN_ID definition"
        )
    
    results = evaluate(
        dataset=dataset,
        model_wrapper=model_wrapper,
        classifier=classifier,
        comp_id=COMP_TOKEN_ID,
        static_interval=128,
        cls_threshold=0.5,
        max_examples=100
    )

    print("Summary metrics:", results["summary"])

if __name__ == "__main__":
    # Directory paths for baseline and compression probe models. E.g. outputs/classifier stores .hydra and compression_classifier.pt
    main(CCM_model_path="outputs/baseline", classifier_model_path="outputs/classifier")