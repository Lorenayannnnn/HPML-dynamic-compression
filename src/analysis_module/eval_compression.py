import time
import numpy as np
import torch

from omegaconf import OmegaConf
from transformers import AutoConfig

from src.model_module.compression_classifier import CompressionClassifier


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
# -----------------------------
def run_static_baseline(example, model_wrapper, comp_id: int, interval: int):
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

def main(CCM_model_path: str, classifier_model_path: str):
    dataset = None      # TODO

    CCM_model = None  # TODO: Our baseline / model finetuned on our dataset following the CCM method
    COMP_TOKEN_ID = None    # TODO

    configs = OmegaConf.load(f"{classifier_model_path}/.hydra/config.yaml")
    hidden_size = AutoConfig.from_pretrained(configs.model_args.model_name_or_path).hidden_size
    classifier = CompressionClassifier(hidden_size=hidden_size, dropout=configs.classifier_args.dropout)
    classifier.load_classifier(classifier_model_path)
    classifier.eval()

    results = evaluate(
        dataset=dataset,
        model_wrapper=CCM_model,
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