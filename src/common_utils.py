"""
Common utility functions for training and evaluation.

Includes:
- Random seed management for reproducibility
- WandB initialization and logging helpers
- Configuration setup utilities
"""

import os
import random

import hashlib
import subprocess
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf


def _git_commit_hash() -> Optional[str]:
    """Get current git commit hash for experiment tracking"""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None


def _sha256_file(path: str) -> Optional[str]:
    """Compute SHA256 hash of a file for artifact versioning"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def seed_all(seed):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_wandb(configs):
    """Initialize WandB logging if enabled in config"""
    # Check if parameter passed or if set within environ
    if (
        configs.training_args.use_wandb
        and (
            len(configs.training_args.wandb_project) > 0
            or ("WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0)
        )
        and configs.training_args.do_train
    ):
        configs.training_args.use_wandb = True
        # Only overwrite environ if wandb param passed
        if len(configs.training_args.wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = configs.training_args.wandb_project
        if len(configs.training_args.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = configs.training_args.wandb_watch
        if len(configs.training_args.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = configs.training_args.wandb_log_model
        # Use OmegaConf to allow adding new keys
        OmegaConf.set_struct(configs, False)
        configs.wandb_run_name = configs.training_args.output_dir.split("/")[-1]
        wandb.init(project=os.environ["WANDB_PROJECT"], name=configs.wandb_run_name)
    else:
        configs.training_args.use_wandb = False
    return configs


def setup(configs):
    """Initialize training: set seeds, handle checkpoint resume, setup WandB"""
    if configs.training_args.resume_from_checkpoint is not None:
        # Load configs from checkpoint
        output_dir = os.path.dirname(configs.training_args.resume_from_checkpoint)
        loaded_configs = OmegaConf.load(os.path.join(output_dir, "configs.yaml"))
        loaded_configs.training_args.do_train = configs.training_args.do_train
        loaded_configs.training_args.do_predict = configs.training_args.do_predict
        loaded_configs.training_args.resume_from_checkpoint = (
            configs.training_args.resume_from_checkpoint
        )
        configs = loaded_configs

    seed_all(configs.training_args.seed)

    configs = prepare_wandb(configs)

    return configs


def init_wandb_eval(
    *,
    enabled: bool,
    project: str,
    entity: Optional[str] = None,
    group: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """
    W&B init for evaluation scripts (separate from training_args/do_train logic).
    Returns wandb.Run or None.
    """
    if not enabled:
        return None

    # Respect environment defaults, allow CLI override
    if project:
        os.environ["WANDB_PROJECT"] = project

    run = wandb.init(
        entity=entity,
        project=os.environ.get("WANDB_PROJECT", project),
        group=group,
        name=name,
        job_type="eval",
        tags=tags or [],
        config=config or {},
    )
    return run


def log_profiled_eval_results(
    run,
    *,
    results: Dict[str, Any],
    output_json_path: str,
    log_per_sample_table: bool = True,
    artifact_name: str = "profiled_eval_results",
):
    """
    Logs the output of profiled_eval.py:
      - classifier_profile scalars
      - aggregated metrics per method
      - (optional) per-sample W&B Table
      - output JSON as a W&B Artifact
    """
    if run is None:
        return

    agg = results.get("aggregated", {})
    cls_prof = results.get("classifier_profile", {})

    # standalone profiling for the classifer
    if cls_prof:
        wandb.log(
            {
                "cls_profile/avg_latency_ms": cls_prof.get("avg_latency_ms"),
                "cls_profile/std_latency_ms": cls_prof.get("std_latency_ms"),
                "cls_profile/min_latency_ms": cls_prof.get("min_latency_ms"),
                "cls_profile/max_latency_ms": cls_prof.get("max_latency_ms"),
                "cls_profile/profiler_cuda_time_ms": cls_prof.get(
                    "profiler_cuda_time_ms"
                ),
                "cls_profile/profiler_cpu_time_ms": cls_prof.get(
                    "profiler_cpu_time_ms"
                ),
                "cls_profile/memory_mb": cls_prof.get("memory_mb"),
                "cls_profile/num_parameters": cls_prof.get("num_parameters"),
            }
        )

    # per-method aggregates
    for method, m in agg.items():
        prefix = f"agg/{method}"
        wandb.log(
            {
                f"{prefix}/accuracy_pct": m.get("accuracy_pct"),
                f"{prefix}/avg_comp_tokens": m.get("avg_comp_tokens"),
                f"{prefix}/avg_kv_cache_mb": m.get("avg_kv_cache_mb"),
                f"{prefix}/compression_ratio": m.get("compression_ratio"),
                f"{prefix}/avg_latency_sec": m.get("avg_latency_sec"),
                f"{prefix}/avg_throughput_tps": m.get("avg_throughput_tps"),
                f"{prefix}/avg_tokens_generated": m.get("avg_tokens_generated"),
                f"{prefix}/avg_classifier_time_ms": m.get("avg_classifier_time_ms"),
                f"{prefix}/classifier_overhead_pct": m.get("classifier_overhead_pct"),
                f"{prefix}/avg_peak_memory_mb": m.get("avg_peak_memory_mb"),
                f"{prefix}/classifier_memory_mb": m.get("classifier_memory_mb"),
                f"{prefix}/total_time_sec": m.get("total_time_sec"),
            }
        )

    # per-sample table (optional)
    if log_per_sample_table:
        table = wandb.Table(
            columns=[
                "sample_id",
                "method",
                "accuracy",
                "comp_tokens",
                "tokens_generated",
                "kv_cache_mb",
                "latency_seconds",
                "throughput_tps",
                "classifier_calls",
                "classifier_time_ms",
                "classifier_overhead_pct",
                "peak_memory_mb",
            ]
        )
        per_sample = results.get("per_sample", {})
        for method, rows in per_sample.items():
            for r in rows:
                table.add_data(
                    r.get("sample_id"),
                    r.get("method", method),
                    r.get("accuracy"),
                    r.get("comp_tokens"),
                    r.get("tokens_generated"),
                    r.get("kv_cache_mb"),
                    r.get("latency_seconds"),
                    r.get("throughput_tokens_per_sec"),
                    r.get("classifier_calls", 0),
                    r.get("classifier_time_ms", 0.0),
                    r.get("classifier_overhead_pct", 0.0),
                    r.get("peak_memory_mb", 0.0),
                )
        wandb.log({"per_sample_metrics": table})

    # final WandB artifact: the JSON output (checkpoint/final)
    art = wandb.Artifact(
        name=artifact_name,
        type="eval_results",
        metadata={
            "git_commit": _git_commit_hash(),
            "output_sha256": _sha256_file(output_json_path),
        },
    )
    art.add_file(output_json_path)
    wandb.log_artifact(art)
