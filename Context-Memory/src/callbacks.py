"""Custom wandb integrations"""

import dataclasses
import json
import os

import wandb
try:
    from transformers.integrations import TrainerCallback, WandbCallback
except:
    from transformers import TrainerCallback
    from transformers.integrations import WandbCallback
from transformers.utils import logging

# Compatibility shim: is_torch_tpu_available was removed in transformers 4.41+
def is_torch_tpu_available():
    return False

from .arguments import Arguments

logger = logging.get_logger(__name__)

# File to store run ID for checkpoint and wandb resumption
RUN_ID_FILE = "run_id.json"


def get_run_id_path(output_dir: str) -> str:
    """Get path to run ID file."""
    return os.path.join(output_dir, RUN_ID_FILE)


def load_run_id(output_dir: str) -> str | None:
    """Load existing run ID from output directory if it exists."""
    run_id_path = get_run_id_path(output_dir)
    if os.path.exists(run_id_path):
        try:
            with open(run_id_path, "r") as f:
                data = json.load(f)
                run_id = data.get("run_id")
                if run_id:
                    logger.info(f"Found existing run ID: {run_id}")
                    return run_id
        except Exception as e:
            logger.warning(f"Failed to load run ID: {e}")
    return None


def save_run_id(output_dir: str, run_id: str) -> None:
    """Save run ID to output directory for future resumption."""
    os.makedirs(output_dir, exist_ok=True)
    run_id_path = get_run_id_path(output_dir)
    try:
        with open(run_id_path, "w") as f:
            json.dump({"run_id": run_id}, f)
        logger.info(f"Saved run ID to {run_id_path}")
    except Exception as e:
        logger.warning(f"Failed to save run ID: {e}")


class CustomWandbCallback(WandbCallback):

    def __init__(self, wandb_args: Arguments, *args, **kwargs):
        """Just do standard wandb init, but save the arguments for setup."""
        super().__init__(*args, **kwargs)
        self._wandb_args = wandb_args

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.
        Supports resuming from a previous run if checkpoint exists.

        One can subclass and override this method to customize the setup if
        needed. Find more information
        [here](https://docs.wandb.ai/integrations/huggingface). You can also
        override the following environment variables:
        Environment:
            WANDB_LOG_MODEL (`bool`, *optional*, defaults to `False`):
                Whether or not to log model as artifact at the end of training.
                Use along with
                *TrainingArguments.load_best_model_at_end* to upload best model.
            WANDB_WATCH (`str`, *optional* defaults to `"gradients"`):
                Can be `"gradients"`, `"all"` or `"false"`. Set to `"false"` to
                disable gradient logging or `"all"` to log gradients and
                parameters.
        """
        del args  # Use self._wandb_args instead.
        args = self._wandb_args

        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            if self._wandb.run is None:
                output_dir = args.training.output_dir

                # Priority: 1) command-line run_id, 2) auto-detected from file
                run_id_to_resume = getattr(args.wandb, 'run_id', None) or load_run_id(output_dir)

                # Build init kwargs, omitting None values
                init_kwargs = {
                    "project": args.wandb.project,
                    "settings": wandb.Settings(start_method="fork"),
                }
                if args.wandb.entity:  # Only include entity if explicitly set
                    init_kwargs["entity"] = args.wandb.entity

                if run_id_to_resume:
                    # Resume existing run
                    logger.info(f"Resuming wandb run: {run_id_to_resume}")
                    self._wandb.init(
                        **init_kwargs,
                        id=run_id_to_resume,
                        resume="allow",
                    )
                else:
                    # Start new run
                    self._wandb.init(
                        **init_kwargs,
                        group=args.wandb.group,
                        name=args.wandb.name,
                        config=dataclasses.asdict(args),
                    )
                    # Save run ID for future resumption
                    if self._wandb.run is not None:
                        save_run_id(output_dir, self._wandb.run.id)

            # define default x-axis (for latest wandb versions)
            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

            # keep track of model topology and gradients, unsupported on TPU
            if not is_torch_tpu_available() and os.getenv("WANDB_WATCH") != "false":
                self._wandb.watch(
                    model,
                    log=os.getenv("WANDB_WATCH", "gradients"),
                    log_freq=max(100, args.training.logging_steps),
                )


class EvaluateFirstStepCallback(TrainerCallback):

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True
