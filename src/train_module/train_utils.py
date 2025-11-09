import transformers

def create_trainer_args(configs):
    args = configs.training_args
    training_args = configs.training_args
    # TODO: change step num based on dataset size
    args = transformers.TrainingArguments(
        per_device_train_batch_size=training_args.micro_batch_size,
        gradient_accumulation_steps=training_args.batch_size // training_args.micro_batch_size,
        warmup_steps=100,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if training_args.do_eval else "no",
        save_strategy=training_args.save_strategy,
        eval_steps=200 if training_args.do_eval else None,
        save_steps=200,
        output_dir=training_args.output_dir,
        save_total_limit=training_args.save_total_limit,
        load_best_model_at_end=True if training_args.do_eval else False,
        metric_for_best_model="eval_loss",
        report_to="wandb" if configs.use_wandb else None,
        run_name=training_args.wandb_run_name if training_args.use_wandb else None,
    )
    return args


def compute_metrics(eval_pred):
    """
    Compute metrics for binary classification.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels
    
    Returns:
        Dictionary of metrics
    """
    import numpy as np
    from sklearn.metrics import accuracy_score
    # from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Flatten and filter out -100 labels
    flat_logits = logits.reshape(-1)
    flat_labels = labels.reshape(-1)
    
    valid_mask = (flat_labels != -100)
    valid_logits = flat_logits[valid_mask]
    valid_labels = flat_labels[valid_mask]
    
    # Get predictions
    probs = 1 / (1 + np.exp(-valid_logits))  # Sigmoid
    predictions = (probs > 0.5).astype(int)
    
    # Compute metrics
    # Only use accuracy for now given the data is balanced
    accuracy = accuracy_score(valid_labels, predictions)
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     valid_labels, predictions, average='binary', zero_division=0
    # )
    
    # try:
    #     auc = roc_auc_score(valid_labels, probs)
    # except ValueError:
    #     auc = 0.0  # In case there's only one class
    
    return {
        'accuracy': accuracy,
        # 'precision': precision,
        # 'recall': recall,
        # 'f1': f1,
        # 'auc': auc,
    }


def create_trainer(model, tokenizer, data_collator, train_data, val_data, trainer_args):
    """
    Create a HuggingFace Trainer for the compression probe model.
    
    Args:
        model: CompressionProbeModel (wraps LM + classifier)
        tokenizer: Tokenizer for padding
        train_data: Training dataset
        val_data: Validation dataset
        trainer_args: TrainingArguments
        data_collator: Optional data collator
    
    Returns:
        Trainer instance
    """
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=trainer_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    return trainer