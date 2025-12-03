import torch


def load_model(configs):
    """Load frozen LLM + compression classifier for training.

    Supports LLaMA-3.1-8B and other models via AutoModel.
    """
    from transformers import AutoModel, AutoModelForCausalLM
    from src.model_module.compression_classifier import CompressionClassifier
    from src.model_module.compression_probe_model import CompressionProbeModel

    # Determine model path
    model_path = (
        configs.training_args.resume_from_checkpoint
        if hasattr(configs.training_args, 'resume_from_checkpoint') and configs.training_args.resume_from_checkpoint
        else configs.model_args.model_name_or_path
    )

    # Determine cache dir
    cache_dir = configs.data_args.cache_dir if hasattr(configs.data_args, 'cache_dir') else None

    # Load frozen LLM (use AutoModel for hidden states, not AutoModelForCausalLM)
    lm = AutoModel.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,  # Use fp16 for memory efficiency
        device_map="auto",
        trust_remote_code=True,  # Needed for some models like LLaMA-3.1
    )

    # Get dropout from classifier_args or model_args (for backwards compatibility)
    dropout = 0.1
    if hasattr(configs, 'classifier_args') and hasattr(configs.classifier_args, 'dropout'):
        dropout = configs.classifier_args.dropout
    elif hasattr(configs.model_args, 'classifier_dropout'):
        dropout = configs.model_args.classifier_dropout

    compression_classifier = CompressionClassifier(
        hidden_size=lm.config.hidden_size,
        dropout=dropout
    )
    
    # Wrap both in the probe model (LM is frozen internally)
    model = CompressionProbeModel(
        language_model=lm,
        compression_classifier=compression_classifier
    )

    # Set up LoRA
    # if configs.training_args.resume_from_checkpoint:
    #     with open(os.path.join(configs.training_args.resume_from_checkpoint, "adapter_config.json")) as f:
    #         # Convert to dict
    #         config_params = json.load(f)
    #         config = LoraConfig(**config_params)
    #         if configs.training_args.do_train or (configs.training_args.do_predict and configs.training_args.save_grads):
    #             config.inference_mode = False
    # else:
    #     config = LoraConfig(
    #         r=configs.model_args.lora_r,
    #         lora_alpha=configs.model_args.lora_alpha,
    #         lora_dropout=configs.model_args.lora_dropout,
    #         target_modules=list(configs.model_args.lora_target_modules),
    #         bias="none",
    #         task_type="CAUSAL_LM"
    #     )
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    return model
