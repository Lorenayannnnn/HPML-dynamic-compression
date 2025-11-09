

def load_model(configs):
    from transformers import AutoModel
    from src.model_module.compression_classifier import CompressionClassifier
    from src.model_module.compression_probe_model import CompressionProbeModel

    """main function for loading the model_module"""
    lm = AutoModel.from_pretrained(
        configs.training_args.resume_from_checkpoint if configs.training_args.resume_from_checkpoint else configs.model_args.model_name_or_path,
        cache_dir=configs.data_args.cache_dir,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype="auto"
    )

    compression_classifier = CompressionClassifier(
        hidden_size=lm.config.hidden_size, 
        dropout=configs.model_args.classifier_dropout if hasattr(configs.model_args, 'classifier_dropout') else 0.1
    )
    
    # Wrap both in the probe model
    model = CompressionProbeModel(
        language_model=lm,
        compression_classifier=compression_classifier,
        freeze_lm=True  # Keep LM frozen, only train classifier
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
