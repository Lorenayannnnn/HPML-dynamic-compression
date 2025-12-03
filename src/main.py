
import hydra

from src.common_utils import setup
from src.data_module.load_data import load_data_from_hf, load_gsm8k_compressed, setup_dataloader
from src.data_module.preprocessing import preprocess
from src.model_module.load_model import load_model
from src.train_module.train_utils import create_trainer_args, create_trainer


@hydra.main(config_path="configs", config_name="base_configs", version_base=None)
def main(configs):
    print("Loading configuration, setting up output directories...")
    configs = setup(configs)

    """Load the data"""
    # Check if this is GSM8K compressed dataset
    if hasattr(configs.data_args, 'dataset_name') and configs.data_args.dataset_name == "gsm8k_compressed":
        print(f"Loading GSM8K compressed dataset from {configs.data_args.data_path}...")
        raw_datasets = load_gsm8k_compressed(
            configs.data_args.data_path,
            configs.data_args.train_dev_test_split_ratio,
            configs.training_args.seed
        )
    else:
        raw_datasets = load_data_from_hf(
            configs.data_args.dataset_name,
            configs.data_args.train_dev_test_split_ratio,
            configs.training_args.seed,
            cache_dir=configs.data_args.cache_dir
        )

    """Preprocess data"""
    tokenized_datasets, tokenizer, data_collator = preprocess(configs, raw_datasets)
    # data_loaders = setup_dataloader(input_datasets=tokenized_datasets, batch_size=configs.running_args.micro_batch_size, tokenizer=tokenizer)

    """Load model"""
    model = load_model(configs)

    """Set up trainer"""
    trainer_args = create_trainer_args(configs)

    trainer = create_trainer(model, tokenizer, data_collator, tokenized_datasets["train"], tokenized_datasets["validation"], trainer_args)

    if configs.training_args.do_train:
        print("Start training...")
        trainer.train()
        model.save_pretrained(configs.training_args.output_dir)
    elif configs.training_args.do_eval:
        print("Start evaluating...")
        trainer.evaluate()
    elif configs.training_args.do_predict:
        print("Start predicting...")
        trainer.predict()

    print("yay!")


if __name__ == "__main__":
    main()
