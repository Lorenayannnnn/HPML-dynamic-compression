"""
- Tokenizer classes / setup_tokenizer
- functions: tokenize, padding, collate_fn, setup_dataloader...
"""

import random
import numpy as np

def create_tokenizer(configs):
    """creates the tokenizer"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_args.tokenizer_name if configs.model_args.tokenizer_name else configs.model_args.model_name_or_path,
        padding_side=configs.data_args.padding_side,
        cache_dir=configs.data_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize(input_text, configs, tokenizer):
    comp_token_str = configs.model_args.compression_token
    
    # Split CoT by comp token str
    subsequences = input_text.split(comp_token_str)
    
    # Tokenize subsequences and mark the last token of each subsequence whose next token should be <COMP> (positive sample)
    input_ids = []
    next_is_comp = []
    comp_indices = []
    non_comp_indices = []
    for part in subsequences[:-1]:
        subseq_token_ids = tokenizer(part, add_special_tokens=False)['input_ids']
        input_ids.extend(subseq_token_ids)
        next_is_comp.extend([0] * (len(subseq_token_ids) - 1) + [1])
        comp_indices.append(len(input_ids) - 1)
        non_comp_indices.extend(range(len(input_ids) - 1))

    # Randomly select equal number of negative samples
    num_negatives = len(comp_indices)
    assert num_negatives <= len(non_comp_indices), "Not enough non-COMP tokens to sample negatives from."
    negative_sample_indices = list(random.sample(non_comp_indices, num_negatives)) if num_negatives > 0 else []
    
    # Create new next_is_COMP label based on the new positive and negative samples (the rest will be ignored in the loss with -100 index)
    next_is_comp_label = np.array([-100] * len(input_ids))
    next_is_comp_label[comp_indices] = 1
    next_is_comp_label[list(negative_sample_indices)] = 0

    tokenized_entry = {
        'chat_gpt_cot_w_COMP': input_text,
        'chat_gpt_cot_wo_COMP': "".join(subsequences),
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'next_is_COMP': next_is_comp,       # original next_is_COMP (not used in training); keep this just in case
        'labels': next_is_comp_label
    }

    return tokenized_entry

def create_data_collator(configs, tokenizer):
    """creates the data collator"""
    from src.data_module.DataCollator import DataCollator
    data_collator = DataCollator(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=configs.data_args.pad_to_multiple_of,
    )
    return data_collator


def preprocess(configs, raw_datasets):
    """takes in the raw dataset and preprocesses it into the format we want"""

    tokenizer = create_tokenizer(configs)

    # shuffle the dataset
    raw_datasets = raw_datasets.shuffle(seed=configs.training_args.seed)
    tokenized_train_dataset, tokenized_eval_dataset, tokenized_predict_dataset = None, None, None

    if configs.training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if configs.data_args.max_train_samples is not None:
            configs.data_args.max_train_samples = min(configs.data_args.max_train_samples, len(train_dataset))
            train_dataset = train_dataset.select(range(configs.data_args.max_train_samples))
        tokenized_train_dataset = train_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})
        # Print an example of the tokenized dataset
        decoded_text = tokenizer.decode(tokenized_train_dataset[0]['input_ids'])
        print("Example: ", decoded_text)

    if configs.training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if configs.data_args.max_eval_samples is not None:
            configs.data_args.max_eval_samples = min(configs.data_args.max_eval_samples, len(eval_dataset))
            eval_dataset = eval_dataset.select(range(configs.data_args.max_eval_samples))
        tokenized_eval_dataset = eval_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})

    if configs.training_args.do_predict:
        if "predict" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if configs.data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), configs.data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        tokenized_predict_dataset = predict_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})

    return {
        "train": tokenized_train_dataset,
        "validation": tokenized_eval_dataset,
        "predict": tokenized_predict_dataset
    }, tokenizer, create_data_collator(configs, tokenizer)


