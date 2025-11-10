# HPML Course Project: Dynamic Compression

## Structure
```
└── datasets: put the synthetic data here
└── outputs
└── scripts
    └── main.sh: bash script to run the main.py. Override config options here.
└── src
    ├── analysis_module: scripts for analysis, plotting, etc.
    └── configs
        ├── base_configs.yaml: contains the base configurations for the main.py script
        └── ...
    ├── data_module: code for preprocessing, creating tokenizer and data collator
    ├── model_module: model class and initialization
    ├── train_module: code for creating huggingface trainer
    ├── common_utils.py: common utility functions
    └── main.py: main entry point for the project. config is specified via the hydra decorator, which can be overridden from bash
└── .gitignore
└── requirements.txt: list of dependencies
└── ...
```

## Workflow:
1. Entry point: [main.py](src/main.py)
2. Configuration file: [base_configs.yaml](src/configs/base_configs.yaml). Change compression_token under model_args and other configs accordingly.
3. Load data: [load_data.py](src/data_module/load_data.py). I added train, val, and test split in here in case our dataset doesn't have it already. We can also create the split beforehand. Up to you guys.
4. Data preprocessing: Refer to ```def preprocess``` function in [load_data.py](src/data_module/load_data.py), which contains splitting CoT by \<COMP\> token
5. Initialize model:
    - Compression Classifier: [compression_classifier.py](src/model_module/compression_classifier.py); takes in hidden states and labels and calculate BCE loss
    - Wrapper Model (wrap language model and the classifier): [compression_probe_model.py](src/model_module/compression_probe_model.py)
6. Create Trainer: [train_utils.py](src/train_module/train_utils.py)

## Usage
### Install dependencies
```
conda create -n hpml-compress python=3.9
conda activate hpml-compress
pip install -r requirements.txt
```
### Run
- Go to [main.sh](scripts/main.sh) and modify the configurations as needed
- Run:
```
bash main.sh
```

## TODOs:
- Evaluation
- I have not run the code yet, so there might be bugs. Will fix them once running the code.
