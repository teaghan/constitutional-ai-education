from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
import os
import sys
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Optional
from transformers import (
    set_seed,
    HfArgumentParser,
)
from trl import SFTTrainer, SFTConfig

## Training Configuration

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = "/home/obriaint/project/obriaint/gemma-3-4b-it-unsloth-bnb-4bit"
    torch_dtype: Optional[str] = "auto"
    max_seq_length: int = 2048
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0
    lora_bias: str = "none"
    # Model loading configuration
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False

@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file: str = os.path.join(wrk_dir, "data/train_dataset.csv")
    validation_file: str = os.path.join(wrk_dir, "data/val_dataset.csv")
    preprocessing_num_workers: Optional[int] = 4
    chat_template: str = "gemma-3"
    instruction_token: str = "<start_of_turn>user\n"
    response_token: str = "<start_of_turn>model\n"

@dataclass
class TrainingArguments:
    """
    Training configuration using SFTConfig parameters
    """
    output_dir: str = "models"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 30
    learning_rate: float = 2e-4
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    report_to: str = "none"
    dataset_text_field: str = "text"

    def get_sft_config(self):
        """Convert to SFTConfig"""
        return SFTConfig(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            optim=self.optim,
            weight_decay=self.weight_decay,
            lr_scheduler_type=self.lr_scheduler_type,
            seed=self.seed,
            report_to=self.report_to,
            dataset_text_field=self.dataset_text_field,
        )

# Parse arguments
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingConfig))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Load model and tokenizer with Unsloth optimizations
model, tokenizer = FastModel.from_pretrained(
    model_name=model_args.model_name_or_path,
    max_seq_length=model_args.max_seq_length,
    load_in_4bit=model_args.load_in_4bit,
    load_in_8bit=model_args.load_in_8bit,
    full_finetuning=model_args.full_finetuning,
)

## Model Preparation

# Configure tokenizer with appropriate chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template=data_args.chat_template
)

# Add LoRA adapters
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=model_args.lora_dropout,
    bias=model_args.lora_bias,
    random_state=training_args.seed,
)

## Datasets

train_dataset = pd.read_csv(data_args.train_file)
eval_dataset = pd.read_csv(data_args.validation_file)

train_dataset

### Data Pre-processing

def prepare_dataset(df):
    """
    Prepare dataset for SFT by combining init_prompt and revision_response into conversations
    following the exact Gemma-3 chat template format.
    Returns a list of dictionaries with 'text' field containing the formatted conversation.
    """
    conversations = []
    for _, row in df.iterrows():
        # Format exactly like the example, including <bos> token
        conversation = {
            "text": f"<bos><start_of_turn>user\n{row['init_prompt']}<end_of_turn>\n" +
                    f"<start_of_turn>model\n{row['revision_response']}<end_of_turn>\n"
        }
        conversations.append(conversation)
    return conversations

# Prepare datasets for training
train_dataset = prepare_dataset(train_dataset)
eval_dataset = prepare_dataset(eval_dataset)

# Convert to Dataset format
from datasets import Dataset
train_dataset = Dataset.from_list(train_dataset)
eval_dataset = Dataset.from_list(eval_dataset)

train_dataset[0]

## Training Setup

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args.get_sft_config(),
)

# Apply training on completions only
trainer = train_on_responses_only(
    trainer,
    instruction_part=data_args.instruction_token,
    response_part=data_args.response_token,
)

# Training
train_result = trainer.train()
metrics = train_result.metrics

model.save_pretrained("models/gemma-3-4b")
tokenizer.save_pretrained("models/gemma-3-4b")