import os
from transformers.trainer_utils import get_last_checkpoint
from transformers import AutoTokenizer
from utils.data import DEFAULT_CHAT_TEMPLATE

def get_checkpoint(model_path):
    last_checkpoint = None
    if os.path.isdir(model_path):
        last_checkpoint = get_last_checkpoint(model_path)
    return last_checkpoint

def get_tokenizer(model_name, 
                  token,
                  truncation_side='left', 
                  chat_template=None,
                  auto_set_chat_template=True):
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              token=token,
                                              revision='main',
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if truncation_side is not None:
        tokenizer.truncation_side = truncation_side

    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if chat_template is not None:
        tokenizer.chat_template = chat_template
    elif auto_set_chat_template:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    return tokenizer