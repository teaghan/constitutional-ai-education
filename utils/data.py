from typing import Any, List, Literal, Optional
from datasets import DatasetDict, Dataset
import pandas as pd

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def load_token(file_path):
    with open(file_path) as f:
        key = f.read().strip("\n")

def load_dataset_from_csv(csv_files, split_labels):
    """
    Load a Hugging Face Dataset from a list of CSV files, organizing splits and features.
    
    Parameters:
    - csv_files: List of paths to the CSV files.
    - split_labels: List of split labels corresponding to the CSV files (e.g., ['train', 'test']).
    
    Returns:
    - A Hugging Face DatasetDict with the specified splits and features.
    """
    assert len(csv_files) == len(split_labels), "Each CSV file must have a corresponding split label."

    splits_data = {}
    
    for csv_file, split in zip(csv_files, split_labels):
        # Load CSV into a pandas DataFrame
        df = pd.read_csv(csv_file)
        
        # Transform DataFrame into the required structure
        records = []
        for _, row in df.iterrows():
            records.append({
                "prompt": [
                    {"content": row["init_prompt"], "role": "user"}
                ],
                "rejected": [
                    {"content": row["init_prompt"], "role": "user"},
                    {"content": row["init_response"], "role": "assistant"}
                ],
                "chosen": [
                    {"content": row["init_prompt"], "role": "user"},
                    {"content": row["revision_response"], "role": "assistant"}
                ]
            })
            '''
            records.append({
                "prompt": [{"content": row["init_prompt"], "role": "user"}],
                "rejected": [{"content": row["init_response"], "role": "assistant"}],
                "chosen": [{"content": row["revision_response"], "role": "assistant"}]})
            '''
        # Create a Hugging Face Dataset for this split
        splits_data[split] = Dataset.from_pandas(pd.DataFrame(records))
    
    # Combine splits into a DatasetDict
    return DatasetDict(splits_data)

def apply_chat_template(
    example,
    tokenizer,
    auto_insert_empty_system_msg: bool = True):
    
    if all(k in example.keys() for k in ("prompt", "chosen", "rejected")):
        # The inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
        prompt_messages = example["prompt"]
        chosen_messages = example["rejected"][-1:]
        rejected_messages = example["chosen"][-1:]
        
        # Prepend a system message if the first message is not a system message
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(prompt_messages, tokenizer)
        
        example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
        example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
        example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
    else:
        raise ValueError(f"Could not format example as dialogue! Require either the "
                         f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}")

    return example

def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.get_chat_template()

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})