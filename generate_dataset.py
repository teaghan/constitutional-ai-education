import json
import random
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
import time
import os
import tqdm
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# Get working directory for file paths
wrk_dir = os.path.dirname(os.path.abspath(__file__))

@dataclass
class Config:
    """Configuration settings for dataset generation and model parameters."""
    max_samples: int = 5000
    max_new_tokens: int = 1200
    # Gemma-3 recommended inference settings
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0  # 1.0 means disabled
    constitution_path: str = os.path.join(wrk_dir, "data/constitution_education.json")
    dataset_path: str = os.path.join(wrk_dir, "data/student_prompts.json")
    # Unsloth's optimized 4-bit version
    model_name: str = "/home/obriaint/project/obriaint/gemma-3-4b-it-unsloth-bnb-4bit"

def generate_text(
    prompt: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None,
    model: FastModel = None,
    tokenizer = None,
    config: Config = None
) -> str:
    """
    Generates text using the Gemma-3 model.

    Args:
        prompt: Single prompt for text generation
        message_history: List of message dictionaries containing conversation history
        model: The loaded FastModel instance
        tokenizer: The configured tokenizer
        config: Configuration settings

    Returns:
        str: Generated text response

    Raises:
        AssertionError: If inputs are invalid or missing required components
    """
    # Input validation
    assert not (prompt is None and message_history is None), \
        "Either prompt or message_history must be provided"
    assert not (prompt is not None and message_history is not None), \
        "Cannot provide both prompt and message_history"
    assert model is not None, "Model must be provided"
    assert tokenizer is not None, "Tokenizer must be provided"
    assert config is not None, "Config must be provided"

    if prompt is not None:
        assert isinstance(prompt, str), "prompt must be a string"

    if message_history is not None:
        assert isinstance(message_history, list), "message_history must be a list"
        assert len(message_history) > 0, "message_history cannot be empty"
        assert all(isinstance(msg, dict) for msg in message_history), \
            "all messages in message_history must be dictionaries"
        assert all("role" in msg and "content" in msg for msg in message_history), \
            "each message must contain 'role' and 'content' keys"
        assert all(isinstance(msg["role"], str) and isinstance(msg["content"], str)
                  for msg in message_history), \
            "message role and content must be strings"

    try:
        # Format messages for Gemma-3
        if prompt is not None:
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }]
        else:
            messages = [{
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}]
            } for msg in message_history]

        # Apply chat template with generation prompt
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        # Tokenize and move to device
        inputs = tokenizer([formatted_prompt], return_tensors="pt").to(model.device)

        # Generate text with Gemma-3 recommended parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            min_p=config.min_p,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
        )

        # Decode and clean response
        response = tokenizer.batch_decode(outputs)[0]
        response = response.split('<start_of_turn>model\n')[-1].strip()

        if response.endswith('<end_of_turn>'):
            response = response[:-len('<end_of_turn>')].strip()

        return response

    except Exception as e:
        print(f"Error in generate_text: {e}")
        raise

def create_sample(
    split: str,
    i: int,
    task: str,
    model: FastModel,
    tokenizer,
    config: Config,
    constitutions: List[Dict[str, str]],
    system_chat: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, int, Dict[str, str]]:
    """
    Process a single task with critique and revision using constitutional principles.

    Args:
        split: Dataset split (train/val/test)
        i: Index of the task
        task: The initial student query
        model: The loaded FastModel instance
        tokenizer: The configured tokenizer
        config: Configuration settings
        constitutions: List of constitutional principles
        system_chat: List of system messages

    Returns:
        tuple: (split, index, dictionary containing prompts and responses)

    Raises:
        AssertionError: If inputs are invalid or missing required components
    """
    # Input validation
    assert isinstance(split, str), "split must be a string"
    assert isinstance(i, int), "i must be an integer"
    assert isinstance(task, str), "task must be a string"
    assert task.strip(), "task cannot be empty"
    assert model is not None, "Model must be provided"
    assert tokenizer is not None, "Tokenizer must be provided"

    # Validate system_chat structure
    if system_chat is not None:
        assert isinstance(system_chat, list) and len(system_chat) > 0, \
            "system_chat must be a non-empty list"
        assert all(
            isinstance(msg, dict) and
            "role" in msg and
            "content" in msg and
            isinstance(msg["role"], str) and
            isinstance(msg["content"], str)
            for msg in system_chat
        ), "system_chat messages must be dictionaries with 'role' and 'content' string fields"

    # Initialize chat history with system messages
    chat_history = []
    if system_chat:
        chat_history.extend([
            {"role": msg["role"], "content": msg["content"]}
            for msg in system_chat
        ])

    # Select a random constitutional principle
    constitution = random.choice(constitutions)
    row = {}

    # Go through initial response, critique, and revision phases
    phases = [
        (task, "init_prompt", "init_response"),
        (constitution["critic"], "critic_prompt", "critic_response"),
        (constitution["revision"], "revision_prompt", "revision_response"),
    ]

    for prompt, prompt_key, response_key in phases:
        # Validate prompt for each phase
        assert isinstance(prompt, str) and prompt.strip(), \
            f"Invalid prompt for {prompt_key}"

        prompt_suffix = ''
        if 'revision' in prompt_key:
            prompt_suffix = '\n\n Note: Only include the revised response to the student\'s initial query - DO NOT INCLUDE ANY OTHER TEXT BEFORE OR AFTER THE REVISED RESPONSE. Write your response as if you are addressing the initial query.'
        elif 'critic' in prompt_key:
            prompt_suffix = '\n\n Note: Only include the critique of the previous response - DO NOT INCLUDE ANY PROPOSED REVISIONS.'
        else:
            prompt_suffix = ''
        # Add the current prompt to chat history
        chat_history.append({
            "role": "user",
            "content": prompt + prompt_suffix
        })

        # Generate response using the full chat history
        completion = generate_text(
            message_history=chat_history,
            model=model,
            tokenizer=tokenizer,
            config=config
        )

        # Validate completion
        assert isinstance(completion, str) and completion.strip(), \
            f"Invalid completion received for {response_key}"

        # Add response to conversation history
        chat_history.append({
            "role": "assistant",
            "content": completion
        })

        # Store prompt and response
        row[prompt_key] = prompt
        row[response_key] = completion

    return split, i, row

def main():
    """Main function to process all tasks and generate the dataset."""
    try:
        # Load configuration
        config = Config()
        
        print('Loading models...')
        # Initialize model and tokenizer with Unsloth
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
            full_finetuning=False,
        )

        # Configure tokenizer with Gemma-3 chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3"
        )

        # Load constitutional principles and example conversations
        with open(config.constitution_path) as f:
            data = json.load(f)
            constitutions = data["constitutions"]
            #system_chat = [item for sublist in data["system_chat"] for item in sublist]
            # Select subset of system chat messages
            #system_chat = system_chat[:18]

        # Load Training and Test Prompts
        with open(config.dataset_path, "r") as f:
            ds = json.load(f)
            for split in ds:
                ds[split] = ds[split][:config.max_samples]

        # Validate dataset structure
        assert isinstance(ds, dict) and len(ds) > 0, "Dataset must be a non-empty dictionary"
        for split, data in ds.items():
            assert isinstance(split, str), f"Split name must be string, got {type(split)}"
            assert isinstance(data, list), f"Split {split} must contain a list of samples"
            assert all(isinstance(row, dict) and "prompt" in row for row in data), \
                f"Each row in {split} must be a dictionary containing 'prompt'"

        all_ds = defaultdict(lambda: defaultdict(list))
        
        # Process each split
        for split in ds:
            print(f"\nProcessing {split} split...")
            for idx, row in tqdm.tqdm(enumerate(ds[split]), total=len(ds[split])):
                split, i, result = create_sample(
                    split, idx, row["prompt"],
                    model, tokenizer, config,
                    constitutions)
                for key, value in result.items():
                    all_ds[split][key].append(value)

        # Validate and save results to CSV files
        for split, data in all_ds.items():
            df = pd.DataFrame(data)

            # Validate DataFrame structure
            expected_columns = {
                "init_prompt", "init_response",
                "critic_prompt", "critic_response",
                "revision_prompt", "revision_response"
            }
            assert set(df.columns) == expected_columns, \
                f"Missing columns in {split} dataset. Expected: {expected_columns}"

            # Ensure no empty values
            assert not df.isna().any().any(), f"Found null values in {split} dataset"

            # Save to CSV
            output_path = f"data/{split}_dataset.csv"
            df.to_csv(output_path, index=False)
            print(f"Saved {len(df)} samples to {output_path}")

    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Data generation completed in {end_time - start_time:.2f} seconds.")