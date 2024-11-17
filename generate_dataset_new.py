import asyncio
import json
import random
from dataclasses import dataclass
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time

def load_token(file_path):
    with open(file_path) as f:
        key = f.read().strip("\n")
    return key
hf_token = load_token(file_path='hf_token.txt')

@dataclass
class Config:
    max_samples: int = 128
    max_new_tokens: int = 800
    temperature: float = 1.0
    constitution_path: str = "data/constitution_education.json"
    dataset_path: str = "data/student_prompts.json"


# Load configuration
config = Config()

# Load the tokenizer and model
if torch.cuda.is_available():
    quantization_config = None#BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None

print('Loading models...')
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="auto", 
    quantization_config=quantization_config,
    token=hf_token
)

print('Loading data...')

# Load constitutional principles
with open(config.constitution_path) as f:
    data = json.load(f)
    constitutions = data["constitutions"]
    system_chat = [item for sublist in data["system_chat"] for item in sublist]
# Select subset for our purposes
system_chat = system_chat[:16]

# Load the dataset
with open(config.dataset_path, "r") as f:
    ds = json.load(f)
    for split in ds:
        ds[split] = ds[split][:config.max_samples]


async def generate_text(prompt, semaphore):
    """Generates text using the model asynchronously."""
    async with semaphore:
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        # Move input tensors to the same device as the model
        device = model.device
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()


async def process_text(split, i, task, semaphore):
    """Processes a single task with critique and revision."""
    chat = system_chat.copy()
    constitution = random.choice(constitutions)
    row = {}

    for prompt, prompt_key, response_key in [
        (task, "init_prompt", "init_response"),
        (constitution["critic"], "critic_prompt", "critic_response"),
        (constitution["revision"], "revision_prompt", "revision_response"),
    ]:
        prompt_dict = {"role": "user", "content": prompt}
        chat.append(prompt_dict)
        completion = await generate_text(tokenizer.apply_chat_template(chat, tokenize=False), semaphore)
        response_dict = {"role": "assistant", "content": completion}
        chat.append(response_dict)
        row[prompt_key] = prompt
        row[response_key] = completion

    return split, i, row


async def main():
    semaphore = asyncio.Semaphore(torch.cuda.device_count() * 10 or 1)
    tasks = [
        process_text(split, idx, row["prompt"], semaphore)
        for split in ds for idx, row in enumerate(ds[split])
    ]
    results = await tqdm_asyncio.gather(*tasks)

    # Collect results into a dictionary
    all_ds = defaultdict(lambda: defaultdict(list))
    for split, i, row in results:
        for key, value in row.items():
            all_ds[split][key].append(value)

    # Convert to dataset and save
    for split, data in all_ds.items():
        df = pd.DataFrame(data)
        df.to_csv(f"{split}_dataset.csv", index=False)

print('Generating datasets...')

start_time = time.time()
asyncio.run(main())
end_time = time.time()
print(f"Data generation completed in {end_time - start_time:.2f} seconds.")
