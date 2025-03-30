#!/usr/bin/env python3

import argparse
import os
import sys
from utils.data import load_token
from huggingface_hub import snapshot_download

def download_model(output_dir):
    # Load HF token
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hf_token = load_token(file_path=os.path.join(script_dir, 'hf_token.txt'))

    # Download model files without loading the model
    snapshot_download(
        repo_id="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        token=hf_token,
        local_dir=output_dir,
        cache_dir=os.path.join(script_dir, 'tmp/google_cache')
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_model.py /path/to/save/directory")
        sys.exit(1)
    
    save_dir = sys.argv[1]
    download_model(save_dir)