#!/bin/bash
# Install base packages with dependencies
echo "Installing base packages..."
pip install torch==2.5.1 torchvision transformers tqdm pandas triton==3.1.0 sentencepiece protobuf datasets huggingface_hub hf_transfer

# Install packages without dependencies
echo "Installing packages without dependencies..."
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl
pip install --no-deps cut_cross_entropy unsloth_zoo
pip install --no-deps unsloth

echo "Installation complete!"