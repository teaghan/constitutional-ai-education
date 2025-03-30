#!/bin/bash
module load python/3.12.4
module load cuda/12.2
module load arrow/19.0.1
source /home/obriaint/project/obriaint/const_ai/bin/activate

python /home/obriaint/scratch/constitutional-ai-education/download_model.py /home/obriaint/project/obriaint/gemma-3-4b-it-unsloth-bnb-4bit