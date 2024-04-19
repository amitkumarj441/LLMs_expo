#!/bin/bash

# Set model size (8B or 70B)
MODEL_SIZE=${1:-"8B"}

# Check if model size is valid
if [ "$MODEL_SIZE" != "8B" ] && [ "$MODEL_SIZE" != "70B" ]; then
  echo "Invalid model size. Choose 8B or 70B."
  exit 1
fi

# Download model and documentation
wget https://llama.meta.com/get-started/download/$MODEL_SIZE -O llama3-$MODEL_SIZE.tar.gz
wget https://llama.meta.com/get-started/docs/

# Extract downloaded files
tar -xf llama3-$MODEL_SIZE.tar.gz
mv docs llama3-$MODEL_SIZE/docs

# Set environment variables (replace with your paths)
export PYTHONPATH=$PWD/llama3-$MODEL_SIZE:$PYTHONPATH
export MODEL_DIR=$PWD/llama3-$MODEL_SIZE

# (Optional) Create virtual environment (recommended)
# python3 -m venv llama3_env
# source llama3_env/bin/activate

echo "Llama 3 model ($MODEL_SIZE) setup complete!"

