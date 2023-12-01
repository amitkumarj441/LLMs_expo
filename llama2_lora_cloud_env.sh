#!/bin/bash
set -ex

# for download llama2 models
mkdir models
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-70b-chat-hf

# for test_lora.py
export AZUREML_OAI_LLAMA_2_70B_HF_PATH="/models/Llama-2-70b-chat-hf"
export AZUREML_OAI_LLAMA_2_70B_HF_PATH="/models/Llama-2-70b-chat-hf"
export AZUREML_OAI_LLAMA_2_70B_HF_PATH="/sys/fs/cgroup/models/Llama-2-70b-chat-hf"
export AZUREML_OAI_DISABLE_BILLING_EVENTS=0
