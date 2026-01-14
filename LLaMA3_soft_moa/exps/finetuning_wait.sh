#!/bin/bash
# export TIKTOKEN_CACHE_DIR="/mnt/caojie/caojie/cache"
# Loop indefinitely
# gpu memory need
memory=45
while true; do
    echo "Checking at $(date)"  # Print the current time
    
    # Initialize variable to collect eligible GPU IDs
    eligible_gpus=""

    # Get GPU info
    while IFS=, read -r gpu_id total_mem used_mem; do
        unused_mem_gb=$(echo "scale=2; ($total_mem - $used_mem) / 1024" | bc)
        if (( $(echo "$unused_mem_gb > $memory" | bc -l) )); then
            eligible_gpus="$eligible_gpus$gpu_id,"
            echo "GPU ID $gpu_id has more than $memory GB unused memory: $unused_mem_gb GB"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits)

    # Remove trailing comma
    eligible_gpus=${eligible_gpus%,}

    # Check if there are any eligible GPUs
    if [ -n "$eligible_gpus" ]; then
        export CUDA_VISIBLE_DEVICES="$eligible_gpus"
        echo "Running command on GPUs: $CUDA_VISIBLE_DEVICES"
        # bash ./exps/finetuning_llama3-1_math14k_generate_evaluate.sh
        # bash ./exps/finetuning_llama3-1_commonsense15k_generate_evaluate.sh
        # bash ./exps/finetuning_llama3-1_all_commonsense15k_generate_evaluate_seedG.sh
        # bash ./exps/finetuning_llama3-1_all_commonsense15k_generate_evaluate_seedG1.sh
        # bash ./exps/finetuning_llama3-1_all_commonsense15k_generate_evaluate_seedG2.sh
        # bash ./exps/finetuning_llama3-1_all_math14k_generate_evaluate_seedG.sh
        bash ./exps/finetuning_llama3-1_all_math14k_generate_evaluate_seedG1.sh
        bash ./exps/finetuning_llama3-1_all_math14k_generate_evaluate_seedG2.sh
        # bash ./exps/finetuning_llama3-1_all_math14k_generate_evaluate.sh
        # bash ./exps/finetuning_llama3-1_commonsense170k_generate_evaluate.sh
        # bash ./exps/finetuning_llama3-1_commonsense170k_generate_evaluate2.sh
        # bash ./exps/finetuning_llama3-1_commonsense170k_generate_evaluate3.sh
        # bash ./exps/finetuning_llama3-1_math14k_generate_evaluate1.sh
        # bash ./exps/finetuning_llama3-1_math14k_generate_evaluate2.sh
        # bash ./exps/finetuning_llama3-1_math14k_generate_evaluate3.sh
        # bash ./exps/finetuning_llama3-1_math14k_generate_evaluate4.sh
        # bash ./exps/finetuning_llama3-1_math14k_generate_evaluate5.sh
        # bash ./exps/finetuning_llama3-1_commonsense15k_generate_evaluate1.sh
        break
    else
        echo "No GPU with more than $memory GB unused memory."
    fi

    # Wait for 5 seconds before the next check
    sleep 20 
done