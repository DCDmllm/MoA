# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models

<!-- [![arXiv](https://img.shields.io/badge/arXiv-xxxx.xxxxxx-b31b1b.svg)](https://arxiv.org/abs/xxxx.xxxxxx) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) **[Paper Link (if published)]** | **[Project Demo (if applicable)]** | **[Hugging Face Model (if applicable)]** -->

This is the official PyTorch implementation for the paper "**MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models**".

Our work introduces MoA (Mixture of Adapters), a novel approach that leverages a heterogeneous mixture of adapters to achieve superior parameter efficiency and performance when fine-tuning large language models.

## Key Features
* **Training enabled on 1 *  24G GPU for LLaMA-3.1-8B.**: 
* **Heterogeneous Soft MoA & Sparse MoA**: 
Two variants of MoA,
Soft MoA achieves fine-grained integration by performing a weighted fusion of all expert outputs;
Sparse MoA activates adapter experts sparsely based on their contribution, achieving this with negligible performance degradation.
* **Soft-weighted MoE-LoRA baselines**:  HydraLoRA, MoLoRA
* **Sparse MoE-LoRA baselines**: AdaMoLE, MoE-LoRA(TOP-K)
* **Instance-level baselines**: Instance MoA, UniPEFT(LoRA, Prompt, Parallel adatper)
* **Flash-attention2 supported**.


## TODO

- [x] Release Soft MoA code.
- [x] Release Sparse MoA code.
- [x] Release Soft MoE-LoRA baselines (LoRA, HydraLoRA, MoLoRA)
- [x] Release Sparse MoE-LoRA baselines (AdaMoLE, MoE-LoRA(TOP-K))
- [x] Release Instance-level Baselines.
- [x] Qwen3 series & Llama series support: Now integrated based on the `transformers` library.

## `transformers` Library Support

We provide a version of MoA that is fully compatible with the Hugging Face `transformers` library, allowing for seamless integration with a wider range of models.

The code supporting **Qwen3 series** and **LLaMA series** base models based on the `transformers` library is located in the:
* **`MoA_Transformers/`**

This implementation allows you to leverage the standard `AutoModelForCausalLM` interface for loading and fine-tuning.

## Installation
**Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

## Usage

### Prepare Pre-trained model
MoA utilizes [Meta version](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main/original) of `LLaMA3.1-8B` as the base model.


### Train,Inference,Evaluation
**An examples for training, evaluation, and inference on Math dataset.**
    ```

    # export CUDA_VISIBLE_DEVICES="0,1"
    # Count the number of devices
    num_devices=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

    echo "Number of devices: $num_devices"

    max_devices=1

    if [ "$num_devices" -gt "$max_devices" ]; then
        num_devices=$max_devices
        echo "max of devices: $max_devices"
    fi

    # config
    epochs=2
    dataset="math_14k"
    max_seq_len=300
    min_gen_len=120
    max_gen_len=200

    lora_layers="0-32"
    lora_rank=8
    lora_targets="Q,K,V,O,FFN_DOWN"
    lora_alpha=8
    hydra_moe=True # hydra lora, Asymmetric LoRA
    expert_num=1

    p_adapter_layers="0-32"
    p_adapter_size=16
    p_adapter_hydra=True

    prompt_layers="0-32"
    prompt_len=10

    swi_x=4

    blr=6e-3
    flash_attention2=False
    bf16=True
    tag=""
    batch_size_gpu=8
    eff_batch_size=32
    path="/home/"
    output_dir="${path}/outputs/softmoa/${dataset}/b${eff_batch_size}_epoch${epochs}_warme1_loralayers${lora_layers}_lorar${lora_rank}_lora${lora_targets}_alpha${lora_alpha}_expertnum${expert_num}_hydra${hydra_moe}_padapter_layers${p_adapter_layers}_padaptersize${p_adapter_size}_padapterhydra${p_adapter_hydra}_prompt_layers${prompt_layers}_prompt_len${prompt_len}_swi_x${swi_x}_blr${blr}_maxseq${max_seq_len}_flashatt2${flash_attention2}_bf16${bf16}_${tag}/"

    # Train
    torchrun --nproc_per_node $num_devices --master_port=3038 main_finetune.py \
        --llama_path ${path}/pretrain_models/Meta-Llama-3.1-8B-Instruct/ \
        --data_path ${path}/datasets/${dataset}/train.json \
        --expert_num $expert_num \
        --lora_layers $lora_layers \
        --lora_rank ${lora_rank} \
        --lora_targets $lora_targets \
        --lora_alpha $lora_alpha \
        --hydra_moe $hydra_moe \
        --seed 0 \
        --p_adapter_layers $p_adapter_layers \
        --p_adapter_size $p_adapter_size \
        --p_adapter_hydra $p_adapter_hydra \
        --prompt_layers $prompt_layers\
        --prompt_len $prompt_len \
        --swi_x $swi_x \
        --max_seq_len $max_seq_len \
        --batch_size  $batch_size_gpu \
        --accum_iter $(($eff_batch_size/$num_devices/$batch_size_gpu)) \
        --epochs ${epochs} \
        --warmup_epochs 1 \
        --blr ${blr} \
        --flash_attention2 $flash_attention2 \
        --bf16 $bf16 \
        --weight_decay 0.02 \
        --output_dir $output_dir \
        --num_workers 10

    checkpoint="${output_dir}checkpoint-$((epochs-1)).pth"
    # get lora parameters
    python extract_adapter_from_checkpoint.py --checkpoint $checkpoint

    adapter_path="${output_dir}adapter.pth"

    # Inference
    test_dataset_l="AddSub AQuA gsm8k MultiArith SingleEq SVAMP"

    for test_dataset in $test_dataset_l
    do
    save_path="${output_dir}${test_dataset}_predict_mingen${min_gen_len}.jsonl"
    torchrun --nproc_per_node $num_devices --master_port=3038 example.py \
        --ckpt_dir ${path}/pretrain_models/Meta-Llama-3.1-8B-Instruct/ \
        --adapter_path $adapter_path \
        --data_path ${path}/datasets/math_commonsense/${test_dataset}/test.json \
        --save_path $save_path \
        --max_gen_len $max_gen_len \
        --min_gen_len $min_gen_len \
        --max_batch_size 200 \
        --temperature 0.1 \
        --top_p 0.75
    done

    # Evaluation
    save_path1="${output_dir}AddSub_predict_mingen${min_gen_len}.jsonl"
    python evaluate_math.py --predict_file $save_path1
    ```