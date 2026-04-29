# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models

This repository contains the official implementation for the ACL 2026 Main paper **"MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models"**.

MoA builds a heterogeneous mixture over multiple adapter structures for parameter-efficient LLM fine-tuning. The repository provides:

- **Soft MoA**: softly fuses outputs from multiple adapter experts.
- **Sparse MoA**: activates adapter experts sparsely with small performance degradation.
- **Baselines**: LoRA, Prompt Tuning, Parallel Adapter, UniPEFT, HydraLoRA, MoLoRA, AdaMoLE, and Top-K MoE-LoRA variants.
- **Two code paths**:
  - `LLaMA3_*`: implementation based on the native LLaMA 3 code path.
  - `MoA_Transformers/`: implementation based on Hugging Face `transformers`, currently covering Qwen3 and LLaMA-style models.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── datasets/
│   ├── math_14k/
│   ├── commonsense_15k/
│   ├── math_commonsense/
├── LLaMA3_soft_moa/
├── LLaMA3_sparse_moa/
├── LLaMA3_soft_moa_instance/
├── LLaMA3_soft_moe/
├── LLaMA3_sparse_moe/
├── LLaMA3_UniPEFT/
└── MoA_Transformers/
```

## Implementations

### Native LLaMA 3 Implementation

The `LLaMA3_*` directories use the original LLaMA-style checkpoint layout and tokenizer file. They share a similar workflow:

1. `main_finetune.py`: fine-tune adapters.
2. `extract_adapter_from_checkpoint.py`: extract adapter-only weights from a training checkpoint.
3. `example.py`: run generation with the base model plus extracted adapter.
4. `evaluate_math.py` / `evaluate_commonsense.py`: evaluate generated outputs.

Main directories:

| Directory | Purpose |
| --- | --- |
| `LLaMA3_soft_moa/` | Soft MoA over heterogeneous adapter types, including LoRA, parallel adapter, and prompt modules. |
| `LLaMA3_sparse_moa/` | Sparse MoA over heterogeneous adapter types, mainly LoRA and parallel adapter modules. |
| `LLaMA3_soft_moa_instance/` | Instance-level Soft MoA variant. |
| `LLaMA3_soft_moe/` | Soft MoE-style PEFT baselines, including LoRA, HydraLoRA, MoLoRA, prompt, and parallel adapter variants. |
| `LLaMA3_sparse_moe/` | Sparse MoE-LoRA baselines, including AdaMoLE and Top-K MoE-LoRA. |
| `LLaMA3_UniPEFT/` | UniPEFT-style baseline. |

Each directory contains public example scripts in `exps/`.

### Transformers Implementation

`MoA_Transformers/` integrates MoA with a local modified copy of Hugging Face `transformers`.

Key files:

| Path | Purpose |
| --- | --- |
| `MoA_Transformers/train.py` | Fine-tuning entry point. Supports config files via `@configs/...`. |
| `MoA_Transformers/test.py` | Generic testing entry point for choice-style tasks. |
| `MoA_Transformers/test_math.py` | Math benchmark generation/testing entry point. |
| `MoA_Transformers/evaluate_math.py` | Math evaluation script. |
| `MoA_Transformers/evaluate_code.py` | Code evaluation script. |
| `MoA_Transformers/src/` | PEFT wrapper, trainer, config, save/load, and MoA adapter code. |
| `MoA_Transformers/transformers/` | Modified local `transformers` source tree. |
| `MoA_Transformers/configs/` | Public training/testing configs for Qwen3-8B and Qwen3-14B Soft/Sparse MoA on `math_14k`. |

The modified model files are:

- `MoA_Transformers/transformers/src/transformers/models/qwen3/modeling_qwen3.py`
- `MoA_Transformers/transformers/src/transformers/models/llama/modeling_llama.py`

The repository also keeps alternate text copies such as `modeling_qwen3_softmoa.py.txt` and `modeling_llama_softmoa.py.txt`. Rename/copy these over the active `modeling_*.py` files only when you intentionally switch the model implementation variant.

## Installation

### pip

```bash
cd /path/to/moe
pip install -r requirements.txt
```

For the `MoA_Transformers/` implementation:

```bash
cd /path/to/moe/MoA_Transformers
pip install -r requirements.txt
cd transformers
pip install -e ".[torch]"
```

## Data

The repository includes task data under `datasets/`.

Training datasets:

- `datasets/math_14k/`
- `datasets/commonsense_15k/`

Evaluation datasets:

- Math: `datasets/math_commonsense/AddSub`, `AQuA`, `gsm8k`, `MultiArith`, `SingleEq`, `SVAMP`
- Commonsense: `datasets/math_commonsense/boolq`, `piqa`, `social_i_qa`, `hellaswag`, `winogrande`, `ARC-Challenge`, `ARC-Easy`, `openbookqa`

Most training files follow the Alpaca-style JSON format:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

## Running Native LLaMA 3 Experiments

Run commands from the specific `LLaMA3_*` directory.

Example: Soft MoA on `math_14k`.

```bash
cd LLaMA3_soft_moa
export CUDA_VISIBLE_DEVICES=0
bash exps/finetuning_llama3-1_all_math14k_generate_evaluate_seed.sh
```

Example: Sparse MoA on `math_14k`.

```bash
cd LLaMA3_sparse_moa
export CUDA_VISIBLE_DEVICES=0
bash exps/finetuning_llama3-1_all_math14k_generate_evaluate_seed.sh
```

Example: Soft MoA on `commonsense_15k`.

```bash
cd LLaMA3_soft_moa
export CUDA_VISIBLE_DEVICES=0
bash exps/finetuning_llama3-1_all_commonsense15k_generate_evaluate_seed.sh
```

The scripts use a `path` variable, commonly set to `/home`, to construct model, dataset, and output paths:

```bash
path="/home"
```

Before running, edit this variable so it points to your workspace root. The scripts expect paths like:

```text
${path}/pretrain_models/Meta-Llama-3.1-8B-Instruct/
${path}/datasets/math_14k/train.json
${path}/datasets/math_commonsense/AddSub/test.json
${path}/outputs/...
```

### Native Workflow Details

The public scripts perform the following steps:

1. Train:

```bash
torchrun --nproc_per_node ${num_devices} main_finetune.py \
  --llama_path Meta-Llama-3.1-8B-Instruct/ \
  --data_path datasets/math_14k/train.json \
  --output_dir outputs/run_name/
```

2. Extract adapter weights:

```bash
python extract_adapter_from_checkpoint.py \
  --checkpoint outputs/run_name/checkpoint-1.pth
```

This writes adapter files such as:

```text
adapter.pth
adapter_params.json
```

3. Generate predictions:

```bash
torchrun --nproc_per_node ${num_devices} example.py \
  --ckpt_dir Meta-Llama-3.1-8B-Instruct/ \
  --adapter_path outputs/run_name/adapter.pth \
  --data_path datasets/math_commonsense/AddSub/test.json \
  --save_path outputs/run_name/AddSub_predict_mingen120.jsonl \
  --max_gen_len 200 \
  --min_gen_len 120
```

4. Evaluate:

```bash
python evaluate_math.py \
  --predict_file outputs/run_name/AddSub_predict_mingen120.jsonl
```

For commonsense tasks:

```bash
python evaluate_commonsense.py \
  --predict_file outputs/run_name/boolq_predict_mingen10.jsonl
```

## Running Baselines

The baseline directories use the same train/extract/generate/evaluate workflow.

Examples:

```bash
cd LLaMA3_soft_moe
export CUDA_VISIBLE_DEVICES=0
bash exps/finetuning_llama3-1_lora_math14k_generate_evaluate_seed.sh
```

```bash
cd LLaMA3_soft_moe
export CUDA_VISIBLE_DEVICES=0
bash exps/finetuning_llama3-1_hydralora_math14k_generate_evaluate_seed.sh
```

```bash
cd LLaMA3_sparse_moe
export CUDA_VISIBLE_DEVICES=0
bash exps/finetuning_llama3-1_adamole_lora_math14k_generate_evaluate_seed.sh
```

```bash
cd LLaMA3_sparse_moe
export CUDA_VISIBLE_DEVICES=0
bash exps/finetuning_llama3-1_topkmoe_lora_math14k_generate_evaluate_seed.sh
```

## Running MoA_Transformers

Install the local modified `transformers` first:

```bash
cd MoA_Transformers
pip install -r requirements.txt
cd transformers
pip install -e ".[torch]"
cd ..
```

Train Sparse MoA with Qwen3-8B on `math_14k`:

```bash
python train.py @configs/qwen3-8b_sparsemoa_math14k_train.config
```

Train Soft MoA with Qwen3-8B on `math_14k`:

```bash
python train.py @configs/qwen3-8b_softmoa_math14k_train.config
```

Test a trained Sparse MoA model:

```bash
python test.py @configs/qwen3-8b_sparsemoa_math14k_test.config
```

Evaluate math predictions:

```bash
python evaluate_math.py --predict_file /path/to/predictions/addsub_responses.jsonl
```

### Config Files

Config files in `MoA_Transformers/configs/` are passed with Python argparse's `fromfile_prefix_chars='@'` syntax. For example:

```text
--model_path=Qwen/Qwen3-8B
--data_path=/path/to/moe/datasets/math_14k
--peft_type=sparsemoa
--lora_rank=8
--target_modules
q_proj
k_proj
v_proj
o_proj
down_proj
--max_length=300
--batch_size=8
--gradient_accumulation_steps=2
--num_train_epochs=1
--learning_rate=1e-4
--lr_scheduler_type=constant_with_warmup
--warmup_steps=200
--weight_decay=0.0
```

Available public configs cover:

- `qwen3-8b_softmoa_math14k_*`
- `qwen3-8b_sparsemoa_math14k_*`
- `qwen3-14b_softmoa_math14k_*`
- `qwen3-14b_sparsemoa_math14k_*`
- seed variants with `seed125` and `seed1225`

By default, `train.py` saves outputs under:

```text
MoA_Transformers/outputs/<model>-<peft_type>-<dataset>/
```

For example:

```text
outputs/qwen3-8b-sparsemoa-math-14k/
```

## Important Hyperparameters

Common native LLaMA 3 arguments:

| Argument | Meaning |
| --- | --- |
| `--lora_layers` | Layer range for LoRA modules, e.g. `0-32`. |
| `--lora_rank` | LoRA rank. |
| `--lora_targets` | Target modules, e.g. `Q,K,V,O,FFN_DOWN`. |
| `--lora_alpha` | LoRA scaling alpha. |
| `--p_adapter_layers` | Layer range for parallel adapters. |
| `--p_adapter_size` | Hidden size of parallel adapters. |
| `--prompt_layers` | Layer range for prompt modules. |
| `--prompt_len` | Prompt length. |
| `--expert_num` | Number of experts for MoE-style baselines. |
| `--swi_x` | Router hidden-size multiplier for SwiGLU router; `0` uses a linear router. |
| `--max_threshold` | Sparse MoA activation threshold. |
| `--batch_size` | Per-GPU batch size. |
| `--accum_iter` | Gradient accumulation steps. |
| `--bf16` | Use bfloat16. |
| `--flash_attention2` | Enable FlashAttention 2 when available. |

Common `MoA_Transformers` arguments:

| Argument | Meaning |
| --- | --- |
| `--model_path` | HF model id or local HF model path. |
| `--data_path` | HF dataset id or local dataset directory. |
| `--peft_type` | `softmoa` or `sparsemoa`. |
| `--target_modules` | Transformer module names to receive adapters. |
| `--lora_rank` | Adapter rank. |
| `--max_length` | Maximum tokenized sequence length. |
| `--batch_size` | Per-device batch size. |
| `--gradient_accumulation_steps` | Gradient accumulation steps. |
| `--num_train_epochs` | Number of training epochs. |
| `--learning_rate` | Training learning rate. |
| `--seed` | Random seed. |

## Citation

If you find MoA useful in your projects, please consider citing our paper:

```bibtex
@article{cao2025moa,
  title={MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models},
  author={Cao, Jie and Lin, Tianwei and He, Hongyang and Yan, Rolan and Zhang, Wenqiao and Li, Juncheng and Zhang, Dongping and Tang, Siliang and Zhuang, Yueting},
  journal={arXiv preprint arXiv:2506.05928},
  year={2025}
}
```
