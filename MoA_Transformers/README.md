# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models

## Installation

```bash
# Navigate to the MoA directory
cd  MoA_Transformers

# Install required dependencies
pip install -r requirements.txt

# Install custom transformers which Qwen/LLama model support MoA
cd transformers

pip install -e .[torch]
```

## Usage

```bash
# Train the model
python train.py @configs/qwen3-8b_sparsemoa_math14k_train.config

# Test the model
python test.py @configs/qwen3-8b_sparsemoa_math14k_test.config

# evaluete the model on math datasets
# ...predictions/addsub_responses.jsonl should be full path of addsub response file
python evaluate_math.py --predict_file ...predictions/addsub_responses.jsonl
```

## Modified model file in Transformers
### llama:
Default to support sparsemoa
```transformers/src/transformers/models/llama/modeling_llama.py```

### qwen3:
Default to support sparse moa
```transformers/src/transformers/models/qwen3/modeling_qwen3.py```

Rename the "modeling_llama_softmoa.py.txt" to "modeling_llama.py" to support softmoa.

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

## Ackhnowledgement
This repo benefits from AdaMoLE. Thanks for their wonderful works.
