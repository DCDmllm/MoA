## Installation

```bash
# Navigate to the AdaMoLE directory
cd  MoA_Transformers

# Install required dependencies
pip install -r requirements.txt

# Install custom transformers which Qwen model support MoA
cd transformers

pip install -e .[torch]
```

## Usage

```bash
# Train the model
python train.py @configs/Qwen_softmoa_math14k_train.config

# Test the model
python test.py @configs/Qwen_softmoa_math14k_test.config

# evaluete the model on math datasets
# ...predictions/addsub_responses.jsonl should be full path of addsub response file
python evaluate_math.py --predict_file ...predictions/addsub_responses.jsonl
```

## Citation

If you find AdaMoLE useful in your projects, please consider citing our paper:

Liu, Z., & Luo, J. (2024). AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts. arXiv preprint *arXiv:2405.00361*.

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
