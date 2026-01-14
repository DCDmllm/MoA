"""
Testing LLMs on Benchmarks
"""
import argparse
import json
import os
import re
os.environ['HF_HOME']="/data/workspace/cache/huggingface"

import pandas as pd
import torch
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    GenerationConfig,
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from data import get_formatted_datasets
from src import PeftConfig, PeftModelForCausalLM

transformers.set_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser(
        description='Fine-tuning LLMs on training data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@')
    parser.add_argument(
        '--model_path', type=str, default='outputs/llama-2-7b-hf-adamole-the8-commonsense-qa',
        help='huggingface model id or local model path')
    parser.add_argument(
        '--data_path', type=str, default='/data/workspace/projects/moe/datasets/math_commonsense',
        help='huggingface data id or local data path')
    parser.add_argument(
        '--test_datasets', type=str, default=['AddSub','AQuA','gsm8k','MultiArith','SingleEq','SVAMP'],
        help='test datasets')
    parser.add_argument(
        '--max_new_tokens', type=int, default=256,
        help='maximum number of new tokens')
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='batch size in the pipeline')
    # parser.add_argument(
    #     '--logits', default=False, action='store_true',
    #     help='checking choice logits instead of generated texts')

    # Parse arguments
    args = parser.parse_args()
    model_path = args.model_path
    data_path = args.data_path
    model_name = os.path.basename(model_path).lower()
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size

    split = 'test'  

    # Load the configuration and model
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        peft_config.base_model_name_or_path,
        padding_side="left",
    )
    if 'llama' in model_name:
        print('------------setting pad_token_id for llama 3.1')
        tokenizer.pad_token_id = (
            128255 # reserved special token
        ) # llama 3.1 don't have pad_token, pad_token should be different from eos_token
    model = PeftModelForCausalLM.from_pretrained(model=base_model, model_id=model_path)
    model.to(device)
    print(f'Model loaded from {model_path}')
    print(f'Model: {model}')

    for data_name in args.test_datasets:
        data_path_1 = os.path.join(data_path, data_name)
        data_name = data_name.lower()
        # Load and format datasets
        formatted_datasets = get_formatted_datasets(data_path=data_path_1, prompt_only=True)

        # Build the pipeline
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

        # Get the model responses
        responses = []
        for response in tqdm(
            pipeline(
                KeyDataset(formatted_datasets[split], 'text'),
                generation_config=generation_config,
                return_full_text=False,
                batch_size=batch_size,
            ),
            total=len(formatted_datasets[split]),
        ):
            responses.append(response[0]['generated_text'])

        # Print one response
        print(f'Response example:\n{responses[0]}')

        new_data = []
        for i,x in enumerate(formatted_datasets[split]):
            x['response'] = responses[i]
            new_data.append(x)

        new_dataset = Dataset.from_list(new_data)
        new_dataset.to_json(os.path.join(model_path, f'predictions/{data_name}_responses.jsonl'), lines=True)
        # break
    
