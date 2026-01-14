"""
Loading and Preprocessing Datasets
"""
import os

from datasets import load_dataset, concatenate_datasets, DatasetDict


def format_text(example, data_name: str, prompt_only: bool = True):
    """
    Format an example into one text
    """
    PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        )}
    if data_name == 'humaneval':
        example['instruction'] = example['prompt']
    elif data_name == 'mbpp':
        instruction = example['text']
        instruction = instruction + '\nExamples:\n' + '\n'.join(example['test_list'])
        example['instruction'] = instruction
                                         
    if 'input' in example and example["input"]:
        text = PROMPT_DICT["prompt_input"].format_map(example)
    else:
        text = PROMPT_DICT["prompt_no_input"].format_map(example)

    if not prompt_only:
        text += f"{example['output']}"
        
    example['data_name'] = data_name
    example['text'] = text
    return example


def get_formatted_datasets(data_path: str, prompt_only: bool):
    """
    Get formatted datasets
    """
    data_name = os.path.basename(data_path).lower()

    datasets = load_dataset(path=data_path)
    split_0 = list(datasets.keys())[0]
    print(f'Datasets: {datasets}')
    print(f"Example: {datasets[split_0][0]}")

    # Format datasets
    formatted_datasets = datasets.map(
        lambda example: format_text(example, data_name, prompt_only=prompt_only),
        batched=False, load_from_cache_file=False)
    print(f'Formatted datasets: {formatted_datasets}')
    print(f"Formatted example: {formatted_datasets[split_0][0]}")
    print(f"Text example:\n{formatted_datasets[split_0]['text'][0]}")

    return formatted_datasets


if __name__ == '__main__':
    data_path = 'cb'
    _ = get_formatted_datasets(data_path=data_path, prompt_only=False)
