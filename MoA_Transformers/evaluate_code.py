import json
import fire
import os
from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
    
def main(predict_file:str):
    print(predict_file)
    with open(predict_file, 'r') as f:
        lines = f.readlines()
    predicts = [json.loads(line) for line in lines]

    samples = []
    for x in predicts:
        if type(x['response']) == list:
            for response in x['response']:
                sample = {}
                sample['task_id'] = x['task_id']
                sample['completion'] = response
                samples.append(sample)
        else:
            sample = {}
            sample['task_id'] = x['task_id']
            sample['completion'] = x['response']
            samples.append(sample)

    directory = os.path.dirname(predict_file)
    sample_file = os.path.join(directory, 'sample.jsonl')
    write_jsonl(sample_file, samples)

    result = evaluate_functional_correctness(sample_file, k=[1,5,10])
    print(f'result:{result}')
    result['predict_file'] = predict_file
    uppper_directory = os.path.dirname(directory)
    with open(os.path.join(uppper_directory,'score.jsonl'), 'a', encoding='utf-8') as f:
        json_data = json.dumps(result, ensure_ascii=False)
        f.write(json_data+'\n')

if __name__ == "__main__":
    fire.Fire(main)
