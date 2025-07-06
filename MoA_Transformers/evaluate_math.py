import json
import fire
import os
import re
import copy
def extract_answer_number(dataset, sentence: str) -> float:
    dataset = dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''

def math_acc(predict_file):
    test_dataset_l="AddSub AQuA gsm8k MultiArith SingleEq SVAMP"
    result = {}
    for dataset in test_dataset_l.split():
        dataset = dataset.lower()
        save_path = predict_file.replace("addsub", dataset)
        with open(save_path, 'r') as f:
            data_l = f.readlines()
        data_l = [json.loads(one) for one in data_l]
        total = len(data_l)
        correct = 0
        miss = 0.001
        for data in data_l:
            label = data.get('answer')
            flag = False
            if dataset in ['aqua']:
                predict = extract_answer_letter(data.get('response'))
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(dataset, data.get('response'))
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True
            new_data = copy.deepcopy(data)
            new_data['pred'] = predict
            new_data['flag'] = flag

            directory = os.path.dirname(save_path)
            with open(os.path.join(directory, f'{dataset}_predict_checkanswer.jsonl'), 'a', encoding='utf-8') as f:
                json_data = json.dumps(new_data, ensure_ascii=False)
                f.write(json_data+'\n')
        result[dataset]= correct / total
        print(f'{dataset}: accuracy {correct}  {correct / total}')
        
    acc_l = result.values()
    result['average'] = sum(acc_l)/len(acc_l)
    return result
    
def main(predict_file:str):
    print(predict_file)
    result = math_acc(predict_file)
    print(f'acc:{result}')
    result['predict_file'] = predict_file
    directory = os.path.dirname(predict_file)
    directory = os.path.dirname(directory)
    with open(os.path.join(directory,'acc_score.jsonl'), 'a', encoding='utf-8') as f:
        json_data = json.dumps(result, ensure_ascii=False)
        f.write(json_data+'\n')

if __name__ == "__main__":
    fire.Fire(main)
