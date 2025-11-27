import re
import argparse
import os
from datasets import Dataset

def cal_acc(predict, ground_truth):
    match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', predict)

    if match is None:
        return 0
    
    answer = set(re.findall(r'[A-Z]', match.group()))
    gt = set(ground_truth)

    return 1 if answer == gt else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data', default='data/infer_answer/JEC_1_multi_choice_test_inferenced.parquet', type=str)
    
    args = parser.parse_args()

    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f'{args.input_data} not exists !')
    
    dataset = Dataset.from_parquet(args.input_data)

    total = 0
    correct = 0

    for i, example in enumerate(dataset):
        ground_truth = example["reward_model"]["ground_truth"]
        predict = example["response"]["answer"]

        total += cal_acc(predict, ground_truth)
        total += 1

    print(f'ACC: {correct/total}')