# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the multiple choice dataset to parquet format
"""

import os
import datasets
import json
# from verl.utils.hdfs_io import copy, makedirs
import argparse


allow_no_answer = False  # 是否允许没有正确答案的题目
system_msg = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.")

if allow_no_answer:
    medqa_prompt_en = '''You are a medical expert. Please answer the following multiple-choice question from a medical qualification exam. Before selecting the correct answer, you need to provide a detailed explanation for each of the options. At the end of your answer, indicate the final choice by enclosing it in curly braces, for example "{{}}" or "{{B}}" or "{{ABD}}".\n\nQuestion: {question}\n\nOptions:'''
else:
    medqa_prompt_en = '''You are a medical expert. Please answer the following multiple-choice question from a medical qualification exam. Before selecting the correct answer, you need to provide a detailed explanation for each of the options. At the end of your answer, indicate the final choice by enclosing it in curly braces, for example "{{A}}".\n\nQuestion: {question}\n\nOptions:'''


def check_no_answer(data):
    if allow_no_answer:
        any_question_has_no_answer = any(dt['reward_model']['ground_truth'] == [] for dt in data)
        if not any_question_has_no_answer:
            raise ValueError("All data items have answers, but allow_no_answer is set to True. Please check your data.")
    else:
        for dt in data:
            if dt['reward_model']['ground_truth'] == []:
                raise ValueError(f"Data item {dt} has no answer, but allow_no_answer is set to False.")


def medqa_usmle_prompt_template(entry, system_prompt):
    prompt = medqa_prompt_en.format(question=entry['question']) # , option_a=entry['option_list']["A"], option_b=entry['option_list']["B"], option_c=entry['option_list']["C"], option_d=entry['option_list']["D"])
    # 根据选项个数，添加选项内容
    for idx, option in enumerate(entry['options'].values()):
        option_key = chr(ord('A') + idx)
        prompt += f"\n{option_key}: {option}"
        
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data', default='data/original_data/med/dev.jsonl', type=str)
    parser.add_argument('--output_dir', default='data/original_data/med/prepared_parquet', type=str)

    args = parser.parse_args()

    def process_fn(example, idx):
        if 'question' not in example:
            example['question'] = example['statement']
        if 'options' not in example:
            example['options'] = example['option_list']
        if 'answer_idx' not in example:
            example['answer_idx'] = example['answer']
        else:
            example['answer_idx'] = [example['answer_idx']]
        data = {
            "prompt": medqa_usmle_prompt_template(example, system_msg),
            "data_source": "medqa-usmle",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer_idx'],
            },
            "extra_info": {
                'idx': idx,
            }
        }
        return data

    def process_dataset(dataset):
        data = []
        for idx, example in enumerate(dataset):
            if example['answer'] == []:
                print(f"Skipping example {idx} with empty answer")
                print(example)
                continue
            data.append(process_fn(example, idx))
        return data

    ## process and save the dataset to parquet
    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f'{args.input_data} not exists !')
    else:
        with open(args.input_data, 'r') as f:
            train_dataset = [json.loads(line) for line in f]
            #train_dataset = json.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    train_data = process_dataset(train_dataset)
    check_no_answer(train_data)
    print(f"len:{len(train_data)}")
    train_data = datasets.Dataset.from_list(train_data)
    train_data.to_parquet(f'{args.output_dir}/dev_prepared.parquet')