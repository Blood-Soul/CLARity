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


system_msg = (
    "用户和助手之间的对话。用户提出一个问题，由助手来回答。助手首先在脑海中逐步思考推理过程，然后向用户提供答案。推理过程和答案分别用<思考> </思考>和<回答> </回答>标签括起来，即，"
    "<思考> 推理过程 </思考><回答> 答案 </回答>")


allow_no_answer = True  # 是否允许没有正确答案的题目
if allow_no_answer:
    JEC_multi_choice_prompt = '''你是一名法学专家。现在请你解答司法考试中的一道选择题，请你找出所有正确的选项。每道题可能没有正确答案，也可能有一个或者多个正确答案。在解答之前，你需要先针对每个提供的选项给出详细的解释。你需要在回答的最后用大括号圈出给出的答案，例如"{{}}"或者"{{B}}"或者"{{ABD}}"。\n\n问题：{question}\n\n选项：'''
else:
    JEC_multi_choice_prompt = '''你是一名法学专家。现在请你解答司法考试中的一道选择题，请你找出所有正确的选项。每道题可能有一个或者多个正确答案。在解答之前，你需要先针对每个提供的选项给出详细的解释。你需要在回答的最后用大括号圈出给出的答案，例如"{{B}}"或者"{{ABD}}"。\n\n问题：{question}\n\n选项：'''

def check_no_answer(data):
    if allow_no_answer:
        any_question_has_no_answer = any(dt['reward_model']['ground_truth'] == [] for dt in data)
        if not any_question_has_no_answer:
            raise ValueError("All data items have answers, but allow_no_answer is set to True. Please check your data.")
    else:
        for dt in data:
            if dt['reward_model']['ground_truth'] == []:
                raise ValueError(f"Data item {dt} has no answer, but allow_no_answer is set to False.")

def jec_multi_choice_prompt_template(entry, system_prompt):
    prompt = JEC_multi_choice_prompt.format(question=entry['statement'])# , option_a=entry['option_list']["A"], option_b=entry['option_list']["B"], option_c=entry['option_list']["C"], option_d=entry['option_list']["D"])
    # 根据选项个数，添加选项内容
    for idx, option in enumerate(entry['option_list'].values()):
        option_key = chr(ord('A') + idx)
        prompt += f"\n{option_key}: {option}"
        
    return [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='output/parquet', type=str)

    args = parser.parse_args()

    train_dataset = json.load(open('train_random_bs=4_mix_with_original_data_mix_by_difficulty.json'))

    def process_fn(example, idx):
        data = {
            "prompt": jec_multi_choice_prompt_template(example, system_msg),
            "data_source": "jec-qa-1-multi-choice",
            "reward_model": {
                "style": "rule",
                "ground_truth": example['answer'],
            },
            "extra_info": {
                'idx': idx,
                'original_id': example['id'],
            }
        }
        return data

    def process_dataset(dataset):
        data = []
        for idx, example in enumerate(dataset):
            # if example['answer'] == []:
                # print(f"Skipping example {idx} with empty answer")
                # print(example)
                # continue
            data.append(process_fn(example, idx))
        return data

    ## process and save the dataset to parquet
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)
    train_data = process_dataset(train_dataset)
    check_no_answer(train_data)
    print(f"len:{len(train_data)}")
    train_data = datasets.Dataset.from_list(train_data)
    train_data.to_parquet(f'{args.local_dir}/data.parquet')