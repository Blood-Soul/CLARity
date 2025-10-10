import json
import os
import re
import random
import numpy as np
from typing import List, Tuple

# 题干模板
statement_list = [
    "Which of the following statements are correct:",
    "Please select all correct options from the following:",
    "Here are some statements, which ones are correct:",
    "Which of the following statements is correct:",
    "Which of the following options contains the correct statement:"
]

# 返回：List[List[Tuple[str, bool]]]，每个内层list即为一个batch的选项组
def batch_select_random(options_list: List[Tuple[str, bool, str]], batch_size_candidate: list = [4]):
    output_options_list = []
    while len(options_list) > 0:
        batch_size = random.choice(batch_size_candidate)  # 随机选择一个batch大小
        # print(f"当前处理第 {i} 个batch，剩余选项数：{len(options_list)}")
        batch = []
        idx_list = list(range(len(options_list)))
        sample_idx = random.sample(idx_list, min(batch_size, len(options_list)))
        for idx in sample_idx:
            batch.append(options_list[idx])
        new_option_list = []
        for j in range(len(options_list)):
            if j not in sample_idx:
                new_option_list.append(options_list[j])
        options_list = new_option_list  # 更新选项列表，移除已选项
        output_options_list.append(batch)

    return output_options_list

def select_multi_choice(data):
    selected_data = []
    corr_num = 2
    while len(selected_data) <= 2:
        selected_data.extend([item for item in data if len(item["answer"]) == corr_num])
        # print(len([item for item in data if len(item["answer"]) == corr_num]))
        corr_num += 1
    # print(corr_num)
    return selected_data
        
def fix_all_wrong_questions(data: List[dict]) -> List[dict]:
    # 1. 分类：全错题 & 正确选项 ≥ 2 的题目
    all_wrong = [item for item in data if len(item["answer"]) == 0]
    multi_correct = select_multi_choice(data) # [item for item in data if len(item["answer"]) >= 2]

    print(f"需修复题数（全错）：{len(all_wrong)}，候选替换题数（多对）：{len(multi_correct)}")

    if not multi_correct:
        print("无可用替换题，跳过修复")
        return data

    for wrong_item in all_wrong:
        # 随机选一个错误选项的位置
        wrong_keys = list(wrong_item["option_list"].keys())
        wrong_choice = random.choice(wrong_keys)

        # 从“多对题”中随机找一题
        replacement_item = random.choice(multi_correct)
        correct_keys = replacement_item["answer"]
        replacement_choice = random.choice(correct_keys)

        # 对调文本
        wrong_text = wrong_item["option_list"][wrong_choice]
        right_text = replacement_item["option_list"][replacement_choice]

        wrong_item["option_list"][wrong_choice] = right_text
        replacement_item["option_list"][replacement_choice] = wrong_text

        # 更新 answer 字段（添加新的正确项，删除旧的）
        wrong_item["answer"].append(wrong_choice)
        replacement_item["answer"].remove(replacement_choice)

        # 更新multicorrect列表
        multi_correct = select_multi_choice(data) # [item for item in data if len(item["answer"]) >= 2]
        # if not multi_correct:
            # print("无可用替换题，跳过修复")
        # print(len(multi_correct))

    return data

def clean_option_string(option_str: str) -> str:
    if "{" not in option_str and "}" not in option_str:
        return option_str.strip()
    # 提取第一个 {} 中的内容
    match = re.search(r'\{(.*?)\}', option_str, re.DOTALL)
    if match:
        return match.group(1)
    if '}' in option_str:
        option_str = option_str[:option_str.index('}')]
    elif '}}' in option_str:
        option_str = option_str[:option_str.index('}}')]
    if '{' in option_str:
        option_str = option_str[option_str.index('{') + 1:]
    elif '{{' in option_str:
        option_str = option_str[option_str.index('{{') + 2:]
    return option_str.strip()

def proposition_data_stats(proposition_data):
    # 统计：答案数目为n的data个数 for n in range(0, 9)
    stats = {"all": {n: 0 for n in range(0, 9)},
             "correct": {n: 0 for n in range(0, 9)},
             "false": {n: 0 for n in range(0, 9)}}
    for item in proposition_data:
        answer_count = len(item["answer"])
        stats["all"][answer_count] += 1
        if "correct" in item["statement"]:
            stats["correct"][answer_count] += 1
        else:
            stats["false"][answer_count] += 1
        
    return stats










random.seed(42)

dataset_split = "train"
output_dir = 'output/aggregation/med'
original_choice_num = 5
save_unshuffled_data = False
mix_with_diversify = False

mix_with_original_data = True
mix_by_difficulty = True

polished_data_dir = f'output/dpsk/polish/med/med_usmle_{dataset_split}_polished.json'
with open(polished_data_dir, 'r', encoding='utf-8') as f:
    all_polished_data = json.load(f)
all_polished_data.sort(key=lambda x: x['id'])

# diversify_data_dir = f'output/dpsk/diversify/med/med_usmle_{dataset_split}_polished_filtered_diversified.json'
# with open(diversify_data_dir, 'r', encoding='utf-8') as f:
    # all_diversify_data = json.load(f)
# all_diversify_data.sort(key=lambda x: x['id'])
all_diversify_data = []

def find_all_choices_by_id(data, id):
    options = {}
    correctness = {}
    for item in data:
        if "_".join(item['id'].split('_')[:-1]) == id:
            option_idx = item['id'].split('_')[-1]
            options[option_idx] = item["response"]
            correctness[option_idx] = item["correctness"]
    return options, correctness

if mix_with_diversify:
    assert len(all_polished_data) == len(all_diversify_data), "Polished and diversified data must have the same length."
    all_data = []
    for question_idx in range(0, len(all_polished_data), original_choice_num):
        assert all_diversify_data[question_idx]['id'] == all_polished_data[question_idx]['id'], "IDs must match between polished and diversified data."
        # 随机从4个选项中选两个，用all_diversify_data中的选项。其余用all_polished_data中的选项
        random_num = random.choice([original_choice_num // 2, original_choice_num // 2 + 1])
        # print(random_num)
        use_polished_options = random.sample(range(original_choice_num), random_num)
        for i in range(original_choice_num):
            if i in use_polished_options:
                all_data.append(all_polished_data[question_idx + i])
            else:
                all_data.append(all_diversify_data[question_idx + i])
else:
    all_data = all_polished_data

# 保存一下all_data
if save_unshuffled_data:
    unshuffled_data = []
    for idx in range(0, len(all_data), original_choice_num):
        id = "_".join(all_data[idx]['id'].split('_')[:-1])
        statement = "以下说法正确的是："
        options = {
            chr(65 + i): clean_option_string(all_data[idx + i]["response"]) for i in range(original_choice_num)
        }
        answer = [chr(65 + i) for i in range(original_choice_num) if all_data[idx + i]["correctness"]]
        unshuffled_data.append({
            "statement": statement,
            "option_list": options,
            "id": id,
            "answer": answer,
        })       
    output_all_data_path = os.path.join(output_dir, f'{dataset_split}_all_unshuffled_data.json')
    with open(output_all_data_path, 'w', encoding='utf-8') as f:
        json.dump(unshuffled_data, f, ensure_ascii=False, indent=4)
    exit(0)


if mix_with_original_data:
    def calc_total_length(data):
        total_len = 4*len(data['question']) + sum(len(opt) for opt in data['options'].values())
        return total_len
    
    all_original_data = []
    all_original_data_ids = set()
    for item in all_data:
        item['original_data']['id'] = ('medqa-usmle_' + item['id'].split('_')[-2])
        if ('medqa-usmle_' + item['id'].split('_')[-2]) not in all_original_data_ids:
            all_original_data_ids.add('medqa-usmle_' + item['id'].split('_')[-2])
            all_original_data.append(item['original_data'])

    # print(f"原始数据总量：{len(all_original_data)}")
    if not mix_by_difficulty:
        all_original_data.sort(key=calc_total_length, reverse=True)
        original_data_half = all_original_data[:len(all_original_data)//2]
        original_data_ids = [item['id'] for item in original_data_half]
    else:
        original_data_ids = json.loads(open('original_data/med/easiest_half/medqa_easiest_half_id.json', 'r').read()) # easiest half of data keep as original data
        original_data_half = [item for item in all_original_data if item['id'] not in original_data_ids]
        # print(f"选取的原始数据量：{len(original_data_half)}")
        
    # print(original_data_ids[:10])
    all_data = [data for data in all_data if ('medqa-usmle_' + data['id'].split('_')[-2]) in original_data_ids]
    # print(f"剔除后的增强数据量：{len(all_data)}")
    

batch_size = [5]  # 每个batch的大小
options_list = [(item['response'], item['correctness'], "") for item in all_data]
output_options_list = batch_select_random(options_list, batch_size)

print(len(output_options_list))

# random.shuffle(output_options_list)


# 构造最终输出格式
output_data_second_half_proposition_evolve = []
for idx, options in enumerate(output_options_list):
    new_statement = random.choice(statement_list)
    new_answer = [chr(65 + i) for i, option in enumerate(options) if option[1]]
    new_option_list = {chr(65 + i): clean_option_string(option[0]) for i, option in enumerate(options)}
    new_subjects = list(set([option[2] for option in options if (option[2] and option[2] != "")]))
    
    new_data = {
        "statement": new_statement,
        "option_list": new_option_list,
        "id": f"evolve_{idx}",
        "answer": new_answer,
    }
    output_data_second_half_proposition_evolve.append(new_data)

print(f"生成的题目数：{len(output_data_second_half_proposition_evolve)}")

# output_data_second_half_proposition_evolve = fix_all_wrong_questions(output_data_second_half_proposition_evolve)
# print(f"修复后的题目数：{len(output_data_second_half_proposition_evolve)}")

# 以50%概率将“说法正确的是”改成“说法错误的是”
prob = 0.5
for idx in range(len(output_data_second_half_proposition_evolve)):
    item = output_data_second_half_proposition_evolve[idx]
    if random.random() < prob: # and len(item["answer"]) != len(item["option_list"]):
        item["statement"] = item["statement"].replace("correct", "wrong")
        new_ans_list = []
        for i in range(len(item["option_list"])):
            option_key = chr(65 + i)
            if option_key in item["answer"]:
                continue
            new_ans_list.append(option_key)
        item["answer"] = new_ans_list

print(f"命题数据统计: {proposition_data_stats(output_data_second_half_proposition_evolve)}")

if mix_with_original_data:
    mix_proposition_evolve = output_data_second_half_proposition_evolve + original_data_half
else:
    mix_proposition_evolve = output_data_second_half_proposition_evolve
random.shuffle(mix_proposition_evolve)

with open(os.path.join(output_dir, f'{dataset_split}_random_bs={"".join([str(i) for i in batch_size])}{"_mix_with_original_data" if mix_with_original_data else ""}{"_mix_by_difficulty" if mix_by_difficulty else ""}_random.json'), 'w', encoding='utf-8') as f:
    json.dump(mix_proposition_evolve, f, ensure_ascii=False, indent=4)
# print(f"完成！共生成选择题数：{len(mix_proposition)}")
print(f"完成！共生成选择题数：{len(mix_proposition_evolve)}")