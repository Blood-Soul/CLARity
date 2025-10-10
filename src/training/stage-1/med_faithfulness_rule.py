import re
#import jieba
# import math
import json

def format_reward(predict):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>.*?<answer>\s*?\{[\u4e00-\u9fffA-Za-z,\s]*\}\s*?</answer>"
    match = re.match(pattern, predict, re.DOTALL | re.MULTILINE) 
    return 1.0 if match else 0.0


def verify_multiple_choice(predict, ground_truth):
    ## Hard Reward
    match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', predict)
    
    if match is None:
        return 0
    ## search for A B C D
    answer = set(re.findall(r'[A-Z]', match.group()))
    if isinstance(ground_truth, list):
        ground_truth = set(ground_truth)
    else:
        ground_truth = set([ground_truth])
    return 1 if answer == ground_truth else 0

def verify_faithfulness(predict, num_choices=5):
    reward_parsing_failed = 0.0
    reward_faithful = 1.0

    
    # 1. 取出思考内容
    m = re.search(r"<think>(.*?)</think>", predict, re.DOTALL)
    if not m:
        return reward_parsing_failed

    # 2. 按行拆分并清理空串
    lines = [ln.strip() for ln in m.group(1).split("\n") if ln.strip()]
    # print(lines)
    
    # 3. 滑窗查找连续 num_choices 行满足条件的窗口。
    for i in range(len(lines) - num_choices + 1):
        window = lines[i:i+num_choices]
        num_parsed_options = 0
        valid_window = True
        
        for idx, item in enumerate(window):
            opt_key = chr(ord('A') + idx)  # A, B, C, D, ...
            opt = re.search(rf"({opt_key})", item)
            # opt = re.search(r"([A-D])", item)
            # res = re.search(r"(不正确|正确|错误)", item)
            # print(f"opt: {opt}, res: {res}")
            if not (opt):
                valid_window = False
                break
            num_parsed_options += 1
        
        # 4. 判断是否四个选项都出现且结构合法
        if valid_window and num_parsed_options == num_choices:
            return reward_faithful
    
    # 没找到合法窗口
    return reward_parsing_failed


def compute_score(solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    """Compute the score of the completion."""
    format_score = format_reward(solution_str)
    # accuracy_score = verify_multiple_choice(solution_str, ground_truth)
    faithfulness_score = verify_faithfulness(solution_str)
    result = {
        "score": format_score + faithfulness_score,
        "format_reward": format_score,
        "faithfulness_reward": faithfulness_score
    }
    
    return result