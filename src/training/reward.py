import re

def compute_acc_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', solution_str)

    if not match:
        return 0
    
    answer = set(re.findall(r'[A-Z]', match.group()))
    gt = set(ground_truth)

    return 1 if answer == gt else 0