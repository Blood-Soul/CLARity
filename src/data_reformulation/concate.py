import json
import os

statement_key = 'question' # 'statement'
option_list_key = "options" # 'option_list'
answer_key = "answer_idx"
id_key = None # 'id'

def augment_json(data_dir, output_path):
    if data_dir.endswith('.jsonl'):
        with open(data_dir, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    else:
        with open(data_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)

    new_data = []
    for item_idx, item in enumerate(data):
        statement = item[statement_key].strip()
        options = item[option_list_key]
        if isinstance(item[answer_key], list):
            correct_answers = set(item[answer_key])
        elif isinstance(item[answer_key], str):
            correct_answers = {item[answer_key]}
        if id_key is None:
            item_id = f"medqa_usmle_idx_{item_idx}"
        else:
            item_id = item[id_key]

        # 如果statement最后不是标点符号，需要手动加一个问号
        if statement[-1] not in ['?', '。', '！', '？', '.', '!', '…']:
            statement += '？'

        for opt_key, opt_text in options.items():
            new_item = {
                'id': f"{item_id}_{opt_key}",
                'statement_option': f"{statement}{opt_text}",
                'correctness': opt_key in correct_answers,
                'original_data': item
            }
            new_data.append(new_item)

    print(len(new_data), "items after augmentation")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

# Example usage:
augment_json('original_data/med/dev.jsonl', 'output/concatenation/med/med_usmle_test.json')