import os
import sys
import json
import time
import logging
import datetime
import subprocess
from tqdm import tqdm, trange
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import codecs
import argparse

random.seed(42)
sys.path.append("..")

# File handling with codecs
codecs_out = lambda x: codecs.open(x, 'w', 'utf-8')
codecs_in = lambda x: codecs.open(x, 'r', 'utf-8')

json_load = lambda x: json.load(codecs.open(x, 'r', 'utf-8'))
json_dump = lambda d, p: json.dump(d, codecs.open(p, 'w', 'utf-8'), indent=2, ensure_ascii=False)
json_dumps = lambda d: json.dumps(d, indent=2, ensure_ascii=False)

from util_agents import LLM_Agent

# Prompt Templates
JEC_polish_prompt = """
I am a university law professor designing final exam questions for my legal course. I want to create a statement analysis question, asking students to determine if the given statement is correct and provide reasoning. To do this, I have selected a question from the multiple-choice question bank and combined it with one of its options, attempting to turn it into a statement for analysis. However, the stem might contain phrasing specific to multiple-choice questions (e.g., "Which of the following is...?" or similar), which may result in awkward or illogical phrasing. Therefore, I need your help to polish the statement by removing such phrases and ensuring that the final version is logically coherent and easy to understand. The polished statement should be enclosed in curly braces.
"""

med_polish_prompt = """
I am a medical school professor preparing final exam questions for my medical course. I would like to design a statement analysis question, where students are required to judge whether the given statement is correct and provide supporting reasoning. To do this, I select a question stem from my multiple-choice question bank and combine it with one of its answer options to transform it into a statement for analysis. However, the stem may still contain phrasing specific to multiple-choice format (e.g., “Which of the following is …?”) or redundant punctuation, and may result in illogical reasoning. Therefore, I need your help to polish the statement: remove any phrasing related to multiple-choice format, adjust sentence structure or add connecting words if necessary, and ensure that the final version is logically coherent and easy to understand. Please enclose the polished statement within curly braces.  
"""

# Function to clean answer text
def clean_ans(ans):
    delete_list = ['"', '[', ']', ',', '\\', '\n', '\t', ' ', '{', '}', '.', '\'']
    for idx in range(len(ans)):
        if ans[idx] not in delete_list:
            ans = ans[idx:]
            break
    for idx in range(len(ans) - 1, -1, -1):
        if ans[idx] not in delete_list:
            ans = ans[:idx + 1]
            break
    return ans


class TaskTransAgentAsync():
    def __init__(self, output_dir=None, data_path=None, data_type='law', do_sample=False, max_workers=8):
        # Removed sensitive API key
        self.agent = LLM_Agent(model='deepseek-chat', max_len=160000, base_url=, api_key= )
        self.output_dir = output_dir
        self.data = json_load(data_path)
        self.dataset_name = os.path.basename(data_path).split('.')[0]
        self.do_sample = do_sample
        self.max_workers = max_workers

        if data_type == 'law':
            self.prompt = JEC_polish_prompt
        else:
            self.prompt = med_polish_prompt

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def build_prompt(self, data_item):
        return self.prompt.format(
            question=data_item['statement_option'].strip(),
        )

    def process_item(self, data_item, max_retries=3):
        attempt = 0
        while attempt < max_retries:
            try:
                prompt = self.build_prompt(data_item)
                response = self.agent.query(prompt)
                return {'id': data_item['id'], 'response': response, "correctness": data_item['correctness'], "original_data": data_item["original_data"]}
            except Exception as e:
                attempt += 1
                wait_time = 2 ** attempt  # Exponential backoff
                logging.warning(f"[Retry {attempt}] Error processing item {data_item['id']}: {e}")
                time.sleep(wait_time)
        logging.error(f"Max retries exceeded for item {data_item['id']}")
        return {'id': data_item['id'], 'response': '', "correctness": data_item['correctness'], "original_data": data_item["original_data"]}

    def run_llm_agent(self):
        output = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_item, item) for item in self.data]

            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
                result = future.result()
                output.append(result)

                # Save output every 100 items
                if i % 100 == 0:
                    json_dump(output, os.path.join(self.output_dir, f'{self.dataset_name}_polished_tmp.json'))

        # Sort output by 'id'
        output.sort(key=lambda x: x['id'])
        json_dump(output, os.path.join(self.output_dir, f'{self.dataset_name}_polished.json'))


def setup_logging(log_dir='/path/to/logs', log_filename='task_trans.log'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8')]
    )


if __name__ == '__main__':
    setup_logging(log_dir='/path/to/logs', log_filename='task_trans.log')

    parser = argparse.ArgumentParser(description='Run TaskTransAgentAsync')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data JSON file')
    parser.add_argument('--data_type', default='law', type=str)
    args = parser.parse_args()

    random.seed(42)
    agent = TaskTransAgentAsync(args.output_dir, args.data_path, data_type=args.data_type)
    agent.run_llm_agent()
