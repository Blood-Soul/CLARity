import re
#import jieba
# import math
import json
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import requests
import logging

BASE_URL = ""
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 32

MODEL_NAME = "Qwen2.5-7B-Instruct"

PROMPT_TEMPLATE = """
You are a medical expert. You will be given a multiple‑choice medical‑exam question and a student's line of reasoning. Based on the student's reasoning, identify the final answer selected by the student. Follow the format in the examples below and enclose the answer the student believes is correct in curly braces, e.g. "{{}}", "{{B}}" or "{{ABD}}".

# Example 1:
Question:
A 37-year-old female with a history of type II diabetes mellitus presents to the emergency department complaining of blood in her urine, left-sided flank pain, nausea, and fever. She also states that she has pain with urination. Vital signs include: temperature is 102 deg F (39.4 deg C), blood pressure is 114/82 mmHg, pulse is 96/min, respirations are 18, and oxygen saturation of 97% on room air. On physical examination, the patient appears uncomfortable and has tenderness on the left flank and left costovertebral angle. Which of the following is the next best step in management?\n\nOptions:\nA: Obtain an abdominal CT scan\nB: Obtain blood cultures\nC: Obtain a urine analysis and urine culture\nD: Begin intravenous treatment with ceftazidime\nE: No treatment is necessary
Reasoning:
To determine the next best step in management, we need to consider the patient's symptoms and the most likely diagnosis. The patient presents with hematuria, flank pain, nausea, fever, and dysuria, which are classic symptoms of a urinary tract infection (UTI) or pyelonephritis, especially given her history of type II diabetes mellitus, which is a risk factor for these conditions.\n\nLet's evaluate each option:\n\nA: Obtain an abdominal CT scan - While imaging can be useful, it is not the immediate next step in a patient with suspected UTI or pyelonephritis. It is more appropriate to first confirm the diagnosis and start appropriate treatment.\n\nB: Obtain blood cultures - Blood cultures are typically reserved for patients with sepsis or suspected bacteremia. While the patient has a fever, the primary concern is a UTI, and blood cultures are not the first-line test.\n\nC: Obtain a urine analysis and urine culture - This is the most appropriate initial step. A urine analysis can confirm the presence of blood and white blood cells, and a urine culture can identify the causative organism and guide antibiotic therapy.\n\nD: Begin intravenous treatment with ceftazidime - This is premature without confirming the diagnosis. While the patient has a fever and other signs of infection, starting broad-spectrum antibiotics without a confirmed diagnosis could lead to unnecessary side effects and antibiotic resistance.\n\nE: No treatment is necessary - This is clearly incorrect given the patient's symptoms and fever.
Therefore, the answer chosen by the student is:
{{C}}


# Example 2:
Question:
A 56-year-old woman is brought to the physician by her husband because of a two-day history of fever, malaise, headaches, and confusion. She recently immigrated from rural China and her immunization status is unknown. Her temperature is 39.1°C (102.4°F). Physical examination shows nuchal rigidity and photophobia. Cerebrospinal fluid analysis shows a neutrophil count of 1,500/mm3. Cerebrospinal fluid cultured on Thayer-Martin agar and normal sheep blood agar shows no growth. However, when the sheep blood agar sample is heated, numerous non-hemolytic, opaque, cream-colored colonies grow. Which of the following characteristics best describes the most likely causal pathogen?\n\nOptions:\nA: Gram-negative coccobacillus\nB: Gram-negative, facultative intracellular bacillus\nC: Gram-positive, lancet-shaped diplococcus\nD: Gram-negative diplococcus\nE: Gram-positive, facultative intracellular bacillus
Reasoning:
To determine the most likely causal pathogen, let's analyze the clinical picture and laboratory findings. The patient presents with fever, malaise, headaches, and confusion, and has nuchal rigidity and photophobia, which are suggestive of meningitis. The cerebrospinal fluid (CSF) analysis shows a neutrophil count of 1,500/mm3, which is consistent with bacterial meningitis. The CSF culture on Thayer-Martin agar and normal sheep blood agar shows no growth, but when the sheep blood agar sample is heated, numerous non-hemolytic, opaque, cream-colored colonies grow. This suggests that the pathogen is heat-labile, which is characteristic of Neisseria meningitidis. Neisseria meningitidis is a Gram-negative diplococcus that grows poorly on routine agar but grows well when the agar is heated, a process known as the \"vapor phase\" or \"heat-labile\" growth.
Therefore, the answer chosen by the student is:
{{D}}


# Now, based on the examples above, analyze the following question and the student's reasoning, and state the answer the student believes is correct:
Question:\n{problem}\nReasoning:\n{solution}\nTherefore, the answer chosen by the student is:
""".strip()

def get_response(problem, solution_str):
    prompt = PROMPT_TEMPLATE.format(problem=problem, solution=solution_str)
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            chat_url = f"{BASE_URL}/v1/chat/completions"
            data = {"model": MODEL_NAME, "messages": messages}
            # Log the outgoing request data
            # logging.info(f"Sending request to {chat_url} with data: {json.dumps(data, ensure_ascii=False)}")
            output = requests.post(chat_url, headers=headers, json=data, timeout=60)
            response = output.json()["choices"][0]["message"]["content"]           

            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    return ""

def parse_answer(problem, cot):

    parsed_answer = ""
    for attempt in range(MAX_RETRIES):
        response = get_response(problem, cot)
        match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', response)
    
        if match is None:
            attempt += 1
            continue
        ## search for A B C D
        parsed_answer = set(re.findall(r'[A-Z]', match.group()))

    return parsed_answer

def parse_answer_batch(problems, cots):
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for problem, cot in zip(problems, cots):
            future = executor.submit(parse_answer, problem, cot)
            futures.append(future)

        results = [future.result() for future in futures]

    return results



def verify_faithfulness_batch(predicts, prompts):
    reward_parsing_failed = -1.0
    reward_unfaithful = -1.0
    reward_faithful = 0.0

    rewards = []
    questions_to_parse = []
    cots_to_parse = []
    generated_answers = []
    for predict, prompt in zip(predicts, prompts):

        # 1. 取出思考内容
        try:
            cot = predict.split('<think>')[1].split('</think>')[0].strip()
            if not cot:
                rewards.append(reward_parsing_failed)
                continue
            cot = cot.strip()
            cot_list = cot.split('\n\n')
            if cot_list[-1].strip() == "":
                cot = '\n\n'.join(cot_list[:-2])
            else:
                cot = '\n\n'.join(cot_list[:-1])
        except:
            rewards.append(reward_parsing_failed)
            continue
    
        question = prompt.split("\n\nQuestion:")[1].split('<|im_end|>')[0].strip()

        generated_answer = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', predict.strip())
        if generated_answer is not None:
            questions_to_parse.append(question)
            cots_to_parse.append(cot)
            generated_answer = set(re.findall(r'[A-Z]', generated_answer.group()))
            generated_answers.append(generated_answer)
            rewards.append(-114514)  # Placeholder for later replacement
        else:
            rewards.append(reward_parsing_failed)
            continue

    def replace_elements(a, b):
        b_iter = iter(b)  # 创建一个 b 的迭代器
        for i in range(len(a)):
            if a[i] == -114514:  # 找到值为 -114514 的位置
                a[i] = next(b_iter)  # 用 b 中的元素替换
        return a

    if len(questions_to_parse) != 0:
        parsed_answers = parse_answer_batch(questions_to_parse, cots_to_parse)
        faithfulness_rewards = []
        for parsed_answer, generated_answer in zip(parsed_answers, generated_answers):
            if generated_answer == parsed_answer:   # the cot is faithful when measuring with inference
                faithfulness_rewards.append(reward_faithful)
            else:
                faithfulness_rewards.append(reward_unfaithful)
        rewards = replace_elements(rewards, faithfulness_rewards)

    return rewards
        
        
        
        

def format_reward_batch(predicts):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>.*?<answer>\s*?\{[\u4e00-\u9fffA-Za-z,\s]*\}\s*?</answer>"
    rewards = []
    for predict in predicts:
        match = re.match(pattern, predict, re.DOTALL | re.MULTILINE) 
        reward = (0.0 if match else -1.0)
        rewards.append(reward)

    return rewards


def verify_multiple_choice_batch(predicts, ground_truths):
    rewards = []
    for predict, ground_truth in zip(predicts, ground_truths):
        ## Hard Reward
        match = re.search(r'\{[\u4e00-\u9fffA-Za-z,\s]*\}[^{]*$', predict)
    
        if match is None:
            if ground_truth != []:
                rewards.append(0)
            else:
                rewards.append(1)
            continue
        ## search for A B C D
        answer = set(re.findall(r'[A-Z]', match.group()))
        if isinstance(ground_truth, list):
            ground_truth = set(ground_truth)
        else:
            ground_truth = set([ground_truth])
        reward = (1 if answer == ground_truth else 0)
        rewards.append(reward)
    
    return rewards


def compute_score(solution_strs, ground_truths, **kwargs) -> dict:
    weight_faithfulness = 1.0
    weight_format = 0.1
    weight_acc = 1.0
    predicts = solution_strs
    """Compute the score of the completion."""
    prompts_str = kwargs.get("prompts_str")
    assert prompts_str is not None, "prompt_str must be provided in kwargs"

    format_scores = format_reward_batch(predicts)
    format_scores = [weight_format*format_score for format_score in format_scores]
    accuracy_scores = verify_multiple_choice_batch(predicts, ground_truths)
    accuracy_scores = [weight_acc*accuracy_score for accuracy_score in accuracy_scores]
    faithfulness_scores = verify_faithfulness_batch(predicts, prompts_str)
    faithfulness_scores = [weight_faithfulness*faithfulness_score for faithfulness_score in faithfulness_scores]


    result = [{
        "score": format_score + accuracy_score + faithfulness_score,
        "format_reward": format_score,
        "answer_reward": accuracy_score,
        "faithfulness_reward": faithfulness_score
    } for format_score, accuracy_score, faithfulness_score in zip(format_scores, accuracy_scores, faithfulness_scores)]

    
    return result