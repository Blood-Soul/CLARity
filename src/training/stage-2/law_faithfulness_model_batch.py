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
你是一名法学专家。现在给你一道司法考试中的选择题，以及一位考生的思考过程。请根据考生的思考过程，指出该考生给出的最终答案。请模仿以下示例，使用大括号圈出答案，例如“{{}}”或“{{B}}”或“{{ABD}}”。

# 示例1：
问题：
齐某和某公司签订了劳动合同，合同期限为2年。齐某工作的一年期间，公司一直以种种理由拖欠齐某的工资，齐某和该公司解除了劳动合同。现齐某欲就劳动报酬申请仲裁，下列说法中正确的有?\n\n选项：\nA: 齐某申请仲裁的期间为1年，自齐某知道或者应当知道权利受到损害之日起计算\nB: 齐某无须支付劳动仲裁的费用\nC: 齐某应当在劳动关系终止之日起1年内申请仲裁\nD: 仲裁庭在收到齐某的仲裁申请后应当先行调解
思考过程：
分析每个选项：\n\nA: 根据《劳动争议调解仲裁法》第二十七条，劳动争议申请仲裁的期间为一年，自当事人知道或者应当知道其权利被侵害之日起计算。因此，齐某申请仲裁的期间为1年，自齐某知道或者应当知道权利受到损害之日起计算，这个说法是正确的。\n\nB: 根据《劳动争议调解仲裁法》第五十三条，劳动争议仲裁不收费。因此，齐某无须支付劳动仲裁的费用，这个说法是正确的。\n\nC: 根据《劳动争议调解仲裁法》第二十七条，劳动争议申请仲裁的期间为一年，自当事人知道或者应当知道其权利被侵害之日起计算。劳动关系终止的，应当自劳动关系终止之日起一年内提出。因此，齐某应当在劳动关系终止之日起1年内申请仲裁，这个说法是正确的。\n\nD: 根据《劳动争议调解仲裁法》第四十三条，仲裁庭应当在开庭五日前，将开庭日期、地点书面通知双方当事人。仲裁庭应当将开庭情况记入笔录。因此，仲裁庭在收到齐某的仲裁申请后应当先行调解的说法是正确的。
则考生认为的答案是：
{{ABCD}}


# 示例2：
问题：
甲公司与乙公司签订了一份承揽合同，由乙公司为甲公司制造一批货物，甲公司给付乙公司定金5万元。事后，甲公司不再需要这批货物，遂要求解除合同。关于本案的正确表述是下列哪一项?\n\n选项：\nA: 乙公司应退还定金5万元\nB: 乙公司应退还定金10万元\nC: 甲公司无权要求退还定金\nD: 是否返还定金需甲乙双方协商而定
思考过程：
分析这个题目需要理解定金的相关法律规定。根据《中华人民共和国合同法》的规定，定金具有担保合同履行的作用。如果合同一方违约，定金可以适用定金罚则，即给付定金的一方不履行约定的债务的，无权要求返还定金；收受定金的一方不履行约定的债务的，应当双倍返还定金。\n\n在这个案例中，甲公司与乙公司签订了一份承揽合同，甲公司支付了5万元定金。如果甲公司不再需要这批货物并要求解除合同，那么甲公司作为定金的给付方，如果解除合同是由于甲公司的原因，那么甲公司无权要求返还定金。因此，选项A、B、D都不正确。\n\n根据《合同法》的规定，如果合同解除是由于非违约方的原因，那么定金应当返还。但是在这个案例中，甲公司解除合同的原因是不再需要这批货物，这属于甲公司单方面的原因，因此甲公司无权要求返还定金。
则考生认为的答案是：
{{C}}


# 示例3：
问题：
乙女与甲男婚后多年未生育，后甲男发现乙女因不愿生育曾数次擅自中止妊娠，为此甲男多次殴打乙女。乙女在被打住院后诉至法院要求离婚并请求损害赔偿，甲男以生育权被侵害为由提起反诉，请求乙女赔偿其精神损害。法院经调解无效，拟判决双方离婚。下列哪些选项是正确的?\n\n选项：\nA: 法院应支持乙女的赔偿请求\nB: 乙女侵害了甲男的生育权\nC: 乙女侵害了甲男的人格尊严\nD: 法院不应支持甲男的赔偿请求
思考过程：
本题涉及夫妻双方在婚姻关系中的权利和义务，特别是生育权和人格尊严权。首先，我们需要明确生育权是否属于法律明确保护的权利。根据我国《婚姻法》的规定，夫妻双方都有实行计划生育的义务，但并没有明确规定生育权为一项独立的权利。因此，甲男的生育权是否被侵害需要具体分析，不可直接下定论。\n\n其次，乙女擅自中止妊娠的行为是否构成对甲男生育权的侵害。从法律角度来看，乙女的行为虽然可能影响到甲男的生育计划，但并不直接构成对生育权的侵害。生育权的侵害通常是指因一方故意或过失导致另一方无法生育的情况，而乙女的行为只是拒绝生育，并未直接导致甲男无法生育，因此甲男的赔偿请求不应得到支持。\n\n最后，甲男多次殴打乙女的行为构成对乙女人格尊严的侵害。根据《婚姻法》第三十二条，夫妻双方应当互相尊重，互相扶助，禁止家庭暴力。因此，乙女有权要求甲男赔偿其因家庭暴力造成的精神损害。
则考生认为的答案是：
{{AD}}


# 现在请你根据上述示例，分析以下问题和考生的思考过程，给出考生认为的答案：
问题：\n{problem}\n思考过程：\n{solution}\n则考生认为的答案是：
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

            """
            with open('/workspace/verl/tmp.json', 'a+', encoding='utf-8') as f:
                f.write(json.dumps({
                    "prompt": prompt,
                    "response": response
                }, ensure_ascii=False) + '\n')
            """

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
            cot = predict.split('<思考>')[1].split('</思考>')[0].strip()
            if not cot:
                rewards.append(reward_parsing_failed)
                continue
            cot = cot.strip()
            if '综上所述' in cot or '综合分析' in cot or '综合以上分析' in cot:
                cot = cot.split('综上所述')[0].split('综合分析')[0].split('综合以上分析')[0].strip()
            else:
                pass
        except:
            rewards.append(reward_parsing_failed)
            continue
    
        question = prompt.split("\n\n问题：")[1].split('assistant')[0].strip()

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
    pattern = r"<思考>.*?</思考>.*?<回答>.*?</回答>"
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
        ground_truth = set(ground_truth)
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