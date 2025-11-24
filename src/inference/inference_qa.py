import argparse
from openai import OpenAI
import os
from datasets import Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_data", default='data/original_data/jec/prepared_parquet/JEC_1_multi_choice_test_prepared.parquet', type=str)
    parser.add_argument("--output_dir", default='data/infer_answer', type=str)
    parser.add_argument("--output_name", default='JEC_1_multi_choice_test_inferenced', type=str)
    parser.add_argument("--base_url", default='http://localhost:8000/v1', type=str)
    parser.add_argument("--api_key", default='', type=str)
    parser.add_argument("--model_name", default='Qwen/Qwen2.5-7B-Instruct', type=str)

    args = parser.parse_args()

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )

    def call_llm(prompt):
        try:
            resp = client.chat.completion.create(
                model=args.model_name,
                messages=prompt
            )
            answer = resp.choices[0].message["content"]
        except Exception as e:
            print(f'Error: {e}')
            answer = "error"
        return answer
    
    def get_answer(example):
        prompt = example["prompt"]
        answer = call_llm(prompt)

        example["response"] = {
            "model_name": args.model_name,
            "answer": answer
        }

    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f'{args.input_data} not exists !')
    else:
        dataset = Dataset.from_parquet(args.input_data)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    inferenced_dataset = dataset.map(get_answer)
    inferenced_dataset.to_parquet(f'{args.output_dir}/{args.output_name}.parquet')