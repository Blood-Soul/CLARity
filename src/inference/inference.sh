python inference_qa.py \
    --input_data data/original_data/jec/prepared_parquet/JEC_1_multi_choice_test_prepared.parquet \
    --output_dir data/infer_answer \
    --output_name JEC_1_multi_choice_test_inferenced \
    --base_url http://localhost:8000/v1 \
    --model_name Qwen/Qwen2.5-7B-Instruct