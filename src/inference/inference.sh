python src/inference/inference_qa.py \
    --input_data data/original_data/med/prepared_parquet/dev_prepared.parquet \
    --output_dir data/infer_answer \
    --output_name dev_inferenced \
    --base_url http://localhost:8000/v1 \
    --model_name shared-nvme/Qwen2.5-7B-Instruct