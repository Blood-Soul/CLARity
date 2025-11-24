python jec-qa.py \
    --input_data data/original_data/jec/JEC_1_multi_choice_test.json \
    --output_dir data/original_data/jec/prepared_parquet

python medqa-usmle.py \
    --input_data data/original_data/med/dev.jsonl \
    --output_dir data/original_data/med/prepared_parquet
