python src/data_preparation/jec-qa.py \
    --input_data data/original_data/jec/JEC_1_multi_choice_train.json \
    --output_dir data/original_data/jec/prepared_parquet \
    --output_file_name prepared_jec_train_data

python src/data_preparation/medqa-usmle.py \
    --input_data data/original_data/med/train.jsonl \
    --output_dir data/original_data/med/prepared_parquet \
    --output_file_name prepared_med_train_data
