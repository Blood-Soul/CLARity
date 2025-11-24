CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.api_server \
    --model model/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000