CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
    --model shared-nvme/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000