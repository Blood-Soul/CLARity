#!/bin/bash

# pip 下载使用清华源
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "1. install inference framework vLLM and PyTorch it needs"
pip install --no-cache-dir -i $PIP_INDEX_URL "vllm==0.11.0"

echo "2. install basic packages"
pip install -i $PIP_INDEX_URL --no-cache-dir \
    "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pre-commit ruff tensorboard

echo "pyext is lack of maintainace and cannot work with python 3.12."
echo "if you need it for prime code rewarding, please install using patched fork:"
echo "pip install git+https://github.com/ShaohonChen/PyExt.git@py311support"

pip install -i $PIP_INDEX_URL "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. install FlashInfer"
pip install -i $PIP_INDEX_URL --no-cache-dir flashinfer-python==0.3.1

echo "4. May need to fix opencv"
pip install -i $PIP_INDEX_URL --no-cache-dir opencv-python==4.11.0.86
pip install -i $PIP_INDEX_URL --no-cache-dir opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "Successfully installed all packages"
