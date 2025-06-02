#!/bin/bash
#SBATCH --job-name=infer_intro           # your job name
#SBATCH --nodes=1                          # 1 node is enough for inference
#SBATCH --ntasks-per-node=1                # one task per node
#SBATCH --cpus-per-task=16                 # adjust if you need more CPU threads
#SBATCH --gres=gpu:8                       # request 8 GPUs (with less GPUs, you can change this to 4, 2, 1)
#SBATCH --output=${SCRIPT_DIR}/../logs/stdout/vllm_iq_infer_%x_%j.out
#SBATCH --error=${SCRIPT_DIR}/../logs/stdout/vllm_iq_infer_%x_%j.err
#SBATCH --account=iq

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load your Conda setup and activate environment
source ${SCRIPT_DIR}/../miniconda3/etc/profile.d/conda.sh
conda activate inference_env

# Set up per‐process HuggingFace cache (to avoid clashes across GPUs)
export HF_HOME=${SCRIPT_DIR}/../hf_cache_$(hostname)_$SLURM_PROCID
mkdir -p $HF_HOME

# (Optional) Triton cache dir if vLLM uses it
export TRITON_HOME=${SCRIPT_DIR}/../triton_cache
mkdir -p $TRITON_HOME

# Move to directory containing generate_response.py
cd ${SCRIPT_DIR}/../inference_intro

srun --cpu-bind=none python ${SCRIPT_DIR}/../inference_intro/generate_response.py \
     --model_path ${SCRIPT_DIR}/../checkpoints/Qwen3-0.6B \
     --model_name Qwen3-0.6B \
     --tensor_parallel_size 8 \
     --output_dir ${SCRIPT_DIR}/../inference_intro 