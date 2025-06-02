#!/bin/bash
#SBATCH --job-name=infer_qwen3           # your job name
#SBATCH --nodes=1                        # 1 node is enough for inference
#SBATCH --ntasks-per-node=1              # one task per node
#SBATCH --cpus-per-task=16               # adjust if you need more CPU threads
#SBATCH --gres=gpu:8                     # request 8 GPUs (change to 2, 4, etc. as needed)
#SBATCH --output=logs/stdout/vllm_qwen3_infer_%x_%j.out
#SBATCH --error=logs/stdout/vllm_qwen3_infer_%x_%j.err
#SBATCH --qos=iq
#SBATCH --exclusive

# Get the directory of this script
# This ensures SCRIPT_DIR is always the absolute path where this .sh file resides
SCRIPT_DIR="/mnt/weka/home/liyuan/inference_intro" 

# Load your Conda setup and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate inference_env  # Change <env_name> to your environment name

# Set up per-process HuggingFace cache in your home directory
# This avoids permission issues and ensures the cache is in a writable location.
export HF_HOME=~/hf_cache_$(hostname)_$SLURM_PROCID
mkdir -p "$HF_HOME" # Use quotes for safety, though not strictly necessary here

# (Optional) Triton cache dir if vLLM uses it
export TRITON_HOME=${SCRIPT_DIR}/triton_cache
mkdir -p "$TRITON_HOME" # Use quotes for safety

# No need to 'cd' here, as we will use the full path for the Python script.

# Execute the Python script using its full absolute path
# This ensures the script is found regardless of the current working directory.
srun --cpu-bind=none python "$SCRIPT_DIR/generate_response.py" \
     --model_path "$SCRIPT_DIR/checkpoints/Qwen3-0.6B" \
     --model_name Qwen3-0.6B \
     --tensor_parallel_size 8 \
     --output_dir "$SCRIPT_DIR/outputs"