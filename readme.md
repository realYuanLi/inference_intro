

## Intro
This is a minimal (non-optimal, a bit dumb, but beginner-friendly) example for model inference (letting the model generate responses). It does not include advanced inference techniques, but simply calls vLLM.

- The testing data is in `testing_data.json`, which includes questions to ask LLMs.

## Prerequisites

- Access to a SLURM cluster with GPUs
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Python 3.8+
- CUDA 12.4 drivers (for torch==2.5.1)

## Environment Setup

1. **Clone the repository and navigate to this directory:**
   ```bash
   git clone <your-repo-url>
   cd inference_intro
   ```

2. **Create and activate the conda environment:**
   ```bash
   conda create -n <env_name> python=3.10 -y
   conda activate <env_name>
   ```

3. **Install requirements:**
   ```bash
   pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

   pip3 install vllm transformers
   ```

## Model Download

1. **Download Qwen3-0.6B model weights** and place them in the following directory:
   ```
   ../checkpoints/Qwen3-0.6B
   ```
   - The directory should contain the model files (e.g., config.json, tokenizer files, etc.)
   - You can obtain the model from [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) .


## Running Inference

1. **Edit SLURM parameters in `run_inference.sh` as needed** (e.g., number of GPUs, CPUs, job name).
   - For example, if you have access to 2 GPUs, set `tensor_parallel_size` to 2 and `#SBATCH --gres=gpu:2`

2. **Submit the job to SLURM:**
   ```bash
   sbatch run_inference.sh
   ```

## Output

- The generated responses will be saved in:
  ```
  inference_intro/outputs/Qwen3-0.6B.json
  ```
- SLURM logs will be saved in:
  ```
  inference_intro/logs/stdout/
  ```

In the log file, you will see the model being loaded within minutes, and start to generate responses.

## File Structure

```
inference_intro/
├── run_inference.sh
├── readme.md
├── generate_response.py
├── testing_data.json
├── outputs/
│   └── Qwen3-0.6B.json
├── logs/
│   └── stdout/
├── ../models/Qwen3-0.6B/
```

## References
- [vLLM Documentation](https://vllm.readthedocs.io/)
- [Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)

