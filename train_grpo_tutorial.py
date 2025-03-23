# train_grpo.py

import modal
import modal.gpu

image = (
    modal.Image.debian_slim()
    .pip_install("datasets", "trl[vllm]", "wandb")
    .add_local_file("./train_grpo.py", "/root/train_grpo.py")
    .add_local_file("./training_dataset.csv", "/root/training_dataset.csv")
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")  # for wandb credentials
    .add_local_file("/Users/htong/.kaggle/kaggle.json", "/root/kaggle/kaggle.json")
)

GPU = "H100:3"
app = modal.App("grpo-training", image=image)


REFERENCE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
)
def train_grpo():
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Monkey patch to stop generating on ```python
    file_path = "/usr/local/lib/python3.11/site-packages/trl/scripts/vllm_serve.py"
    with open(file_path, 'r') as file:
        content = file.read()
    print("index", content.index("n=request.n,"))
    modified_content = content.replace('n=request.n,', 'n=request.n, stop="```python",')
    with open(file_path, 'w') as file:
        file.write(modified_content)

    # Start vllm server as a background process
    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    subprocess.Popen(
        ["trl", "vllm-serve", "--model", REFERENCE_MODEL_NAME],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    import wandb

    # Initialize wandb for metrics logging
    wandb.init(project="grpo-training", name=REFERENCE_MODEL_NAME)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    subprocess.run(
        ["accelerate", "launch", "--num_processes", "2", "train_grpo.py"],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    # Close wandb run when training is complete
    wandb.finish()


@app.local_entrypoint()
def main():
    train_grpo.remote()
