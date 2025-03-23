# train_grpo.py

import modal
import modal.gpu

from train_config import (
    REFERENCE_MODEL_NAME,
    VLLM_CUDA_VISIBLE_DEVICES,
    TRAIN_CUDA_VISIBLE_DEVICES,
    TRAIN_NUM_PROCESSES,
    GPU,
)

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .pip_install(
        "datasets",
        "trl[vllm]",
        "torch==2.5.1",
        "wandb",
        "liger-kernel",
        "wheel",
    )
    .pip_install(  # add flash-attn
        "flash-attn==2.7.4.post1", extra_options="--no-build-isolation"
    )
    .add_local_file("./train_grpo.py", "/root/train_grpo.py")
    .add_local_file("./train_config.py", "/root/train_config.py")
    .add_local_file("./training_dataset.csv", "/root/training_dataset.csv")
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")  # for wandb credentials
    .add_local_file("/Users/htong/.kaggle/kaggle.json", "/root/kaggle/kaggle.json")
)

app = modal.App("grpo-training", image=image)


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
)
def train_grpo():
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Monkey patch to stop generating on ```python
    import trl

    print(trl.__file__)
    file_path = "/usr/local/lib/python3.11/site-packages/trl/scripts/vllm_serve.py"
    with open(file_path, "r") as file:
        content = file.read()
    print("index", content.index("n=request.n,"))
    modified_content = content.replace("n=request.n,", 'n=request.n, stop="```python",')
    with open(file_path, "w") as file:
        file.write(modified_content)

    # Start vllm server as a background process
    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = VLLM_CUDA_VISIBLE_DEVICES
    subprocess.Popen(
        [
            "trl",
            "vllm-serve",
            "--model",
            REFERENCE_MODEL_NAME,
            "--max_model_len",
            "4096",
        ],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    import wandb

    # Initialize wandb for metrics logging
    wandb.init(project="grpo-training", name=REFERENCE_MODEL_NAME)

    os.environ["CUDA_VISIBLE_DEVICES"] = TRAIN_CUDA_VISIBLE_DEVICES
    subprocess.run(
        [
            "accelerate",
            "launch",
            "--num_processes",
            TRAIN_NUM_PROCESSES,
            "train_grpo.py",
        ],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    # Close wandb run when training is complete
    wandb.finish()


@app.local_entrypoint()
def main():
    train_grpo.remote()
