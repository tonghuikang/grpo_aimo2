# https://github.com/huggingface/trl/pull/2899

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


import modal
import modal.gpu

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
    .add_local_file("./tutorial_worker_vllm.py", "/root/tutorial_worker_vllm.py")
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")  # for wandb credentials
    .add_local_file("/Users/htong/.kaggle/kaggle.json", "/root/kaggle/kaggle.json")
)


dataset = load_dataset("trl-lib/tldr", split="train")


# Dummy reward function: the closer the completion is to 20 characters, the higher the reward
def reward_len(completions, **kwargs):
    return [-abs(100 - len(completion)) for completion in completions]


app = modal.App("grpo-training", image=image)

GPU = "L4:3"


import os
import subprocess


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
)
def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    subprocess.Popen(
        [
            "trl",
            "vllm-serve",
            "--model",
            "Qwen/Qwen2.5-0.5B",
            "--max_model_len",
            "4096",
        ],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    # Launch training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    subprocess.run(
        [
            "accelerate",
            "launch",
            "--num_processes",
            "2",
            "tutorial_worker_vllm.py",
        ],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )


@app.local_entrypoint()
def main():
    train.remote()
