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
    .add_local_file("./train_grpo.py", "/root/train_grpo.py")
    .add_local_file("./train_config.py", "/root/train_config.py")
    .add_local_file("./training_dataset.csv", "/root/training_dataset.csv")
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")  # for wandb credentials
    .add_local_file("/Users/htong/.kaggle/kaggle.json", "/root/kaggle/kaggle.json")
)


dataset = load_dataset("trl-lib/tldr", split="train")


# Dummy reward function: the closer the completion is to 20 characters, the higher the reward
def reward_len(completions, **kwargs):
    return [-abs(100 - len(completion)) for completion in completions]


app = modal.App("grpo-training", image=image)

GPU = "H100:1"


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
)
def train():
    num_iterations = 4
    training_args = GRPOConfig(
        output_dir=f"Qwen2.5-0.5B-GRPO-2899-μ={num_iterations}",
        logging_steps=5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        num_generations=8,
        max_prompt_length=64,
        max_completion_length=64,
        log_completions=True,
        max_steps=200,
        num_iterations=num_iterations,
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-0.5B",
        reward_funcs=reward_len,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()


@app.local_entrypoint()
def main():
    train.remote()
