# train_grpo.py

import modal
import modal.gpu

image = (
    modal.Image.debian_slim()
    .pip_install("datasets", "trl[vllm]", "wandb")
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")  # for wandb credentials
    .add_local_file("/Users/htong/.kaggle/kaggle.json", "/root/kaggle/kaggle.json")
    .add_local_file("./training_dataset.csv", "/root/training_dataset.csv")
)

GPU = "H100:3"
app = modal.App("grpo-training", image=image)


REFERENCE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


import math


def length_preference_function(length, preferred_length=256):
    x = length / preferred_length
    return math.e * x * math.exp(-x)


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
)
def train_grpo():
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Start vllm server as a background process
    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    subprocess.Popen(
        ["trl", "vllm-serve", "--model", REFERENCE_MODEL_NAME],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Load tokenizer for scoring use
    from transformers import AutoTokenizer

    # takes time, you want to do this while vLLM is loading
    print("Tokenizer loading")
    tokenizer = AutoTokenizer.from_pretrained(
        REFERENCE_MODEL_NAME,
        trust_remote_code=True,
    )
    print("Tokenizer loaded")

    import wandb

    # Initialize wandb for metrics logging
    wandb.init(project="grpo-training", name=REFERENCE_MODEL_NAME)

    # Define reward function
    import datasets

    dataset = datasets.Dataset.from_csv("training_dataset.csv")
    # dataset = datasets.load_dataset("trl-lib/tldr", split="train")
    # dataset = dataset.select(range(1600))

    import re

    def reward_func_individual(completion, correct_answer):
        match = re.search(r"\\boxed\{(.*?)\}", completion)
        boxed_content = match.group(1) if match else ""

        answer_attempt = bool(boxed_content != "")
        answer_correct = boxed_content == str(correct_answer)

        relevant_index = len(completion)
        if "```python" in completion:
            relevant_index = completion.index("```python")
        completion_length = len(tokenizer.encode(completion[:relevant_index]))

        if not answer_attempt:
            # penalize long sequences, likely attempting to solve on its own
            # not sure if it affects the math sequences
            return 0.1 * length_preference_function(completion_length)

        if not answer_correct:
            return -1

        return length_preference_function(completion_length)

    def reward_func(prompts, completions, correct_answer, **kwargs):
        return [
            reward_func_individual(completion, correct_answer_)
            for completion, correct_answer_ in zip(completions, correct_answer)
        ]

    # Define training config
    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        # length limits (which needs to be changed for these to be useful)
        max_prompt_length=2048,
        max_completion_length=2048,
        # training config, see docstring in GRPOTrainer._get_train_sampler
        # I want a huge number of generations so I can calculate the reward / advantage for all of them
        # Currently, I am forced to have lots of GPUs before I can have a huge number of generations
        # Breaking them down into separate generations is not the same,
        # because that results in a different advantage calculation
        num_generations=8,
        per_device_train_batch_size=4,  # num_devices * this_number should be num_generations
        gradient_accumulation_steps=8,
        num_train_epochs=1.0,
        # output_dir="DeepSeek-R1-Distill-Qwen-1.5B-GRPO",
        num_iterations=1,  # this means reusing completions?
        output_dir="DeepSeek-R1-Distill-Qwen-1.5B-GRPO",
        logging_steps=16,
        # vllm configs
        use_vllm=True,
        # logging configs
        report_to="wandb",
        log_completions=True,
    )
    trainer = GRPOTrainer(
        model=REFERENCE_MODEL_NAME,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )
    print("training start")
    trainer.train()

    # Close wandb run when training is complete
    wandb.finish()


@app.local_entrypoint()
def main():
    train_grpo.remote()
