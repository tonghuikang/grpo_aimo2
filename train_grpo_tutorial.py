# train_grpo.py

import subprocess

import modal
import modal.gpu

image = (
    modal.Image.debian_slim()
    .pip_install("datasets", "trl[vllm]", "wandb")
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")
)

GPU = "H100:2"
app = modal.App("grpo-training", image=image)


REFERENCE_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


@app.function(
    image=image,
    gpu=GPU,
    timeout=1 * 60 * 60,
)
def train_grpo():
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Start vllm server as a background process
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.Popen(
        ["trl", "vllm-serve", "--model", REFERENCE_MODEL_NAME],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer
    import wandb

    # Initialize wandb
    wandb.init(project="grpo-training", name="Qwen2-0.5B")

    dataset = load_dataset("trl-lib/tldr", split="train")
    dataset = dataset.select(range(1600))

    # Define the reward function, which rewards completions that are close to 20 characters
    def reward_func(completions, **kwargs):
        def count(srr):
            prev = -1
            count = 0
            for word in srr.split():
                if word:
                    x = word[0]
                    if ord(x) > prev:
                        prev = ord(x)
                        count += 1
            return count**3 / (1 + len(srr))

        return [count(completion) for completion in completions]

    training_args = GRPOConfig(
        # length limits (which needs to be changed for these to be useful)
        max_completion_length=1024,
        max_prompt_length=1024,
        # training config, see docstring in GRPOTrainer._get_train_sampler
        # I want a huge number of generations so I can calculate the reward for all of them
        # Currently, I am forced to have lots of GPUs before I can have a huge number of generations
        num_generations=16,
        per_device_train_batch_size=16,  # num_devices * this_number should be num_generations
        num_train_epochs=1.0,
        gradient_accumulation_steps=16,
        # output_dir="DeepSeek-R1-Distill-Qwen-1.5B-GRPO",
        num_iterations=1,  # this means reusing completions?
        output_dir="Qwen2-0.5B-GRPO",
        logging_steps=16,
        # vllm configs
        use_vllm=True,
        # logging configs
        report_to="wandb",
        log_completions=True,
    )
    trainer = GRPOTrainer(
        # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
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
