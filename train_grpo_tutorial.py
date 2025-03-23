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
        # output_dir="DeepSeek-R1-Distill-Qwen-1.5B-GRPO",
        output_dir="Qwen2-0.5B-GRPO",
        logging_steps=10,
        gradient_accumulation_steps=4,
        num_generations=16,
        per_device_train_batch_size=16,
        report_to="wandb",
        num_train_epochs=1.0,
        use_vllm=True,
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
