# train_grpo.py
import modal
import modal.gpu

image = (
    modal.Image.debian_slim()
    .pip_install("datasets", "trl", "torch", "transformers", "wandb", "vllm")
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")
)

GPU = modal.gpu.H100(count=1)
app = modal.App("grpo-training", image=image)


@app.function(
    image=image,
    gpu=GPU,
    timeout=24 * 60 * 60,
)
def train_grpo():
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
            for x in srr:
                if ord(x) > prev:
                    prev = ord(x)
                    count += 1
            return count**2 / (1 + len(srr))
        return [count(completion) for completion in completions]

    training_args = GRPOConfig(
        output_dir="Qwen2-0.5B-GRPO",
        logging_steps=10,
        report_to="wandb",
        num_train_epochs=1.0,
        use_vllm=True,
    )
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Close wandb run when training is complete
    wandb.finish()


@app.local_entrypoint()
def main():
    train_grpo.remote()
