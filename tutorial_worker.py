from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


dataset = load_dataset("trl-lib/tldr", split="train")


# Dummy reward function: the closer the completion is to 20 characters, the higher the reward
def reward_len(completions, **kwargs):
    return [-abs(100 - len(completion)) for completion in completions]


def main():
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


if __name__ == "__main__":
    main()
