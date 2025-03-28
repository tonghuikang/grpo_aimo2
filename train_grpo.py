# train_grpo.py

from train_config import (
    REFERENCE_MODEL_NAME,
    TRAIN_CUDA_VISIBLE_DEVICES,
    NUM_GENERATIONS,
    TRAIN_NUM_GPUS,
    MAX_PROMPT_LENGTH,
    MAX_COMPLETION_LENGTH,
)


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Start vllm server as a background process
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = TRAIN_CUDA_VISIBLE_DEVICES

    # Load tokenizer for scoring use
    from transformers import AutoTokenizer

    # takes time, you want to do this while vLLM is loading
    print("Tokenizer loading")
    tokenizer = AutoTokenizer.from_pretrained(
        REFERENCE_MODEL_NAME,
        trust_remote_code=True,
    )
    print("Tokenizer loaded")

    import torch
    from liger_kernel.transformers import AutoLigerKernelForCausalLM

    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    model = AutoLigerKernelForCausalLM.from_pretrained(
        REFERENCE_MODEL_NAME, **model_kwargs
    )

    # Define reward function
    import datasets

    dataset = datasets.Dataset.from_csv("training_dataset.csv")
    # dataset = datasets.load_dataset("trl-lib/tldr", split="train")
    # dataset = dataset.select(range(1600))

    import math

    def length_preference_function(length, preferred_length=256, ceiling=1.0):
        # maps [0,inf] -> [0,1]
        x = length / preferred_length
        y = math.e * x * math.exp(-x)
        return (min(y, ceiling)) ** (1 / 3) / ceiling

    import re

    def reward_func_individual(completion, correct_answer):
        match = re.search(r"\\boxed\{(.*?)\}", completion)
        boxed_content = match.group(1) if match else ""

        answer_attempt = bool(boxed_content != "")
        answer_correct = boxed_content == str(correct_answer)

        relevant_index = len(completion)
        contains_python_opening = "```python" in completion
        if contains_python_opening:
            relevant_index = completion.index("```python")
        completion_length = len(tokenizer.encode(completion[:relevant_index]))

        if not answer_attempt and not contains_python_opening:
            return -5

        if not answer_attempt:
            # penalize long sequences, likely attempting to solve on its own
            # not sure if it affects the math sequences
            return -0.5 + length_preference_function(completion_length)

        if not answer_correct:
            return -10

        # attempted answer and is correct
        return 9.5 + length_preference_function(completion_length)

    def reward_func(prompts, completions, correct_answer, **kwargs):
        return [
            reward_func_individual(completion, correct_answer_)
            for completion, correct_answer_ in zip(completions, correct_answer)
        ]

    # Define training config
    from trl import GRPOConfig, GRPOTrainer

    learning_rate = 3e-5
    num_iterations = 2
    gradient_accumulation_steps = 8

    training_args = GRPOConfig(
        # length limits (which needs to be changed for these to be useful)
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        # training config, see docstring in GRPOTrainer._get_train_sampler
        # I want a huge number of generations so I can calculate the reward / advantage for all of them
        # Currently, I am forced to have lots of GPUs before I can have a huge number of generations
        # Breaking them down into separate generations is not the same,
        # because that results in a different advantage calculation
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=NUM_GENERATIONS // TRAIN_NUM_GPUS,
        # num_devices * this_number should be num_generations
        gradient_accumulation_steps=num_iterations,
        num_train_epochs=1.0,
        # output_dir="DeepSeek-R1-Distill-Qwen-1.5B-GRPO",
        num_iterations=num_iterations,
        output_dir=f"DeepSeek-R1-Distill-Qwen-1.5B-GRPO-{num_iterations}-{gradient_accumulation_steps}-{learning_rate:.0e}",
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_steps=7,
        learning_rate=learning_rate,
        epsilon_high=0.28,
        beta=0.1,
        scale_rewards=False,
        # vllm configs
        use_vllm=True,
        # logging configs
        report_to="wandb",
        log_completions=True,
    )

    # Now initialize the trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )

    print("\ntraining start")
    trainer.train()
