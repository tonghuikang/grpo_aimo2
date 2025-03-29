# train_grpo.py

from train_config import (
    REFERENCE_MODEL_NAME,
    TRAIN_CUDA_VISIBLE_DEVICES,
    NUM_GENERATIONS,
    CAPACITY_PER_GPU,
    TRAIN_NUM_GPUS,
    MAX_PROMPT_LENGTH,
    MAX_COMPLETION_LENGTH,
)


if __name__ == "__main__":
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch

    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not available")

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
    evaluation_dataset = datasets.Dataset.from_csv("reward_evaluation_dataset.csv")
    # dataset = datasets.load_dataset("trl-lib/tldr", split="train")
    # dataset = dataset.select(range(1600))

    # functions for reward weights
    correctness_func_counter = [0]
    TOTAL_CALLS_ESTIMATED = len(dataset) // (
        CAPACITY_PER_GPU * TRAIN_NUM_GPUS // NUM_GENERATIONS
    )  # evaluation counts not considered
    CALLS_UNTIL_CONSIDER_CORRECTNESS = TOTAL_CALLS_ESTIMATED // 4
    CALLS_UNTIL_CONSIDER_CORRECTNESS_FULLY = TOTAL_CALLS_ESTIMATED
    MAX_CORRECTNESS_WEIGHT = 0.75

    def get_correctness_weight(increment=True):
        if increment:
            correctness_func_counter[0] += 1
        # correctness weight is zero at the start
        factor = (correctness_func_counter[0] - CALLS_UNTIL_CONSIDER_CORRECTNESS) / (
            CALLS_UNTIL_CONSIDER_CORRECTNESS_FULLY - CALLS_UNTIL_CONSIDER_CORRECTNESS
        )
        factor = min(1, factor)
        return MAX_CORRECTNESS_WEIGHT * factor

    def get_length_weight():
        return (1 - get_correctness_weight(increment=False)) / 2

    def get_formatting_weight():
        return (1 - get_correctness_weight(increment=False)) / 2

    # helper functions for reward

    import math

    def length_preference_function(
        length, preferred_length=256, ceiling=1.0, scaling=1
    ):
        # maps [0,inf] -> [0,1]
        x = length / preferred_length
        y = math.e * x * math.exp(-x)
        return (min(y, ceiling)) ** (scaling) / ceiling

    def is_formatting_valid(completion):
        if "</think>" not in completion:
            return False
        if "oxed{" not in completion and "```python" not in completion:
            return False
        if "oxed{" in completion and "```python" in completion:
            return False
        return True

    def length_function(completion):
        if not is_formatting_valid(completion):
            return 0
        tokens = tokenizer.encode(completion)
        thinking_length = tokens.index(151649)  # </think>
        full_length = len(tokens)
        return 0.5 * length_preference_function(
            thinking_length, preferred_length=256 - 10, scaling=0.95
        ) + 0.5 * length_preference_function(
            full_length, preferred_length=256, scaling=0.95
        )

    def formatting_function(completion):
        if not is_formatting_valid(completion):
            return 0

        completion = completion[: completion.index("</think>")].strip()

        lines = completion.split("\n\n")
        denominator = 1
        numerator = 1
        for line in lines:
            length = len(line)
            score = length_preference_function(length, preferred_length=100, scaling=0.90)
            numerator += score * length
            denominator += length
            numerator += 100 * score
            denominator += 100
        return numerator / denominator

    def style_func(prompts, completions, correct_answer, **kwargs):
        # each line of thinking text should be 100 characters
        # -> [0, formatting_weight + length_weight]

        formatting_weight = get_formatting_weight()
        length_weight = get_length_weight()
        return [
            formatting_weight * formatting_function(completion)
            + length_weight * length_function(completion)
            for completion in completions
        ]

    def formatting_ref(prompts, completions, correct_answer, **kwargs):
        # each line of thinking text should be 100 characters
        # -> [0, 1]

        weight = 0.001
        return [weight * formatting_function(completion) for completion in completions]

    def length_ref(prompts, completions, correct_answer, **kwargs):
        # each line of thinking text should be 100 characters
        # -> [0, 1]

        weight = 0.001
        return [weight * length_function(completion) for completion in completions]

    import re

    def correctness_function(completion, correct_answer):
        # if there is an answer in the response, it should be correct
        if not is_formatting_valid(completion):
            return 0

        match = re.search(r"\\boxed\{(.*?)\}", completion)
        boxed_content = match.group(1) if match else ""

        answer_attempt = bool(boxed_content != "")
        answer_correct = boxed_content == str(correct_answer)
        contains_python_opening = "```python" in completion
        if not answer_attempt and not contains_python_opening:
            # with is_formatting_valid, this branch is redundant
            return -0.5

        if not answer_attempt:
            # penalize long sequences, likely attempting to solve on its own
            # not sure if it affects the math sequences
            return 0

        if not answer_correct:
            return -1

        # attempted answer and is correct
        return 1

    def correctness_func(prompts, completions, correct_answer, **kwargs):
        # the answer should be correct

        weight = get_correctness_weight()
        return [
            weight * correctness_function(completion, correct_answer_)
            for completion, correct_answer_ in zip(completions, correct_answer)
        ]

    def correctness_ref(prompts, completions, correct_answer, **kwargs):
        # use to log correctness without affecting reward value
        factor = 0.001
        return [
            factor * correctness_function(completion, correct_answer_)
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
        per_device_train_batch_size=CAPACITY_PER_GPU,
        # num_devices * this_number should be num_generations
        gradient_accumulation_steps=num_iterations,
        num_train_epochs=1.0,
        # output_dir="DeepSeek-R1-Distill-Qwen-1.5B-GRPO",
        num_iterations=num_iterations,
        output_dir=f"DeepSeek-R1-Distill-Qwen-1.5B-GRPO-{num_iterations}-{gradient_accumulation_steps}-{learning_rate:.0e}",
        logging_steps=1,
        lr_scheduler_type="cosine",
        warmup_steps=32,
        learning_rate=learning_rate,
        epsilon_high=0.28,
        beta=0.1,
        scale_rewards=False,
        # vllm configs
        use_vllm=True,
        # logging configs
        report_to="wandb",
        log_completions=True,
        eval_on_start=True,
        eval_strategy="steps",
        eval_steps=16,
    )

    # Now initialize the trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            style_func,
            correctness_func,
            # reference functions
            formatting_ref,
            length_ref,
            correctness_ref,
        ],
        args=training_args,
        train_dataset=dataset,
        eval_dataset=evaluation_dataset,
    )

    print("\ntraining start")
    trainer.train()
