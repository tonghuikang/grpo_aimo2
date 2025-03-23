# train_grpo.py

from train_config import (
    REFERENCE_MODEL_NAME,
    TRAIN_CUDA_VISIBLE_DEVICES,
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

    # Define reward function
    import datasets

    dataset = datasets.Dataset.from_csv("training_dataset.csv")
    # dataset = datasets.load_dataset("trl-lib/tldr", split="train")
    # dataset = dataset.select(range(1600))

    import math

    def length_preference_function(length, preferred_length=256):
        x = length / preferred_length
        return math.e * x * math.exp(-x)

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
        max_prompt_length=1024,
        max_completion_length=1024,
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
        logging_steps=8,
        # vllm configs
        use_vllm=True,
        # logging configs
        report_to="wandb",
        log_completions=True,
    )
    # Print relevant environment variables and configuration
    import torch.distributed as dist

    print("Environment Variables:")
    print(
        f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
    print(f"  RANK: {os.environ.get('RANK', 'not set')}")

    import torch

    print("\nPyTorch/CUDA Configuration:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CUDA device count: {torch.cuda.device_count()}")
    print(
        f"  PyTorch distributed initialized: {dist.is_available() and dist.is_initialized()}"
    )

    print("\nGRPO Configuration:")
    print(f"  num_generations: {training_args.num_generations}")
    print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")

    # Now initialize the trainer
    trainer = GRPOTrainer(
        model=REFERENCE_MODEL_NAME,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=dataset,
    )
    print(
        "trainer.model.generation_config.stop_strings",
        trainer.model.generation_config.stop_strings,
    )

    print(f"\nTrainer Configuration:")
    print(f"  trainer.accelerator.num_processes: {trainer.accelerator.num_processes}")
    print(f"  trainer.accelerator.device: {trainer.accelerator.device}")

    # Print trainer.accelerator state information
    print(f"  Accelerator state: {trainer.accelerator.state}")
    print(f"  Accelerator distributed type: {trainer.accelerator.distributed_type}")
    print(
        f"  Accelerator local process index: {trainer.accelerator.local_process_index}"
    )
    print(f"  Accelerator process index: {trainer.accelerator.process_index}")

    # Print trainer device map if available
    if hasattr(trainer, "model") and hasattr(trainer.model, "hf_device_map"):
        print(f"  Model device map: {trainer.model.hf_device_map}")

    # Print detailed trainer model device info
    print("\nModel Device Details:")
    if hasattr(trainer, "model"):
        try:
            print(f"  Model's device: {next(trainer.model.parameters()).device}")

            # Inspect n_gpu attribute
            if hasattr(trainer, "args") and hasattr(trainer.args, "n_gpu"):
                print(f"  trainer.args.n_gpu: {trainer.args.n_gpu}")

            # Try to get unwrapped model
            if hasattr(trainer, "accelerator") and hasattr(
                trainer.accelerator, "unwrap_model"
            ):
                unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
                print(
                    f"  Unwrapped model's device: {next(unwrapped_model.parameters()).device}"
                )

            # Inspect device_map if it exists
            if hasattr(trainer.model, "device_map"):
                print(f"  Model's device_map: {trainer.model.device_map}")
            elif hasattr(trainer.model, "hf_device_map"):
                print(f"  Model's hf_device_map: {trainer.model.hf_device_map}")

            # Check model's modules devices
            for name, module in trainer.model.named_children():
                try:
                    device = next(module.parameters()).device
                    print(f"  Model module '{name}' device: {device}")
                except StopIteration:
                    print(f"  Model module '{name}' has no parameters")
                except Exception as e:
                    print(f"  Error inspecting module '{name}': {e}")
        except Exception as e:
            print(f"  Error inspecting model: {e}")

    if hasattr(trainer, "_wrapped_model") and trainer._wrapped_model is not None:
        print(
            f"  Wrapped model's device: {next(trainer._wrapped_model.parameters()).device}"
        )

    # Check CUDA distribution
    if hasattr(trainer, "accelerator") and hasattr(
        trainer.accelerator, "use_distributed"
    ):
        print(f"  Accelerator using distributed: {trainer.accelerator.use_distributed}")

    # Check GPU allocation
    try:
        # Get process memory allocation for all devices
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            print(
                f"  Device {i} memory allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB"
            )
    except Exception as e:
        print(f"  Error getting GPU memory allocation: {e}")

    # Print PyTorch CUDA info
    print(f"  Current CUDA device: {torch.cuda.current_device()}")
    print(
        f"  Current CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

    # Print all visible CUDA devices
    visible_devices = []
    for i in range(torch.cuda.device_count()):
        visible_devices.append(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Visible CUDA devices: {visible_devices}")

    # Print memory info for each device
    memory_info = []
    for i in range(torch.cuda.device_count()):
        mem_info = torch.cuda.mem_get_info(i)
        free_mem = mem_info[0] / 1024**3  # Convert to GB
        total_mem = mem_info[1] / 1024**3  # Convert to GB
        memory_info.append(
            f"Device {i}: {free_mem:.2f}GB free / {total_mem:.2f}GB total"
        )
    print(f"  CUDA memory info: {memory_info}")

    # Print information from the accelerator state
    print(f"  Accelerator's state device: {trainer.accelerator.state.device}")

    # Distributed check
    print(
        f"  trainer.accelerator.distributed_type: {trainer.accelerator.distributed_type}"
    )
    print(
        f"  Relationship check: {training_args.num_generations} = {trainer.accelerator.num_processes} * {training_args.per_device_train_batch_size}? {training_args.num_generations == trainer.accelerator.num_processes * training_args.per_device_train_batch_size}"
    )

    print("\ntraining start")
    trainer.train()
