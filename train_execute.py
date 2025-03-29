# train_grpo.py

import modal
import modal.gpu

from train_config import (
    REFERENCE_MODEL_NAME,
    VLLM_CUDA_VISIBLE_DEVICES,
    TRAIN_CUDA_VISIBLE_DEVICES,
    TRAIN_NUM_PROCESSES,
    MAX_PROMPT_LENGTH,
    MAX_COMPLETION_LENGTH,
    GPU,
)

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
    .add_local_file(
        "./reward_evaluation_dataset.csv", "/root/reward_evaluation_dataset.csv"
    )
    .add_local_file("/Users/htong/.netrc", "/root/.netrc")  # for wandb credentials
    .add_local_file("/Users/htong/.kaggle/kaggle.json", "/root/kaggle/kaggle.json")
)

app = modal.App("grpo-training", image=image)


@app.function(
    image=image,
    gpu=GPU,
    timeout=2 * 60 * 60,
    max_containers=1,
)
def train_grpo():
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = (
        VLLM_CUDA_VISIBLE_DEVICES + "," + TRAIN_CUDA_VISIBLE_DEVICES
    )
    # import wandb
    # wandb.init(project="grpo-training", name=REFERENCE_MODEL_NAME)

    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Monkey patch to stop generating on ```python
    import trl

    print(trl.__file__)
    file_path = "/usr/local/lib/python3.11/site-packages/trl/scripts/vllm_serve.py"
    with open(file_path, "r") as file:
        content = file.read()
    old_string = "n=request.n,"
    print("index", content.index(old_string))
    modified_content = content.replace(old_string, f'{old_string} stop="```python",')
    with open(file_path, "w") as file:
        file.write(modified_content)

    file_path = "/usr/local/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py"
    with open(file_path, "r") as file:
        content = file.read()
    old_string = '"reward": rewards.tolist(),'
    print("index", content.index(old_string))
    modified_content = content.replace(
        old_string,
        f'{old_string} "advantage": advantages_to_log,',
    )
    with open(file_path, "w") as file:
        file.write(modified_content)

    file_path = "/usr/local/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py"
    with open(file_path, "r") as file:
        content = file.read()
    old_string = "advantages = rewards - mean_grouped_rewards"
    print("index", content.index(old_string))
    modified_content = content.replace(
        old_string,
        f"{old_string}; advantages_to_log = advantages.tolist(); [print(val.tolist()) for val in [rewards, advantages] if self.accelerator.is_main_process];",
    )
    with open(file_path, "w") as file:
        file.write(modified_content)

    file_path = "/usr/local/lib/python3.11/site-packages/trl/trainer/utils.py"
    with open(file_path, "r") as file:
        content = file.read()
    old_string = 'header_style="bold white", expand=True'
    print("index", content.index(old_string))
    modified_content = content.replace(
        old_string,
        f"{old_string}, width=290",
    )
    with open(file_path, "w") as file:
        file.write(modified_content)

    file_path = "/usr/local/lib/python3.11/site-packages/trl/trainer/utils.py"
    with open(file_path, "r") as file:
        content = file.read()
    old_string = "Console("
    print("index", content.index(old_string))
    modified_content = content.replace(
        old_string,
        f"{old_string}width=300",
    )
    with open(file_path, "w") as file:
        file.write(modified_content)

    # file_path = "/usr/local/lib/python3.11/site-packages/transformers/trainer.py"
    # with open(file_path, "r") as file:
    #     content = file.read()
    # old_string = "DataLoader(train_dataset,"
    # print("index", content.index(old_string))
    # modified_content = content.replace(
    #     old_string,
    #     f"{old_string} shuffle=False,",
    # )
    # with open(file_path, "w") as file:
    #     file.write(modified_content)

    file_path = "/usr/local/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py"
    with open(file_path, "r") as file:
        content = file.read()
    old_string = (
        "indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()"
    )
    print("index", content.index(old_string))
    modified_content = content.replace(
        old_string,
        f"indexes = torch.arange(self.num_samples).tolist()",
    )
    with open(file_path, "w") as file:
        file.write(modified_content)

    # Start vllm server as a background process
    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = VLLM_CUDA_VISIBLE_DEVICES
    subprocess.Popen(
        [
            "trl",
            "vllm-serve",
            "--model",
            REFERENCE_MODEL_NAME,
            "--max_model_len",
            f"{MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH}",
        ],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    # Launch training
    os.environ["CUDA_VISIBLE_DEVICES"] = TRAIN_CUDA_VISIBLE_DEVICES
    subprocess.run(
        [
            "accelerate",
            "launch",
            "--num_processes",
            TRAIN_NUM_PROCESSES,
            "train_grpo.py",
        ],
        # stdout=subprocess.DEVNULL,  # suppress stdout
    )

    # Close wandb run when training is complete
    # wandb.finish()


@app.local_entrypoint()
def main():
    train_grpo.remote()
