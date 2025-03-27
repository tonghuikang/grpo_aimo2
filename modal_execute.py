import modal
import modal.gpu


"""
Execute

modal run modal_execute.py

Download artifacts

for file in $(modal nfs ls saved_files); do echo "Checking: $file"; if [ ! -f "generation/$file" ]; then echo "Downloading: $file"; modal nfs get saved_files $file generation/$file; else echo "Skipping (already exists): $file"; fi; done
"""

nfs = modal.NetworkFileSystem.from_name("saved_files", create_if_missing=True)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "polars",
        "pandas",
        "vllm",
        "numpy",
        "transformers",
        "openai",
        "triton",
    )
    .add_local_file("./reference.csv", "/root/reference.csv")
    .add_local_file("./AIME_Dataset_1983_2024.csv", "/root/AIME_Dataset_1983_2024.csv")
    .add_local_file("./notebook.py", "/root/notebook.py")
)


app = modal.App("execute-notebook")


MODELS_DIR = "/aimo2"
volume = modal.Volume.from_name("aimo2", create_if_missing=False)  # for the models


@app.function(
    image=image,
    gpu="H100",
    cpu=32,
    timeout=24 * 60 * 60,
    volumes={MODELS_DIR: volume},
)
def execute():

    import os
    import subprocess

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(
        [
            "python3",
            "-u",
            "notebook.py",
        ],
        text=True,
    )

    import glob
    import datetime

    now = datetime.datetime.now()
    suffix = now.strftime("%m-%d-%H-%M-%S")

    for file in glob.glob("generation_*.csv"):
        nfs.add_local_file(file, file.replace(".csv", f"_{suffix}.csv"))


@app.local_entrypoint()
def main():
    execute.remote()
