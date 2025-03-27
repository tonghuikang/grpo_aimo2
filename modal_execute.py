import modal
import modal.gpu


# cuda_version = "11.8.0"  # should be no greater than host CUDA version
# flavor = "devel"  #  includes full CUDA toolkit
# operating_sys = "ubuntu22.04"
# tag = f"{cuda_version}-{flavor}-{operating_sys}"

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
    .add_local_file("./notebook.py", "/root/notebook.py")
    .add_local_file("./reference.csv", "/root/reference.csv")
)


app = modal.App("execute-notebook")


MODELS_DIR = "/aimo2"
volume = modal.Volume.from_name("aimo2", create_if_missing=False)  # for the models


@app.function(
    image=image,
    gpu="H100",
    cpu=32,
    timeout=2 * 60 * 60,
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
