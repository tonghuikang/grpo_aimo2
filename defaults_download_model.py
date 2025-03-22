# ---
# args: ["--force-download"]
# ---
import modal

MODELS_DIR = "/aimo2"

DEFAULT_NAME = "Qwen/QwQ-32B-Preview"
DEFAULT_NAME = "KirillR/QwQ-32B-Preview-AWQ"
DEFAULT_NAME = "casperhansen/deepseek-r1-distill-qwen-7b-awq"
DEFAULT_NAME = "casperhansen/deepseek-r1-distill-qwen-14b-awq"
DEFAULT_NAME = "casperhansen/deepseek-r1-distill-qwen-32b-awq"
DEFAULT_NAME = "Qwen/QwQ-32B-AWQ"

volume = modal.Volume.from_name("aimo2", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface")])


@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, model_revision="", force_download=False):
    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        model_name,
        local_dir=MODELS_DIR + "/" + model_name,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        # revision=model_revision,
        force_download=force_download,
    )

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = DEFAULT_NAME,
    force_download: bool = False,
):
    download_model.remote(model_name, force_download)
