# modal deploy defaults_serve_model.py && curl https://tonghuikang--example-vllm-openai-compatible-salt1337-serve.modal.run/v1/models
# modal deploy defaults_serve_model.py && modal container list --json | jq -r '.[] | select(.["App Name"]=="example-vllm-openai-compatible-salt1337") | .["Container ID"]' | xargs -I {} modal container stop {} && curl https://tonghuikang--example-vllm-openai-compatible-salt1337-serve.modal.run/v1/models
# modal container exec ta-01J6375C1VRJ2HP00Q2GB3TBZ4 nvidia-smi
# CONTAINER_ID=$(modal container list --json | jq -r '.[] | select(."App Name" == "example-vllm-openai-compatible-salt1337") | ."Container ID"') && modal container exec $CONTAINER_ID nvidia-smi
# CONTAINER_ID=$(modal container list --json | jq -r '.[] | select(."App Name" == "example-vllm-openai-compatible-salt1337") | ."Container ID"') && modal container exec $CONTAINER_ID /bin/bash

import modal
import modal.gpu
import asyncio

vllm_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.7.1"
)

MODELS_DIR = "/aimo2"

MODEL_NAME = "Qwen/QwQ-32B-Preview"
N_GPU = 1
GPU = modal.gpu.A100(count=N_GPU, size="80GB")

MODEL_NAME = "KirillR/QwQ-32B-Preview-AWQ"
N_GPU = 1
GPU = modal.gpu.A100(count=N_GPU, size="80GB")

MODEL_NAME = "casperhansen/deepseek-r1-distill-qwen-7b-awq"
N_GPU = 1
GPU = modal.gpu.L4(count=N_GPU)

try:
    volume = modal.Volume.lookup("aimo2", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_llama.py")


app = modal.App("example-vllm-openai-compatible-salt1337-4xl4")

TOKEN = "super-secret-token"  # auth token. for production use, replace with a modal.Secret

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


from argparse import Namespace

@app.function(
    image=vllm_image,
    gpu=GPU,
    container_idle_timeout=5 * MINUTES,
    timeout=24 * HOURS,
    allow_concurrent_inputs=200,
    volumes={MODELS_DIR: volume},
)
@modal.asgi_app()
def serve():
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.usage.usage_lib import UsageContext
    from vllm.entrypoints.openai.api_server import init_app_state, mount_metrics

    volume.reload()  # ensure we have the latest version of the weights

    # create a fastAPI app that uses vLLM's OpenAI-compatible router
    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
    )

    # # security: CORS middleware for external requests
    # http_bearer = fastapi.security.HTTPBearer(
    #     scheme_name="Bearer Token",
    #     description="See code for authentication details.",
    # )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # # security: inject dependency on authed routes
    # async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
    #     if api_key.credentials != TOKEN:
    #         raise fastapi.HTTPException(
    #             status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
    #             detail="Invalid authentication credentials",
    #         )
    #     return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[])

    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODELS_DIR + "/" + MODEL_NAME,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=32768,
        max_seq_len_to_capture=32768,
        served_model_name=MODEL_NAME,
        enable_prefix_caching=True,
        max_logprobs=200,
        # enforce_eager=True,  # capture the graph for faster inference, but slower cold starts (30s > 20s)
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    model_config = get_model_config(engine)
    
    args = Namespace()
    args.response_role = "assistant"
    args.chat_template = None
    args.chat_template_content_format = "auto"
    args.enable_auto_tool_choice = None
    args.tool_call_parser = None
    args.model = MODEL_NAME
    args.served_model_name = [MODEL_NAME]
    args.lora_modules = []
    args.prompt_adapters = []
    args.max_log_len = 2048
    args.disable_log_stats = False
    args.disable_log_requests = False
    args.return_tokens_as_token_ids = False
    args.enable_prompt_tokens_details = False
    args.enable_reasoning = False
    args.reasoning_parser = None

    asyncio.run(init_app_state(engine_client=engine, model_config=model_config, state=web_app.state, args=args))
    mount_metrics(web_app)

    return web_app


def get_model_config(engine):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config