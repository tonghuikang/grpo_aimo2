import os
import argparse

def is_on_kaggle_submission() -> bool:
    return bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


if __name__ == "__main__":
    # Add command line arguments parser for our custom arguments
    parser = argparse.ArgumentParser(description="Serve a model with vLLM")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the model")
    parser.add_argument("--max-model-len", type=int, required=True, help="Maximum model length")
    parser.add_argument("--max-num-seqs", type=int, required=True, help="Maximum number of sequences")
    parser.add_argument("--n-gpu", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--cuda-visible-devices", type=str, default="0,1,2,3", 
                        help="CUDA visible devices")
    custom_args = parser.parse_args()

    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser

    serve_parser = FlexibleArgumentParser()
    serve_parser.add_argument("model_tag", type=str, help="The model tag to serve")
    serve_parser = make_arg_parser(serve_parser)

    args = serve_parser.parse_args([custom_args.model_path])
    args.model = custom_args.model_path
    args.served_model_name = [custom_args.model_name]
    args.max_model_len = custom_args.max_model_len
    args.max_num_seqs = custom_args.max_num_seqs
    args.max_seq_len_to_capture = custom_args.max_model_len
    args.tensor_parallel_size = custom_args.n_gpu
    args.enable_prefix_caching = True

    if is_on_kaggle_submission():
        args.disable_log_requests = True
        args.disable_log_stats = True

    # https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/560682#3113134
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    os.environ["CUDA_VISIBLE_DEVICES"] = custom_args.cuda_visible_devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    import uvloop
    from vllm.entrypoints.openai.api_server import run_server

    uvloop.run(run_server(args))