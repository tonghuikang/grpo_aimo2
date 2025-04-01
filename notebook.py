# %% [markdown] {"execution":{"iopub.execute_input":"2024-10-25T02:51:38.263408Z","iopub.status.busy":"2024-10-25T02:51:38.262446Z","iopub.status.idle":"2024-10-25T02:51:38.269929Z","shell.execute_reply":"2024-10-25T02:51:38.26859Z","shell.execute_reply.started":"2024-10-25T02:51:38.263362Z"},"papermill":{"duration":0.005188,"end_time":"2024-12-13T23:58:57.592616","exception":false,"start_time":"2024-12-13T23:58:57.587428","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false}}
# References
# - https://www.kaggle.com/code/mpware/vllm-0-7 for the current installation script
# - https://www.kaggle.com/code/richolson/ai-math-olympiad-qwen2-5-72b for showing how to submit
# - https://www.kaggle.com/code/abdullahmeda/load-72b-awq-model-using-vllm-on-l4-x4

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:22.443420Z","iopub.execute_input":"2025-03-30T09:16:22.443731Z","iopub.status.idle":"2025-03-30T09:16:22.577918Z","shell.execute_reply.started":"2025-03-30T09:16:22.443706Z","shell.execute_reply":"2025-03-30T09:16:22.577189Z"}}
import os
import glob
import shutil

for file in glob.glob("/kaggle/usr/lib/pip-install-aimo2/triton/backends/nvidia/bin/*"):
    shutil.copy(file, "/usr/local/cuda/bin/")
    os.chmod(f"/usr/local/cuda/bin/{os.path.basename(file)}", 0o755)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:22.578800Z","iopub.execute_input":"2025-03-30T09:16:22.579049Z","iopub.status.idle":"2025-03-30T09:16:22.584376Z","shell.execute_reply.started":"2025-03-30T09:16:22.579026Z","shell.execute_reply":"2025-03-30T09:16:22.583771Z"}}
import time
import pandas as pd
import polars as pl
import numpy as np

pd.set_option("display.max_colwidth", None)
start_time = time.time()
final_cutoff_time = start_time + (4 * 60 + 45) * 60  # 4.75 hours from start time
cutoff_times = [
    int(x) for x in np.linspace(final_cutoff_time, start_time + 4 * 60, 50 + 1)
]  # 4 minutes loading time at the start
cutoff_times.pop()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Configs

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:22.585507Z","iopub.execute_input":"2025-03-30T09:16:22.585716Z","iopub.status.idle":"2025-03-30T09:16:22.758115Z","shell.execute_reply.started":"2025-03-30T09:16:22.585697Z","shell.execute_reply":"2025-03-30T09:16:22.757408Z"}}
# Checklist for GPU commits - LLM_SERVER_URL_MAIN, INTERNET, ACCELERATOR

MODEL_PATHS: list[str] = [
    # will be overridden in Modal environment using MODEL_NAMES
    "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq-casperhansen/1",  # code
    "/kaggle/input/deepseek-r1/transformers/deepseek-r1-distill-qwen-7b-awq-casperhansen/1",  # math
    "/kaggle/input/m/deepseek-ai/deepseek-r1/transformers/deepseek-r1-distill-qwen-1.5b/1",  # classifier
]

MODEL_NAMES: list[str] = [
    "casperhansen/deepseek-r1-distill-qwen-32b-awq",
    "casperhansen/deepseek-r1-distill-qwen-7b-awq",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]


def is_on_modal() -> bool:
    return bool(os.getenv("MODAL_ENVIRONMENT"))


if is_on_modal():
    # Override MODEL_PATHS
    MODELS_DIR = "/aimo2"
    MODEL_PATHS: list[str] = [
        os.path.join(MODELS_DIR, model_name) for model_name in MODEL_NAMES
    ]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:21:29.298270Z","iopub.execute_input":"2025-03-30T09:21:29.298571Z","iopub.status.idle":"2025-03-30T09:21:29.303501Z","shell.execute_reply.started":"2025-03-30T09:21:29.298548Z","shell.execute_reply":"2025-03-30T09:21:29.302744Z"}}
# Deploy a single 32b model
MODEL_NAMES_PATHS_GPUS: list[tuple[str, str, list[int]]] = [
    (MODEL_NAMES[0], MODEL_PATHS[0], [0, 1, 2, 3]),
]
MATH_EXECUTION_COUNT = 8
CODE_EXECUTION_COUNT = 8
MAX_NUM_SEQS = 32

# # Deploy a single 7b model
# MODEL_NAMES_PATHS_GPUS: list[tuple[str, str, list[int]]] = [
#     (MODEL_NAMES[1], MODEL_PATHS[1], [0, 1, 2, 3]),
# ]

# # Deploy a single 2 x 7b model
# MODEL_NAMES_PATHS_GPUS: list[tuple[str, str, list[int]]] = [
#     (MODEL_NAMES[1], MODEL_PATHS[1], [0, 1]),
#     (MODEL_NAMES[1], MODEL_PATHS[1], [2, 3]),
# ]

# # Deploy a 4 x 7b model
# MODEL_NAMES_PATHS_GPUS: list[tuple[str, str, list[int]]] = [
#     (MODEL_NAMES[1], MODEL_PATHS[1], [0]),
#     (MODEL_NAMES[1], MODEL_PATHS[1], [1]),
#     (MODEL_NAMES[1], MODEL_PATHS[1], [2]),
#     (MODEL_NAMES[1], MODEL_PATHS[1], [3]),
# ]


MAX_MODEL_LEN = 8192 * 2
# MATH_EXECUTION_COUNT = 16
# CODE_EXECUTION_COUNT = 16
# MAX_NUM_SEQS = 32

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Environment

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:22.773296Z","iopub.execute_input":"2025-03-30T09:16:22.773510Z","iopub.status.idle":"2025-03-30T09:16:22.784514Z","shell.execute_reply.started":"2025-03-30T09:16:22.773491Z","shell.execute_reply":"2025-03-30T09:16:22.783929Z"}}
import os

# Possible environments
# - local
# - modal
# - Kaggle interactive
# - Kaggle commit
# - Kaggle competition (public and private)


def is_on_kaggle() -> bool:
    return bool(os.getenv("KAGGLE_KERNEL_RUN_TYPE")) or bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_interactive() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_commit() -> bool:
    return os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch" and not bool(
        os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    )


def is_on_kaggle_submission() -> bool:
    return bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))


if is_on_kaggle():
    REFERENCE_CSV = (
        "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
    )
elif is_on_modal():
    REFERENCE_CSV = "./reference.csv"
else:
    REFERENCE_CSV = "./reference.csv"


import pandas as pd

df_reference = pd.read_csv(REFERENCE_CSV)
if is_on_modal():
    df_reference.sample(min(len(df_reference), 1000), random_state=43)

    start_time = time.time()
    final_cutoff_time = start_time + len(df_reference) * 6 * 60
    cutoff_times = [
        int(x)
        for x in np.linspace(
            final_cutoff_time, start_time + 4 * 60, len(df_reference) + 1
        )
    ]  # 4 minutes loading time at the start
    cutoff_times.pop()


question_to_answer_map: dict[str, str] = dict(
    zip(df_reference["problem"], df_reference["answer"])
)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # vLLM Serving

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:21:45.999344Z","iopub.execute_input":"2025-03-30T09:21:45.999660Z","iopub.status.idle":"2025-03-30T09:21:46.245479Z","shell.execute_reply.started":"2025-03-30T09:21:45.999635Z","shell.execute_reply":"2025-03-30T09:21:46.244774Z"}}
import os
from typing import Optional

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import subprocess


def start_model(
    gpu_ids: list[int],
    model_name: str,
    model_path: str,
    max_num_seqs: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    logfile_suffix: str,
    port: int,
):

    command = f"""
    CUDA_VISIBLE_DEVICES={",".join(map(str, gpu_ids))} \
    python -m vllm.entrypoints.openai.api_server \
    --served-model-name {model_name} \
    --model {model_path} \
    --port {port} \
    --max_num_seqs {max_num_seqs} \
    --tensor-parallel-size {len(gpu_ids)} \
    --max-model-len {max_model_len} \
    --max-seq-len-to-capture {max_model_len} \
    --gpu-memory-utilization {gpu_memory_utilization} \
    """

    if is_on_kaggle_submission() or is_on_modal():
        command += " --disable-log-requests"

    if is_on_kaggle_submission():
        command += " --disable-log-stats"

    if is_on_kaggle():
        stdout_fd = open(f"vllm_serve-{logfile_suffix}.log", "w")
        stderr_fd = open(f"vllm_serve-{logfile_suffix}.err", "w")

        # Start the process without blocking, direct outputs to file
        process = subprocess.Popen(
            ["bash", "-c", command],
            stdout=stdout_fd,
            stderr=stderr_fd,
            text=True,
        )
    elif is_on_modal():
        # Start the process without blocking
        process = subprocess.Popen(
            ["bash", "-c", command],
            text=True,
        )
    else:
        raise ValueError

    return process


from typing import Callable
from openai import OpenAI, Stream
from openai.types import Completion

clients: list[OpenAI] = []
stream_funcs: list[Callable[[str, Optional[str]], Stream[Completion]]] = []
if is_on_kaggle() or is_on_modal():
    print("Starting vLLM servers")

    port = 8000
    for idx, (model_name, model_path, gpu_ids) in enumerate(MODEL_NAMES_PATHS_GPUS):
        process = start_model(
            gpu_ids=gpu_ids,
            model_path=model_path,
            model_name=model_name,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=0.90,
            logfile_suffix=f"{idx}",
            port=port,
        )
        client = OpenAI(base_url=f"http://0.0.0.0:{port}/v1", api_key="aimo")
        clients.append(client)

        # Create a closure that properly captures the current client and model_name
        def create_stream_func(
            client: OpenAI, model_name: str
        ) -> Callable[[str, Optional[str]], Stream[Completion]]:
            def stream_func(
                prompt: str, stop: Optional[str] = None
            ) -> Stream[Completion]:
                return client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=MAX_MODEL_LEN - count_tokens(prompt),
                    temperature=1.0,
                    stream=True,
                    stop=stop,
                )

            return stream_func

        stream_funcs.append(create_stream_func(client, model_name))

        port += 1
else:
    print("Using remote vLLM server")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:18:10.407757Z","iopub.execute_input":"2025-03-30T09:18:10.408095Z","iopub.status.idle":"2025-03-30T09:18:23.497753Z","shell.execute_reply.started":"2025-03-30T09:18:10.408067Z","shell.execute_reply":"2025-03-30T09:18:23.496931Z"}}
import os
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

# Loading the tokenizer when vLLM server is starting
# All models should be using the same tokenizer
if is_on_kaggle() or is_on_modal():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS[1],
        trust_remote_code=True,
    )
    print("Tokenizer loaded")
else:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAMES[1],
        trust_remote_code=True,
    )


def count_tokens(text: str) -> int:
    # Calling this might warn
    # "Token indices sequence length is longer than the specified maximum sequence length"
    # You can ignore the warning
    return len(tokenizer.encode(text))


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:07.933701Z","iopub.execute_input":"2025-03-30T09:31:07.934024Z","iopub.status.idle":"2025-03-30T09:31:07.950903Z","shell.execute_reply.started":"2025-03-30T09:31:07.933998Z","shell.execute_reply":"2025-03-30T09:31:07.950212Z"}}
import time
from openai import OpenAI, APIConnectionError

if is_on_kaggle() or is_on_modal():
    num_retries_to_load = 10 * 60
    while num_retries_to_load > 0:
        num_retries_to_load -= 1
        try:
            for client in clients:
                print(client.models.list())
            else:
                break
        except APIConnectionError as e:
            time.sleep(1)
            if num_retries_to_load == 0:
                if not is_on_kaggle_submission():
                    raise

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Helper functions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:18.765959Z","iopub.execute_input":"2025-03-30T09:31:18.766267Z","iopub.status.idle":"2025-03-30T09:31:18.802026Z","shell.execute_reply.started":"2025-03-30T09:31:18.766242Z","shell.execute_reply":"2025-03-30T09:31:18.801310Z"}}
# ----- Text Processing Functions -----


def extract_boxed_text(text: str) -> str:
    """Extract text inside \\boxed{} from LaTeX-formatted text"""
    import re

    pattern: str = r"oxed{(.*?)}"
    matches: list[str] = re.findall(pattern, text)
    if not matches:
        return ""
    for match in matches[::-1]:
        if match != "":
            return match
    return ""


def redact_sections(text: str) -> str:
    import re

    pattern = r"```python(.*?)```"
    text = re.sub(pattern, "[CODE REDACTED]", text, flags=re.DOTALL)
    pattern = r"```output(.*?)```"
    text = re.sub(pattern, "[OUTPUT REDACTED]", text, flags=re.DOTALL)
    # pattern = r"```<think>(.*?)</think>"
    # text = re.sub(pattern, "[THOUGHTS REDACTED]", text, flags=re.DOTALL)
    pattern = r"<thoughts>(.*?)</thoughts>"
    text = re.sub(pattern, "[THOUGHTS REDACTED]", text, flags=re.DOTALL)
    return text


def extract_sections(text: str) -> str:
    import re

    pattern = r"(```python.*?```|```output.*?```)"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    result = "\n\n".join(match.strip() for match in matches)
    lines = []
    for line in result.split("\n"):
        if line.strip().startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_code(text: str) -> str:
    """Extract Python code from Python code blocks"""
    import re

    pattern = r"```python\s*(.*?)\s*```"
    matches: list[str] = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return ""
    return matches[-1]


def does_added_string_complete_code(text: str, added_string: str) -> bool:
    """Check if adding a string to text would complete a code block"""
    if extract_code(text):
        return False
    if extract_code(text + added_string):
        return True
    return False


def get_incomplete_code_suffix(text: str) -> str:
    """Get the content after the last Python code block start marker"""
    delimiter = "```python\n"
    if delimiter not in text:
        return ""
    return text[text.index(delimiter) + len(delimiter) :]


def replace_last_occurance(string: str, old: str, new: str) -> str:
    """Replace the last occurrence of a substring in a string"""
    pos = string.rfind(old)
    if pos == -1:
        if not is_on_kaggle_submission():
            raise ValueError("Old substring not found")
        return string
    return string[:pos] + new + string[pos + len(old) :]


# ----- Code Processing Functions -----


code_prefix = """
import builtins
from sympy import *

if True:
    if True:
        # avoid getting processed by process_code
        def distance(a,b):
            return abs(a - b)
        print2_count = 20
        builtins_pow = builtins.pow

def pow(base, exp, mod=None):
    try:
        return builtins_pow(base, exp, mod)
    except:
        return builtins_pow(sympify(base), sympify(exp), sympify(mod))        

def truncate_line(line: str) -> str:
    if len(line) >= 1000:
        return line[:100] + f"[truncated {len(line) - 100 - 100} characters]" + line[-100:]
    return line

def print2(string: str):
    global print2_count
    print2_count -= 1
    if print2_count > 0:
        print(string)

import math
import numpy as np
import sympy as sp
import sys
sys.set_int_max_str_digits(100_000)
sys.setrecursionlimit(100_000)
""".strip()


def process_code(code: str) -> str:
    """Enhance Python code with imports and print statements for variables"""
    import keyword

    # Add import statements
    code = code_prefix + "\n\n" + code
    current_rows = code.strip().split("\n")
    new_rows = []
    for row in current_rows:
        new_rows.append(row)
        if "=" in row:
            if not row.startswith(" "):
                variables_to_print = row.split("=")[0].strip()
                if "(" in variables_to_print:
                    continue
                for variable_to_print in variables_to_print.split(","):
                    variable_to_print = variable_to_print.strip()
                    if variable_to_print.isidentifier() and not keyword.iskeyword(
                        variable_to_print
                    ):
                        if row.count("(") == row.count(")") and row.count(
                            "["
                        ) == row.count("]"):
                            new_rows.append(
                                f'\ntry:\n    print(f"{variable_to_print}={{truncate_line(str({variable_to_print}))}}")\nexcept:\n    pass\n'
                            )
            elif row.startswith(" " * 4) and not row.startswith(" " * 5):
                row = row[4:]
                indent = " " * 4
                variables_to_print = row.split("=")[0].strip()
                for variable_to_print in variables_to_print.split(","):
                    variable_to_print = variable_to_print.strip()
                    if variable_to_print.isidentifier() and not keyword.iskeyword(
                        variable_to_print
                    ):
                        if row.count("(") == row.count(")") and row.count(
                            "["
                        ) == row.count("]"):
                            new_rows.append(
                                f'\n{indent}try:\n{indent}    print2(f"{variable_to_print}={{truncate_line(str({variable_to_print}))}}")\n{indent}except:\n{indent}    pass\n'
                            )

    return "\n".join(new_rows)


# ----- Static Analysis Functions -----


def find_syntax_error(code_string: str) -> Optional[str]:
    """
    Parse Python code, identify syntax errors, and return a formatted string
    showing the code up to the error with error indicator.
    """
    import ast

    try:
        ast.parse(code_string)
        return None
    except SyntaxError as e:
        line_number = e.lineno
        assert line_number is not None
        lines = code_string.split("\n")
        error_line = lines[line_number - 1] if line_number - 1 < len(lines) else ""
        error_position = error_line.find("on the circle")

        output_lines = []
        for i in range(min(line_number, len(lines))):
            output_lines.append(lines[i])

        if error_position is not None:
            output_lines.append(" " * error_position + "^")
            output_lines.append("SyntaxError: invalid syntax")

        return "\n".join(output_lines)


import sympy
import ast


import builtins

builtin_names = set(dir(builtins))


def find_potential_name_errors(tree: ast.Module) -> list[str]:
    """
    Analyze Python AST to find potential NameError issues.
    Returns a list of variable names that might cause NameError.
    """

    # Track defined and used names
    # Note: does not work with import *
    defined_names = set(sympy.__all__)
    used_names = set()

    # Visitor to collect name information
    class NameVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                defined_names.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_names.add(node.id)
            self.generic_visit(node)

        def visit_Import(self, node):
            for name in node.names:
                if name.asname:
                    defined_names.add(name.asname)
                else:
                    defined_names.add(name.name.split(".")[0])
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for name in node.names:
                if name.asname:
                    defined_names.add(name.asname)
                else:
                    defined_names.add(name.name)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            defined_names.add(node.name)
            # Add function parameters as defined names
            for arg in node.args.args:
                defined_names.add(arg.arg)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            defined_names.add(node.name)
            self.generic_visit(node)

        def visit_For(self, node):
            if isinstance(node.target, ast.Name):
                defined_names.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined_names.add(elt.id)
            self.generic_visit(node)

    # Run the visitor
    visitor = NameVisitor()
    visitor.visit(tree)

    # Find potential NameErrors (used but not defined or built-in)
    potential_name_errors = used_names - defined_names - builtin_names

    return list(potential_name_errors)


def longest_valid_prefix(code_string: str) -> tuple[str, str, list[str]]:
    """
    Find the longest valid prefix of code that can be executed without errors.
    Returns the valid prefix, remaining suffix, and any name errors found.
    """
    import ast

    # returns prefix, suffix, previous_name_errors
    lines = code_string.split("\n")

    if lines:
        # ignore last line, might be incomplete
        lines.pop()

    previous_name_errors = []
    for i in range(len(lines), 0, -1):
        prefix = "\n".join(lines[:i])

        # Make sure the prefix ends with a newline
        if not prefix.endswith("\n"):
            prefix += "\n"

        try:
            # Try to parse the prefix as valid Python code
            tree = ast.parse(prefix)

            # Check for potential NameError issues
            name_errors = find_potential_name_errors(tree)

            # If there are no name errors, return this prefix
            if not name_errors:
                suffix = "\n".join(lines[i:])
                return prefix, suffix, previous_name_errors

            previous_name_errors = name_errors
            # Otherwise, continue looking for a shorter prefix
        except SyntaxError:
            # If there's a syntax error, try a shorter prefix
            continue

    return "", code_string, []  # Return empty string if no valid prefix found


# ----- Code Transformation Functions -----


def transform_code(code: str) -> str:
    """Transform integer literals to Integer() calls while preserving formatting and comments."""
    import ast

    # Step 1: Parse the code to identify the structure
    tree = ast.parse(code)

    # Step 2: Find all integer literals that should be transformed
    integers_to_transform = []
    integers_to_skip = []

    # Helper function to collect integers
    def collect_integers(node, skip=False):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            if (
                hasattr(node, "lineno")
                and hasattr(node, "col_offset")
                and hasattr(node, "end_col_offset")
            ):
                info = (node.lineno, node.col_offset, node.end_col_offset, node.value)
                if skip:
                    integers_to_skip.append(info)
                else:
                    integers_to_transform.append(info)

    # Visitor to find integers
    class IntegerVisitor(ast.NodeVisitor):
        def visit_Subscript(self, node):
            self.visit(node.value)
            # Skip integers in subscripts
            if isinstance(node.slice, ast.Constant) and isinstance(
                node.slice.value, int
            ):
                collect_integers(node.slice, skip=True)
            else:
                self.visit(node.slice)

        def visit_Call(self, node):
            self.visit(node.func)
            # Skip direct integer arguments in function calls
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    collect_integers(arg, skip=True)
                else:
                    self.visit(arg)
            for kw in node.keywords:
                self.visit(kw)

        def visit_Constant(self, node):
            collect_integers(node)

    visitor = IntegerVisitor()
    visitor.visit(tree)

    # Step 3: Filter out integers that should be skipped
    skip_positions = {(ln, col, end) for ln, col, end, _ in integers_to_skip}
    integers_to_transform = [
        (ln, col, end, val)
        for ln, col, end, val in integers_to_transform
        if (ln, col, end) not in skip_positions
    ]

    # Step 4: Transform the code line by line
    lines = code.splitlines()
    for i in range(len(lines)):
        line_number = i + 1

        # Get integers to transform in this line
        line_integers = sorted(
            [
                (col, end, val)
                for ln, col, end, val in integers_to_transform
                if ln == line_number
            ],
            key=lambda x: x[0],
            reverse=True,  # Process from right to left
        )

        # Apply transformations
        for col_start, col_end, _ in line_integers:
            byte_line = lines[i].encode("utf-8")
            # Extract the integer
            integer_str = byte_line[col_start:col_end]
            # Replace with Integer(...)
            byte_line = (
                byte_line[:col_start]
                + "Integer(".encode("utf-8")
                + integer_str
                + ")".encode("utf-8")
                + byte_line[col_end:]
            )
            lines[i] = byte_line.decode("utf-8")

    return "\n".join(lines)


# ----- Code Execution Function -----


import os
import threading

# if we don't lock, we are more likely to see code timeouts for code that should not timeout
excecute_locks = [threading.Lock() for _ in range(max(1, 2 * os.cpu_count() // 3))]
print("cpu_count", os.cpu_count(), len(excecute_locks))


def execute_code(
    code: str, timeout: int = 5, generation_idx: int = 0
) -> tuple[bool, str, str, str]:
    """
    Execute Python code and capture output.
    Returns: (is_successful, stdout, combined_output, valid_prefix)
    """
    import os
    import tempfile
    import subprocess

    execute_lock = excecute_locks[generation_idx % len(excecute_locks)]

    # is_successful, stdout, combined_output, valid_prefix
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "tmp.py")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            with execute_lock:
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            return (
                False,
                f"Execution timed out after {timeout} seconds.",
                f"Execution timed out after {timeout} seconds.",
                code,
            )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            return True, stdout, stdout, code
        else:
            # Process the error message to remove the temporary file path
            # This makes the error message cleaner and more user-friendly
            error_lines = stderr.split("\n")
            cleaned_errors = []
            valid_prefix = code
            for line in error_lines:
                if temp_file_path in line:
                    # Remove the path from the error line
                    line = line.replace(temp_file_path, "<temporary_file>")
                    line_marker_for_line_idx = 'File "<temporary_file>", line '
                    if line_marker_for_line_idx in line:
                        line_idx_string = line.split(",")[1].lstrip(
                            "line "
                        )  # should be ok
                        if line_idx_string.isdigit():
                            line_idx = int(line_idx_string)
                            valid_prefix = "\n".join(code.split("\n")[: line_idx - 1])
                cleaned_errors.append(line)
            cleaned_error_msg = "\n".join(cleaned_errors)

            # Include stdout in the error case
            combined_output = (
                f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
            )
            return False, stdout, combined_output, valid_prefix


def truncate_line(line: str) -> str:
    if len(line) >= 1000:
        return (
            line[:100] + f"[truncated {len(line) - 100 - 100} characters]" + line[-100:]
        )
    return line


def truncate_paragraph(output):
    lines = output.split("\n")
    lines_new = []
    for line in lines:
        line = truncate_line(line)
        lines_new.append(line)
    output_new = "\n".join(lines_new)
    if len(output_new) >= 7_500:
        output_new = (
            output_new[:2000]
            + f"[truncated {len(output_new) - 2000 - 2000} characters]"
            + output_new[-2000:]
        )
    return output_new


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Code worker

# %% [code] {"_kg_hide-input":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:23.952291Z","iopub.execute_input":"2025-03-30T09:31:23.952577Z","iopub.status.idle":"2025-03-30T09:31:23.972010Z","shell.execute_reply.started":"2025-03-30T09:31:23.952554Z","shell.execute_reply":"2025-03-30T09:31:23.971265Z"}}
# NOTE: <｜begin▁of▁sentence｜> is intentionally omitted - https://github.com/vllm-project/vllm/issues/12985
code_initial_prompt = """
<｜User｜>
Write a plan on how to solve this problem, and then write Python code that solves the problem.

The answer of the problem is expected to be an integer. We only need the answer modulo 1000.

You have access to a Python interpreter, with SymPy installed.

Do not solve this problem directly, only write the plan and then the Python code that solves the problem.

<problem>
{question}
</problem>

Write a plan to solve the above problem, and then the Python code that solves the problem.

Reminder - do not solve the problem directly, only write the plan, and then the Python code.
<｜Assistant｜><think>
I do not need to solve the problem, I will only need to write the plan, and then the Python code.

I should not be doing any calculations in my plan.

I will produce the plan.
</think>

Here is my plan, followed by the Python code. Please note that while I will provide the code that solves the problem as requested, I will not solve the problem.

Plan""".strip()


code_output_bridge = """<｜end▁of▁sentence｜><｜User｜>Your code was executed.

```output
{code_output}
```

Analyze issues with your code and output. If there are issues, fix the code so that the code solves the problem.

Reminder - do not solve the problem manually, the Python code should be the one doing calculations and solving the problem.

If you are confident of the answer, you will write your answer in \\boxed{{}}.
<｜Assistant｜><think>
Alright, I shall analyze issues with my code and output.

If the output is correct, I will print my answer in \\boxed{{}}.

If the output is wrong, I will fix the code.

Let me go through my code step by step.

Looking at my code,
""".strip()

from typing import Optional


def run_code_worker(
    question: str,
    generation_idx: int = 0,
    original_question: Optional[str] = None,
    defer_count: int = 0,
) -> str:
    """
    Execute the code worker logic directly as a function
    Returns the answer as a string and updates global code_results
    """
    global code_results
    global generation_logs
    import time

    if original_question is None:
        original_question = question

    question_start_time = time.time()

    # ----- Buffer Management -----
    prompt = code_initial_prompt.format(question=question)
    elapsed = code_initial_prompt.format(question=question)
    buffer = ""
    last_attempted_prompt = prompt
    generation_logs_local = []

    # Define buffer management functions
    def add_text_chunk(chunk: str, reset_buffer: bool = False) -> None:
        nonlocal prompt, buffer, elapsed
        prompt += chunk
        buffer += chunk
        if reset_buffer:
            elapsed += buffer
            buffer = ""

    def overwrite_buffer(new_buffer: str) -> None:
        nonlocal prompt, buffer, elapsed
        prompt = elapsed + new_buffer
        buffer = new_buffer

    # ----- Execution -----
    time_until_executing_again = time.time()
    completion_block_count = 2
    seen_stdout = ""
    flag_for_training = False
    request_counter = 0
    token_counter = 0

    while count_tokens(prompt) <= MAX_MODEL_LEN - 10:
        answer = extract_boxed_text(redact_sections(prompt))
        if answer and is_valid_answer_string(answer):
            break
        if original_question != current_question:
            break
        if request_counter > 20:
            break

        last_attempted_prompt = prompt

        if not flag_for_training:
            stream_func = stream_funcs[generation_idx % len(stream_funcs)]
            stream = stream_func(prompt)
        else:
            stream_func = stream_funcs[generation_idx % len(stream_funcs)]
            stream = stream_func(prompt, stop="```python")

        request_counter += 1
        generation_log = {
            "question": original_question,
            "method": "code",
            "generation_idx": generation_idx,
            "defer_count": defer_count,
            "elapsed": prompt,
            "flag_for_training": flag_for_training,
            "reason": "unassigned",
        }
        flag_for_training = False

        for chunk in stream:
            token_counter += 1
            chunk_text = chunk.choices[0].text

            if original_question != current_question:
                break
            blocking_from_completion = False
            added_string_complete_code = does_added_string_complete_code(
                buffer, chunk_text
            )
            if added_string_complete_code and completion_block_count > 0:
                blocking_from_completion = True
                completion_block_count -= 1

            if time.time() >= time_until_executing_again or blocking_from_completion:
                if incomplete_code := get_incomplete_code_suffix(buffer):
                    # static check is done for every token
                    # syntax check is only done 5 lines late (might have multi-line list definition)
                    # name error is checked earlier
                    valid_prefix, suffix, name_errors = longest_valid_prefix(
                        incomplete_code
                    )
                    if name_errors or suffix.count("\n") > 5 or "\n\n" in suffix:
                        define_name_errors_string = (
                            f"# Define symbols {' '.join(name_errors)}"
                            if name_errors
                            else ""
                        )
                        new_buffer = replace_last_occurance(
                            buffer,
                            incomplete_code,
                            valid_prefix + define_name_errors_string,
                        )
                        generation_log["prompt"] = prompt
                        generation_log["buffer"] = buffer
                        generation_log["new_buffer"] = new_buffer
                        generation_log["code"] = incomplete_code
                        generation_log["reason"] = "longest_valid_prefix"
                        overwrite_buffer(new_buffer)
                        break

                    # code execution check at 5 second intervals
                    is_successful, _, _, exec_valid_prefix = execute_code(
                        valid_prefix,
                        generation_idx=generation_idx,
                    )
                    for _ in range(5):
                        # possible that truncating prefix will cause the code to be invalid
                        # might need to keep truncating exec_valid_prefix
                        is_successful_in_retry, _, _, exec_valid_prefix = execute_code(
                            exec_valid_prefix,
                            generation_idx=generation_idx,
                        )
                        if is_successful_in_retry:
                            break
                    time_until_executing_again = time.time() + 5
                    if (
                        not is_successful
                        and valid_prefix.strip() != exec_valid_prefix.strip()
                    ):
                        try:
                            transformed_prefix = valid_prefix
                            transformed_prefix = transform_code(transformed_prefix)
                            transformed_prefix = process_code(transformed_prefix)
                        except:
                            transformed_prefix = valid_prefix
                        is_successful_transformed, code_output, _, _ = execute_code(
                            transformed_prefix,
                            generation_idx=generation_idx,
                        )
                        code_output = truncate_paragraph(code_output)
                        if is_successful_transformed:
                            # the code is actually okay if we convert to integers
                            continue

                        new_buffer = replace_last_occurance(
                            buffer,
                            incomplete_code,
                            exec_valid_prefix,
                        )
                        generation_log["prompt"] = prompt
                        generation_log["buffer"] = buffer
                        generation_log["new_buffer"] = new_buffer
                        generation_log["code"] = transformed_prefix
                        generation_log["reason"] = "intermediate_code_execution"
                        overwrite_buffer(new_buffer)

                        # inject results from intermediate values
                        _, suffix, _ = longest_valid_prefix(exec_valid_prefix)
                        # to account for multiline list definition
                        new_stdout = code_output.lstrip(seen_stdout)
                        if not suffix.strip() and new_stdout.strip():
                            seen_stdout = code_output
                            text_chunk_to_add = (
                                '\n\n"""\n# The code above was executed and these are the values:\n'
                                + new_stdout
                                + '\n"""\n\n'
                            )
                            generation_log["injected"] = text_chunk_to_add
                            add_text_chunk(text_chunk_to_add)
                        break

            add_text_chunk(chunk_text)

            if code := extract_code(buffer):
                completion_block_count = 2
                seen_stdout = ""
                try:
                    code = transform_code(code)
                    code = process_code(code)
                except:
                    pass
                is_successful, _, code_output, _ = execute_code(
                    code, generation_idx=generation_idx
                )
                code_output = truncate_paragraph(code_output)
                code_output_chunk = code_output_bridge.format(code_output=code_output)
                generation_log["prompt"] = prompt
                generation_log["buffer"] = buffer
                generation_log["code"] = code
                generation_log["reason"] = "terminal_code_execution"
                generation_log["injected"] = code_output_chunk
                flag_for_training = True
                add_text_chunk(code_output_chunk, reset_buffer=True)
                break

        else:
            generation_log["prompt"] = prompt
            generation_log["buffer"] = buffer
            generation_log["code"] = code
            generation_log["reason"] = "stop_token"

            if not extract_boxed_text(redact_sections(prompt)):
                # stop token returned without answer
                if generation_log["flag_for_training"]:
                    # if flag_for_training, if not submitting answer, assume all stop tokens is ```python
                    reset_chunk = "\n```python\n"
                    generation_log["injected"] = reset_chunk
                    add_text_chunk(reset_chunk, reset_buffer=False)
                else:
                    reset_chunk = "<｜end▁of▁sentence｜>" + math_initial_prompt.format(
                        question=question
                    )
                    generation_log["injected"] = reset_chunk
                    add_text_chunk(reset_chunk, reset_buffer=True)

        generation_log["timestamp"] = time.time() - question_start_time
        generation_log["request_counter"] = request_counter
        generation_log["token_counter"] = token_counter
        generation_logs_local.append(generation_log)
        stream.close()

    # ----- Submission -----
    answer = extract_boxed_text(redact_sections(prompt))

    generation_logs_local.append(
        {
            "question": original_question,
            "method": "code",
            "generation_idx": generation_idx,
            "defer_count": defer_count,
            "timestamp": time.time() - question_start_time,
            "prompt": prompt,
            "elapsed": last_attempted_prompt,
            "buffer": prompt[len(last_attempted_prompt) :],
            "request_counter": request_counter,
            "token_counter": token_counter,
            "flag_for_training": False,
            "reason": (
                "final" if original_question == current_question else "terminated"
            ),
        }
    )
    for generation_log in generation_logs_local:
        generation_log["eventual_answer"] = answer if answer != "" else -1
    for generation_log in generation_logs_local:
        generation_log["correct_answer"] = question_to_answer_map.get(
            original_question, ""
        )

    # Use thread lock when modifying shared dictionary
    with results_lock:
        code_results[original_question].append(answer)
        generation_logs[original_question].extend(generation_logs_local)

    if defer_count == 0:
        question_modified = original_question
        extracted_sections = extract_sections(prompt)
        if extracted_sections:
            question_modified += (
                "\n\nHere are some code inputs and outputs to refer to:\n\n"
                + extracted_sections
            )
        if answer:
            question_modified += f"\n\nHint: the answer might be: {answer}"
            question_modified += "\n\nYou could check whether the answer makes sense."
        run_math_worker(
            question_modified,
            generation_idx=generation_idx,
            original_question=original_question,
            defer_count=defer_count + 1,
        )

    return answer


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Math worker

# %% [code] {"_kg_hide-input":false,"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:26.356450Z","iopub.execute_input":"2025-03-30T09:31:26.356731Z","iopub.status.idle":"2025-03-30T09:31:26.365467Z","shell.execute_reply.started":"2025-03-30T09:31:26.356707Z","shell.execute_reply":"2025-03-30T09:31:26.364778Z"}}
# NOTE: <｜begin▁of▁sentence｜> is intentionally omitted - https://github.com/vllm-project/vllm/issues/12985
math_initial_prompt = """
<｜User｜>
{question}

Solve the above math problem. Please reason step by step. Only work with exact numbers. Only submit an answer if you are sure. After you get your final answer, take modulo 1000, and return the final answer within \\boxed{{}}.

<｜Assistant｜><think>
""".strip()


math_output_bridge = """<｜end▁of▁sentence｜><｜User｜>The answer is expected to be an integer.

After you get your final answer, take modulo 1000, and return the final answer within \\boxed{{}}.
<｜Assistant｜><think>"""


def run_math_worker(
    question: str,
    generation_idx: int = 0,
    original_question: Optional[str] = None,
    defer_count: int = 0,
) -> str:
    """
    Execute the math worker logic directly as a function
    Returns the answer as a string and updates global math_results
    """
    global math_results
    global generation_logs
    import time

    if original_question is None:
        original_question = question

    question_start_time = time.time()

    # ----- Math Execution -----
    prompt: str = math_initial_prompt.format(question=question)
    buffer: str = ""
    generation_logs_local = []
    answer = ""

    # Define a simpler processing function for the math worker
    def add_chunk(chunk_text: str):
        nonlocal prompt, buffer
        prompt += chunk_text
        buffer += chunk_text

    request_counter = 0
    token_counter = 0
    while count_tokens(prompt) <= MAX_MODEL_LEN - 10:
        if original_question != current_question:
            break

        if request_counter > 20:
            break

        request_counter += 1
        stream_func = stream_funcs[generation_idx % len(stream_funcs)]
        stream = stream_func(prompt)

        for chunk in stream:
            chunk_text = chunk.choices[0].text
            token_counter += 1

            add_chunk(chunk_text)

            # Check if we've found an answer after each chunk
            answer = extract_boxed_text(buffer)
            if answer:
                # no check for is_valid_answer_string here
                break

            if original_question != current_question:
                break

        # Log the state after each completion stream
        generation_logs_local.append(
            {
                "question": original_question,
                "method": "math",
                "generation_idx": generation_idx,
                "defer_count": defer_count,
                "timestamp": time.time() - question_start_time,
                "prompt": prompt,
                "buffer": buffer,
                "request_counter": request_counter,
                "token_counter": token_counter,
                "flag_for_training": False,
                "reason": "stream_complete",
            }
        )

        stream.close()
        if answer and is_valid_answer_string(answer):
            break

        prompt += math_output_bridge
        buffer = ""

    # ----- Final log entry -----
    generation_logs_local.append(
        {
            "question": original_question,
            "method": "math",
            "generation_idx": generation_idx,
            "defer_count": defer_count,
            "timestamp": time.time() - question_start_time,
            "prompt": prompt,
            "buffer": buffer,
            "request_counter": request_counter,
            "token_counter": token_counter,
            "flag_for_training": False,
            "reason": "final",
        }
    )

    # Add eventual_answer to all log entries
    for generation_log in generation_logs_local:
        generation_log["eventual_answer"] = answer if answer != "" else -1
    for generation_log in generation_logs_local:
        generation_log["correct_answer"] = question_to_answer_map.get(
            original_question, ""
        )

    # Use thread lock when modifying shared dictionary
    with results_lock:
        math_results[original_question].append(answer)
        generation_logs[original_question].extend(generation_logs_local)

    if defer_count == 0:
        question_modified = original_question
        if buffer:
            question_modified += f"\n\nHere are some thoughts to refer to \n\n<thoughts>\n{buffer}\n</thoughts>\n\n"
        if answer:
            question_modified += f"\n\nHint: the answer might be: {answer}"
            question_modified += (
                "\n\nYou write code to check whether the calculation is correct."
            )
        run_code_worker(
            question_modified,
            generation_idx=generation_idx,
            original_question=original_question,
            defer_count=defer_count + 1,
        )

    return answer


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Control logic

# %% [code] {"papermill":{"duration":0.016248,"end_time":"2024-12-14T00:03:20.989048","exception":false,"start_time":"2024-12-14T00:03:20.9728","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:27.969103Z","iopub.execute_input":"2025-03-30T09:31:27.969397Z","iopub.status.idle":"2025-03-30T09:31:27.985200Z","shell.execute_reply.started":"2025-03-30T09:31:27.969373Z","shell.execute_reply":"2025-03-30T09:31:27.984493Z"}}
from typing import Optional
from collections import defaultdict
import random
import threading


def has_early_answer(answers: list[str]) -> bool:
    counter: defaultdict[int, float] = defaultdict(float)
    for answer in answers:
        if is_valid_answer_string(answer):
            counter[int(answer)] += 1
    if not counter:
        return False
    counter[0] = (counter[0] + 1) // 2  # consider zero as sus
    frequencies = sorted(list(counter.values()) + [0, 0])
    highest_frequency = frequencies[-1]
    second_highest_frequency = frequencies[-2]
    third_highest_frequency = frequencies[-3]

    if is_on_modal():
        return highest_frequency >= CODE_EXECUTION_COUNT + MATH_EXECUTION_COUNT

    if (
        highest_frequency >= 2
        and second_highest_frequency <= 0
        and third_highest_frequency <= 0
    ):
        return True
    if highest_frequency >= 3 and second_highest_frequency <= 1:
        return True
    if highest_frequency >= 4 and second_highest_frequency <= 3:
        return True
    if highest_frequency >= 5:
        return True
    return False


def select_answer(answers: list[str]) -> Optional[int]:
    counter: defaultdict[int, float] = defaultdict(float)
    for answer in answers:
        try:
            if int(answer) == float(answer):
                counter[int(answer)] += (1 + random.random() / 1_000) * 1_000_000
        except Exception:
            pass
    if not counter:
        return None
    counter[0] = (counter[0] + 1) // 2
    _, answer_result = sorted([(v, k) for k, v in counter.items()], reverse=True)[0]
    return answer_result % 1000


def start_code_execution(question: str, generation_idx: int = 0):
    """Start a code worker execution in a separate thread"""
    thread = threading.Thread(target=run_code_worker, args=(question, generation_idx))
    thread.daemon = True  # Make thread a daemon so it exits when main program exits
    thread.start()
    return thread


def start_math_execution(question: str, generation_idx: int = 0):
    """Start a math worker execution in a separate thread"""
    thread = threading.Thread(target=run_math_worker, args=(question, generation_idx))
    thread.daemon = True  # Make thread a daemon so it exits when main program exits
    thread.start()
    return thread


def is_valid_answer_string(text: str) -> bool:
    try:
        if int(text) == float(text):
            return True
    except Exception:
        pass
    return False


current_question = ""
math_results: dict[str, list[str]] = {}
code_results: dict[str, list[str]] = {}
generation_logs: dict[str, list[dict]] = {}
results_lock = threading.Lock()  # Thread lock for protecting shared results


def save_generation_logs() -> None:
    generation_logs_all = []
    for generation_logs_for_question in generation_logs.values():
        generation_logs_all.extend(generation_logs_for_question)
    if generation_logs_all:
        df = pd.DataFrame(generation_logs_all)
        df = df.sort_values(
            ["question", "generation_idx", "defer_count", "timestamp", "method"]
        )
        df.to_csv(f"generation_logs.csv", index=False)
        df[df["reason"] == "final"].to_csv(f"generation_logs_final.csv", index=False)
        df[df["flag_for_training"]].to_csv(f"generation_logs_training.csv", index=False)


# %% [code] {"papermill":{"duration":0.016248,"end_time":"2024-12-14T00:03:20.989048","exception":false,"start_time":"2024-12-14T00:03:20.9728","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:27.969103Z","iopub.execute_input":"2025-03-30T09:31:27.969397Z","iopub.status.idle":"2025-03-30T09:31:27.985200Z","shell.execute_reply.started":"2025-03-30T09:31:27.969373Z","shell.execute_reply":"2025-03-30T09:31:27.984493Z"}}
def predict_for_question(question: str, id_: str = "placeholder_id") -> int:
    print(question)
    global math_results, code_results, generation_logs, current_question
    import time

    # Reset global result arrays
    with results_lock:
        math_results[question] = []
        code_results[question] = []
        generation_logs[question] = []
        current_question = question

    selected_questions_only: bool = True
    selected_questions_only: bool = False
    if selected_questions_only and not is_on_kaggle_submission():
        # if "Fred" not in question:
        #     return 210
        if "Triangle" not in question and "Fred" not in question:
            return 210
        # if "circumcircle" not in question:
        #     return 210
        # if "Triangle" not in question and "circumcircle" not in question:
        #     return 210

    if time.time() > final_cutoff_time:
        return 210

    threads = []
    generation_idx = 0
    for idx in range(max(MATH_EXECUTION_COUNT, CODE_EXECUTION_COUNT)):
        # I want to split the GPU usage
        if idx < MATH_EXECUTION_COUNT:
            generation_idx += 1
            thread = start_math_execution(question, generation_idx)
            threads.append(thread)

        if idx < CODE_EXECUTION_COUNT:
            generation_idx += 1
            thread = start_code_execution(question, generation_idx)
            threads.append(thread)

    while True:
        # Check for timeout
        if time.time() > cutoff_times[-1]:
            break

        time.sleep(20)

        # Make thread-safe copies of current results
        with results_lock:
            current_math_results = math_results[question].copy()
            current_code_results = code_results[question].copy()

        print("math", current_math_results)
        print("code", current_code_results)

        # Early termination if we have enough confident answers
        if has_early_answer(current_math_results):
            break
        if has_early_answer(current_code_results + current_math_results):
            break

        if (
            len(current_code_results) + len(current_math_results)
            == MATH_EXECUTION_COUNT + CODE_EXECUTION_COUNT
        ):
            # yes, even as we defer call from code to math and vice versa
            break

        print()
        if len(current_math_results) >= max(1, MATH_EXECUTION_COUNT // 2):
            # the code attempts might be stuck in a loop
            # do not wait for those, otherwise we could only rely on time
            break

    # Get thread-safe copies for final answer selection
    with results_lock:
        current_math_results = math_results[question].copy()
        current_code_results = code_results[question].copy()
        answer: Optional[int] = select_answer(
            current_math_results + current_code_results
        )
        if not is_on_kaggle_submission():
            save_generation_logs()

    print("final", answer)

    if answer is None:
        answer = 210

    cutoff_times.pop()
    return answer  # Note: Do NOT return early, we NEED to pop cutoff_times


# %% [code] {"papermill":{"duration":0.013768,"end_time":"2024-12-14T00:03:21.010372","exception":false,"start_time":"2024-12-14T00:03:20.996604","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:28.106796Z","iopub.execute_input":"2025-03-30T09:31:28.107074Z","iopub.status.idle":"2025-03-30T09:31:28.111496Z","shell.execute_reply.started":"2025-03-30T09:31:28.107051Z","shell.execute_reply":"2025-03-30T09:31:28.110840Z"}}
import pandas as pd
import polars as pl
from typing import Union


# Replace this function with your inference code.
# The function should return a single integer between 0 and 999, inclusive.
# Each prediction (except the very first) must be returned within 30 minutes of the question being provided.
def predict(
    id_object: pl.DataFrame, question_object: pl.DataFrame
) -> Union[pl.DataFrame, pd.DataFrame]:
    id_: int = id_object.item(0)
    print("------")
    print(id_)

    question: str = question_object.item(0)

    answer: int = predict_for_question(question)
    print("------\n\n\n")
    return pl.DataFrame({"id": id_, "answer": answer})


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Local tests

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:29.718790Z","iopub.execute_input":"2025-03-30T09:31:29.719113Z","iopub.status.idle":"2025-03-30T09:31:29.723175Z","shell.execute_reply.started":"2025-03-30T09:31:29.719088Z","shell.execute_reply":"2025-03-30T09:31:29.722450Z"}}
if is_on_kaggle_interactive():
    question_sample = "Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the smallest possible length of segment $CD$?"
    question_sample = "Fred and George take part in a tennis tournament with $4046$ other players. In each round, the players are paired into $2024$ matches. How many ways are there to arrange the first round such that Fred and George do not have to play each other? (Two arrangements for the first round are \textit{different} if there is a player with a different opponent in the two arrangements.)"
    with results_lock:
        current_question = question_sample
        math_results[question_sample] = []
        code_results[question_sample] = []
        generation_logs[question_sample] = []

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:31:29.858792Z","iopub.execute_input":"2025-03-30T09:31:29.859074Z","execution_failed":"2025-03-30T09:32:35.792Z"}}
if is_on_kaggle_interactive():
    answer = run_math_worker(question_sample)
    print(answer)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:23.619615Z","iopub.status.idle":"2025-03-30T09:16:23.619836Z","shell.execute_reply":"2025-03-30T09:16:23.619743Z"}}
if is_on_kaggle_interactive():
    answer = run_code_worker(question_sample)
    print(answer)

# %% [code] {"papermill":{"duration":0.012504,"end_time":"2024-12-14T00:03:21.030438","exception":false,"start_time":"2024-12-14T00:03:21.017934","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:23.620316Z","iopub.status.idle":"2025-03-30T09:16:23.620549Z","shell.execute_reply":"2025-03-30T09:16:23.620454Z"}}
if is_on_kaggle_interactive():
    predict_for_question(question_sample)

# %% [code] {"papermill":{"duration":0.012504,"end_time":"2024-12-14T00:03:21.030438","exception":false,"start_time":"2024-12-14T00:03:21.017934","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:23.620973Z","iopub.status.idle":"2025-03-30T09:16:23.621201Z","shell.execute_reply":"2025-03-30T09:16:23.621107Z"}}
if is_on_modal():
    for question in question_to_answer_map:
        predict_for_question(question)
        # break

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Prediction

# %% [code] {"papermill":{"duration":1644.24778,"end_time":"2024-12-14T00:30:45.363503","exception":false,"start_time":"2024-12-14T00:03:21.115723","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:23.621629Z","iopub.status.idle":"2025-03-30T09:16:23.621858Z","shell.execute_reply":"2025-03-30T09:16:23.621763Z"}}
if is_on_kaggle():
    import kaggle_evaluation.aimo_2_inference_server

    inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(
        predict
    )

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    else:
        pd.read_csv(
            "/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv"
        ).drop("answer", axis=1).to_csv("reference.csv", index=False)
        inference_server.run_local_gateway(
            (
                # '/kaggle/input/ai-mathematical-olympiad-progress-prize-2/test.csv',
                "reference.csv",
            )
        )

    # %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"papermill":{"duration":1644.24778,"end_time":"2024-12-14T00:30:45.363503","exception":false,"start_time":"2024-12-14T00:03:21.115723","status":"completed"},"tags":[],"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-03-30T09:16:23.621629Z","iopub.status.idle":"2025-03-30T09:16:23.621858Z","shell.execute_reply":"2025-03-30T09:16:23.621763Z"}}
import time

if (is_on_kaggle() or is_on_modal()) and not is_on_kaggle_submission():
    # Reset global result arrays
    with results_lock:
        current_question = "ENDED"

    time.sleep(20)  # allow sequences to terminate
    save_generation_logs()
