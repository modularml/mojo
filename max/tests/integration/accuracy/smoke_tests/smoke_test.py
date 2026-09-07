# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from __future__ import annotations

"""
This script is used for the CI "Max Serve Smoke Test" workflow.
It does two things:
    1. Starts the MAX/SGLang/VLLM inference server for the given model
    2. Runs a tiny evaluation task using against the chat/completions API

Currently there is a hard dependency that two virtualenvs are already created:
    - .venv-serve (not needed for max-ci, which uses bazel)
    - .venv-eval

Where the serve environment should already have either MAX/VLLM/SGLang installed.
The eval environment should already have lm-eval installed.
These dependencies are to be removed once this script
has been integrated into bazel.

Note that if you're running this script inside bazel, only available for max-ci,
then the virtualenvs are not needed.
"""

import json
import logging
import os
import shlex
import sys
from functools import cache
from pathlib import Path
from pprint import pformat
from typing import NamedTuple

import click
import yaml
from eval_runner import (
    TEXT_TASK,
    VISION_TASK,
    build_eval_summary,
    call_eval,
    get_gpu_name_and_count,
    print_samples,
    resolve_canonical_repo_id,
    safe_model_name,
    test_single_request,
    validate_hf_token,
    write_github_output,
    write_results,
)
from inference_server_harness import start_server
from pydantic import BaseModel, ConfigDict, Field
from requests.structures import CaseInsensitiveDict

URL = "http://127.0.0.1:8000/v1/chat/completions"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _metrics_url(framework: str) -> str:
    """Prometheus ``/metrics`` URL for the given framework (MAX uses port 8001, others 8000)."""
    port = 8001 if framework in ("max", "max-ci", "max-nightly") else 8000
    return f"http://127.0.0.1:{port}/metrics"


# Maps alias model names to reusable MAX recipe configs. Aliases let the same
# weights be tested under different configurations while keeping results
# separate in dashboards. Paths use the portable ``max/pipelines/architectures/``
# prefix (same convention as the MAX CLI config loader). The smoke driver
# resolves them from the checkout for YAML parsing; ``max serve`` still loads
# the same paths from the installed package in ``.venv-serve``.
#
# Values are fully spelled-out so they can be copy-pasted into a CLI invocation:
#   max serve --config-file max/pipelines/architectures/deepseekV3/recipes/nvfp4_8x_b200.yaml
# fmt: off
MODEL_RECIPES = CaseInsensitiveDict({
    "deepseek-ai/DeepSeek-V2-Lite-Chat__modulev3": "max/pipelines/architectures/deepseekV2_modulev3/recipes/deepseekv2_lite.yaml",
    "deepseek-ai/DeepSeek-V3.1-Terminus": "max/pipelines/architectures/deepseekV3/recipes/terminus_8x_b200.yaml",
    "google/gemma-4-12B-it__device_graph_synthesis": "max/pipelines/architectures/gemma4/recipes/gemma4_12b_device_graph_synthesis.yaml",
    "google/gemma-4-12B-it__dspark": "max/pipelines/architectures/gemma4/recipes/gemma4_12b_dspark.yaml",
    "google/gemma-4-26B-A4B-it__tuned": "max/pipelines/architectures/gemma4/recipes/gemma4_26b_a4b_tuned.yaml",
    "google/gemma-4-31B-it__tuned": "max/pipelines/architectures/gemma4/recipes/gemma4_31b_tuned.yaml",
    "nvidia/Gemma-4-26B-A4B-NVFP4__tuned": "max/pipelines/architectures/gemma4/recipes/gemma4_26b_a4b_nvfp4_tuned.yaml",
    "nvidia/Gemma-4-31B-IT-NVFP4__tuned": "max/pipelines/architectures/gemma4/recipes/gemma4_31b_nvfp4_tuned.yaml",
    "google/gemma-3-27b-it__modulev3": "max/pipelines/architectures/gemma3_modulev3/recipes/gemma3_27b.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__dflash": "max/pipelines/architectures/llama3/recipes/llama31_8b_dflash.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__eagle": "max/pipelines/architectures/llama3/recipes/llama31_8b_eagle.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__modulev3": "max/pipelines/architectures/llama3_modulev3/recipes/llama31_8b.yaml",
    "meta-llama/Llama-3.1-8B-Instruct__rust_tiered_kvconnector": "max/pipelines/architectures/llama3/recipes/llama31_8b_rust_tiered_kvconnector.yaml",
    "microsoft/Phi-3.5-mini-instruct__modulev3": "max/pipelines/architectures/phi3_modulev3/recipes/phi35_mini.yaml",
    "microsoft/phi-4__modulev3": "max/pipelines/architectures/phi3_modulev3/recipes/phi4.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__fp8kv": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_fp8kv_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__mtp": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_mtp_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__mtp_tpep": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_mtp_tpep_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__tpep": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_tpep_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__tpep_ar": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_tpep_ar_8x_b200.yaml",
    "nvidia/DeepSeek-V3.1-NVFP4__tptp": "max/pipelines/architectures/deepseekV3/recipes/nvfp4_tptp_8x_b200.yaml",
    "nvidia/GLM-5.2-NVFP4__mtp_tpep": "max/pipelines/architectures/glm5_1/recipes/glm_5_2_fp8_tp_ep_8x_b200_mtp.yaml",
    "RadixArk/GLM-5.3-NVFP4__mtp_tpep": "max/pipelines/architectures/glm5_1/recipes/glm_5_3_nvfp4_tp_ep_8x_b200.yaml",
    "amd/Kimi-K2.7-Code-MXFP4": "max/pipelines/architectures/kimik2_5/recipes/mxfp4_kimi_k2_7_code_8x_mi355.yaml",
    "nvidia/Kimi-K2.7-Code-NVFP4": "max/pipelines/architectures/kimik2_5/recipes/nvfp4_kimi_k2_7_code_eagle_tpep_8x_b200.yaml",
    "nvidia/Kimi-K2.7-Code-NVFP4__modulev3": "max/pipelines/architectures/kimik2_5_modulev3/recipes/nvfp4_kimi_k2_7_code_b200.yaml",
    "thinkingmachines/Inkling-Small-NVFP4__mtp": "max/pipelines/architectures/inkling/recipes/inkling_small_nvfp4_mtp.yaml",
})
# fmt: on


class RecipeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    class KVConnector(BaseModel):
        model_config = ConfigDict(extra="ignore")

        type: str = "null"

    class KVCache(BaseModel):
        model_config = ConfigDict(extra="ignore")

        device_memory_utilization: float | None = None
        kv_connector_config: RecipeConfig.KVConnector = Field(
            default_factory=lambda: RecipeConfig.KVConnector()
        )

    class Model(BaseModel):
        model_config = ConfigDict(extra="ignore")

        model_path: str | None = None
        served_model_name: str | None = None
        device_specs: list[int] | None = None
        data_parallel_degree: int = 1
        kv_cache: RecipeConfig.KVCache = Field(
            default_factory=lambda: RecipeConfig.KVCache()
        )

    class Runtime(BaseModel):
        model_config = ConfigDict(extra="ignore")

        ep_size: int | None = None
        enable_chunked_prefill: bool | None = None

    class Speculative(BaseModel):
        model_config = ConfigDict(extra="ignore")

        num_speculative_tokens: int | None = None

    model: Model = Field(default_factory=Model)
    draft_model: Model | None = None
    runtime: Runtime = Field(default_factory=Runtime)
    speculative: Speculative | None = None


class ServerCommand(NamedTuple):
    cmd: list[str]
    env: dict[str, str]


# TODO Refactor this to a model list/matrix specifying type of model
def is_vision_model(model: str) -> bool:
    """Check if the model supports vision tasks."""
    model = model.casefold()
    # gemma-4-12B is served text-only in MAX (gemma4_unified arch).
    if any(kw in model for kw in ("gemma-3-1b", "gemma-4-12b")):
        return False
    return any(
        kw in model
        for kw in (
            "gemma-3",
            "gemma-4",
            "idefics",
            "inkling",
            "internvl",
            "kimi-k2",
            "kimi-vl",
            "minimax-m3",
            "olmocr",
            "pixtral",
            "qwen2.5-vl",
            "qwen3-vl",
            "qwen3.5",
            "vision",
        )
    )


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


# Private entrypoints keep hw-keyed recipe tables next to their recipes (this
# OSS-synced file can't name them); the glob matches nothing in the OSS tree.
_HW_RECIPES_GLOB = "max_private/*/recipes/hw_recipes.yaml"


def _private_recipe_paths_for_model(model: str) -> list[str]:
    """Recipes a private entrypoint may serve for this model, on any GPU.

    Pre-fetching wants the union across hardware: every table entry for a
    model serves the same weights, so the union warms exactly the repos the
    entrypoint's own GPU-keyed selection will load.
    """
    for base in Path(__file__).resolve().parents:
        tables = sorted(base.glob(_HW_RECIPES_GLOB))
        if tables:
            break
    else:
        return []
    return [
        str(table.parent / entry["recipe"])
        for table in tables
        for entry in yaml.safe_load(table.read_text()).values()
        if entry["model"].lower() == model.lower()
    ]


def _resolve_recipe_path(recipe_path: str) -> str:
    """Resolve a recipe path to an absolute file path.
    Recipe paths use the ``max/pipelines/architectures/`` prefix and are
    resolved by the shared config resolver against the installed package.
    """
    if not recipe_path.startswith("max/pipelines/architectures/"):
        return recipe_path
    max_dir = Path(__file__).resolve().parents[4]
    resolved = max_dir / "python" / recipe_path
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Built-in recipe not found: {recipe_path} (resolved to {resolved})"
        )
    return str(resolved)


@cache
def _load_recipe(recipe_path: str) -> RecipeConfig:
    with open(_resolve_recipe_path(recipe_path), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RecipeConfig.model_validate(data)


def resolve_model_path(
    model: str, recipe_path: str | None = None
) -> tuple[str, str | None]:
    """Resolve a model alias to a canonical HF repo id and its recipe path."""
    if recipe_path is None:
        recipe_path = MODEL_RECIPES.get(model)
    if recipe_path:
        recipe_model_path = _load_recipe(recipe_path).model.model_path
        if recipe_model_path is None:
            raise ValueError("Recipe model section must contain model_path.")
        hf_model_path = recipe_model_path
    else:
        hf_model_path = model
    return resolve_canonical_repo_id(hf_model_path), recipe_path


def hf_repos_for_model(model: str) -> list[str]:
    """Return the repos to pre-cache for the given model.

    Always includes the base repo (alias prefix before __), plus the
    draft_model.model_path when the alias maps to a recipe with one.
    """
    repos: list[str] = []
    seen: set[str] = set()

    def add(repo: str) -> None:
        # Local filesystem paths can't be downloaded from HF.
        if repo.startswith(("/", "./", "../")):
            return
        key = repo.casefold()
        if key in seen:
            return
        seen.add(key)
        repos.append(repo)

    def add_recipe_repos(recipe_path: str) -> None:
        recipe = _load_recipe(recipe_path)
        if recipe.model.model_path:
            add(recipe.model.model_path)
        if recipe.draft_model and recipe.draft_model.model_path:
            add(recipe.draft_model.model_path)

    # Recipe-derived paths win the casefold dedup, so a lowercased alias
    # input still resolves to the canonical casing the cache expects.
    recipe_path = MODEL_RECIPES.get(model)
    if recipe_path is not None:
        add_recipe_repos(recipe_path)
    add(model.split("__", 1)[0])
    # A private recipe can serve a different repo than the alias; adding its
    # repos after the alias keeps the alias as the base repo callers get back.
    if recipe_path is None:
        for private_path in _private_recipe_paths_for_model(model):
            add_recipe_repos(private_path)
    return repos


def _recipe_gpu_overrides(recipe: RecipeConfig, gpu_count: int) -> list[str]:
    """Builds smoke-test GPU overrides for a fixed-GPU recipe.

    Recipes are reusable CLI presets with explicit device IDs and parallelism.
    Smoke tests run on several GPU counts, so fields that equal the recipe's
    device count are treated as scalable and overridden to match the local
    machine. Fixed one-GPU recipes and intentional non-scaled parallelism values
    are preserved.

    Note that this may be removed in the future when we implement
    flexible "auto" gpu args for max parallelism CLI and config.
    """
    if gpu_count <= 0:
        return []

    model_gpu_count = (
        len(recipe.model.device_specs)
        if recipe.model.device_specs is not None
        else None
    )
    draft_gpu_count = (
        len(recipe.draft_model.device_specs)
        if recipe.draft_model is not None
        and recipe.draft_model.device_specs is not None
        else None
    )
    devices_arg = f"gpu:{','.join(str(i) for i in range(gpu_count))}"

    args = []
    if model_gpu_count != 1:
        args += ["--devices", devices_arg]
    if model_gpu_count is not None and model_gpu_count != 1:
        if recipe.model.data_parallel_degree == model_gpu_count:
            args += ["--data-parallel-degree", str(gpu_count)]
        if recipe.runtime.ep_size == model_gpu_count:
            args += ["--ep-size", str(gpu_count)]

    if recipe.draft_model is not None:
        if draft_gpu_count != 1:
            args += ["--draft-devices", devices_arg]
        if (
            draft_gpu_count is not None
            and draft_gpu_count != 1
            and recipe.draft_model.data_parallel_degree == draft_gpu_count
        ):
            args += ["--draft-data-parallel-degree", str(gpu_count)]

    return args


def merge_serve_extra_args(args: list[str], extra: str) -> list[str]:
    """Returns a copy of *args* with *extra* folded into ``--serve-extra-args``.

    ``--serve-extra-args`` is a single-value option, so appending a second
    occurrence would silently drop one of the two values. *extra* is instead
    spliced in front of any caller-supplied value (later serve flags win, so
    an explicit caller override of the same flag takes precedence); when the
    caller passed no ``--serve-extra-args``, the flag is appended.
    """
    merged = list(args)
    for i, arg in enumerate(merged):
        if arg == "--serve-extra-args" and i + 1 < len(merged):
            merged[i + 1] = f"{extra} {merged[i + 1]}"
            return merged
        if arg.startswith("--serve-extra-args="):
            merged[i] = f"--serve-extra-args={extra} {arg.split('=', 1)[1]}"
            return merged
    return merged + ["--serve-extra-args", extra]


def get_server_cmd(
    framework: str,
    model: str,
    *,
    serve_extra_args: str = "",
    recipe_path: str | None = None,
    autoscale_devices: bool = True,
    gpu_spec: tuple[str, int],
) -> ServerCommand:
    gpu_model, gpu_count = gpu_spec
    env: dict[str, str] = {}
    if recipe_path is None:
        recipe_path = MODEL_RECIPES.get(model)
    recipe = _load_recipe(recipe_path) if recipe_path else None
    recipe_config: tuple[str, RecipeConfig] | None = None
    if (
        recipe is not None
        and recipe_path is not None
        and framework in ["max-ci", "max"]
    ):
        recipe_config = (recipe_path, recipe)

    sglang_backend = "triton" if "b200" in gpu_model.lower() else "fa3"
    SGLANG = [
        "sglang.launch_server",
        "--attention-backend",
        sglang_backend,
        "--enable-metrics",
    ]
    # limit-mm-per-prompt.video is for InternVL3 on B200
    VLLM = [
        "vllm.entrypoints.openai.api_server",
        "--max-model-len",
        "auto",
        "--limit-mm-per-prompt.video",
        "0",
        "--limit-mm-per-prompt.audio",
        "0",
    ]
    # vLLM's KV cache sizing misses Inkling's mamba conv cache and OOMs.
    if "inkling" in model.casefold():
        VLLM += ["--gpu-memory-utilization", "0.8"]
    MAX = ["serve", "--pretty-print-config"]

    if gpu_count > 1:
        if recipe is not None:
            if (
                recipe.runtime.ep_size is not None
                and recipe.runtime.ep_size > 1
            ):
                VLLM += ["--enable-expert-parallel"]
                SGLANG += ["--expert-parallel-size", str(gpu_count)]

            if recipe.runtime.enable_chunked_prefill is not False:
                VLLM += ["--enable-chunked-prefill"]
            else:
                SGLANG += ["--chunked-prefill-size", "-1"]

            if recipe.model.kv_cache.device_memory_utilization is not None:
                mem_cap = recipe.model.kv_cache.device_memory_utilization
                VLLM += [
                    "--gpu-memory-utilization",
                    f"{mem_cap:g}",
                ]
                SGLANG += ["--mem-fraction-static", f"{mem_cap:g}"]

            if recipe.model.data_parallel_degree > 1:
                VLLM += [f"--data-parallel-size={gpu_count}"]
                SGLANG += [
                    f"--data-parallel-size={gpu_count}",
                    "--enable-dp-attention",
                ]
            else:
                VLLM += [f"--tensor-parallel-size={gpu_count}"]
                SGLANG += [f"--tp-size={gpu_count}"]

            # Remove once vLLM >= 0.17 (which includes vllm-project/vllm#34673).
            if "minimax-m2" in model.casefold():
                env["VLLM_USE_FLASHINFER_MOE_FP8"] = "0"
                VLLM += ["--attention-backend", "FLASH_ATTN"]

        else:  # gpu_count > 1 and recipe is None
            MAX += [
                "--devices",
                f"gpu:{','.join(str(i) for i in range(gpu_count))}",
            ]
            SGLANG += [f"--tp-size={gpu_count}"]
            VLLM += [f"--tensor-parallel-size={gpu_count}"]

    if framework in ("max", "max-ci"):
        # Force MAX to rely solely on the KVConnector for prefix cache hits to test
        # cpu/disk KV offload code paths.
        if "--kv-connector-config" in serve_extra_args or (
            recipe is not None
            and recipe.model.kv_cache.kv_connector_config.type != "null"
        ):
            env["MODULAR_ONLY_USE_KV_CONNECTOR_LAST_LEVEL_CACHE"] = "1"

    if _inside_bazel():
        assert framework == "max-ci", "bazel invocation only supports max-ci"
        cmd = [sys.executable, "-m", "max._entrypoints.pipelines", *MAX]
    else:
        assert framework != "max-ci", "max-ci must be run through bazel"
        interpreter = [".venv-serve/bin/python", "-m"]
        commands = {
            "sglang": [*interpreter, *SGLANG],
            "vllm": [*interpreter, *VLLM],
            "max": [".venv-serve/bin/max", *MAX],
        }
        cmd = commands[framework]

    cmd = cmd + ["--port", "8000"]
    if recipe_config is not None:
        config_file_path, recipe = recipe_config
        cmd += ["--config-file", config_file_path]
        if autoscale_devices:
            cmd += _recipe_gpu_overrides(recipe, gpu_count)
    else:
        cmd += ["--trust-remote-code", "--model", model]

    # GPT-OSS uses repetition_penalty in lm_eval to prevent reasoning loops,
    # so we need to enable penalties on the server
    if "gpt-oss" in model.casefold() and framework in ["max-ci", "max"]:
        cmd += ["--enable-penalties"]

    if serve_extra_args:
        if framework in ["max-ci", "max"]:
            cmd += shlex.split(serve_extra_args)
        else:
            logger.warning(
                "Ignoring --serve-extra-args for framework %s", framework
            )
    return ServerCommand(cmd, env)


# Verified stock lm-eval tasks runnable unmodified (qa4+ intentionally excluded).
VALID_STOCK_TASKS = {"babilong_qa1", "babilong_qa2", "babilong_qa3"}


def valid_tasks() -> set[str]:
    """Return the allowlist of task names accepted by ``--override-tasks``.

    Combines the task names of our mirrored ``tasks/`` yamls (the ``task:``
    field, parsed line-by-line rather than with ``yaml.safe_load`` since some
    configs use lm-eval's ``!function`` tag) with the verified stock lm-eval
    long-context tasks. Restricting to this set keeps arbitrary lm-eval tasks
    out of the smoke test.
    """
    tasks_dir = Path(__file__).resolve().parent / "tasks"
    names: set[str] = set(VALID_STOCK_TASKS)
    for yaml_path in tasks_dir.glob("**/*.yaml"):
        for line in yaml_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("task:"):
                names.add(line.split(":", 1)[1].strip())
                break
    return names


@click.command()
@click.argument(
    "hf_model_path",
    type=str,
    required=True,
)
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max", "max-ci"]),
    default="max-ci",
    required=False,
    help="Framework to use for the smoke test. Only max-ci is supported when running in bazel.",
)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, writable=True, path_type=Path),
    default=None,
    help="If provided, a summary json file and the eval result are written here",
)
@click.option(
    "--print-responses",
    is_flag=True,
    default=False,
    help="Print question/response pairs from eval samples after the run finishes",
)
@click.option(
    "--print-cot",
    is_flag=True,
    default=False,
    help="Print the model's chain-of-thought reasoning for each sample. Must be used with --print-responses",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=64,
    help="Maximum concurrent requests to send to the server",
)
@click.option(
    "--num-questions",
    type=int,
    default=320,
    help="Number of questions to ask the model",
)
@click.option(
    "--serve-extra-args",
    type=str,
    default="",
    help=(
        "Extra args appended to MAX serve command, for example: "
        '"--device-graph-capture --max-batch-size=16"'
    ),
)
@click.option(
    "--disable-timeouts",
    is_flag=True,
    default=False,
    help="Disable all timeouts. Useful when debugging hangs.",
)
@click.option(
    "--recipe-path",
    type=str,
    default=None,
    help="Recipe config YAML to serve instead of one looked up by model name.",
)
@click.option(
    "--autoscale-devices/--no-autoscale-devices",
    default=True,
    help="Scale a recipe's device count to the local machine's GPU count.",
)
@click.option(
    "--override-tasks",
    "override_tasks",
    multiple=True,
    type=click.Choice(sorted(valid_tasks())),
    help=(
        "Run these eval task(s) instead of the model-derived defaults. "
        "Repeatable (e.g. --override-tasks babilong_qa1 --override-tasks "
        "babilong_qa3). Restricted to the verified task set; click validates "
        "against it. When set, the TEXT/VISION default selection is bypassed."
    ),
)
@click.option(
    "--lm-eval-metadata",
    "lm_eval_metadata",
    type=str,
    default=None,
    help=(
        "JSON passed verbatim to lm-eval's --metadata, merged into each task's "
        "config to parameterize it at runtime. For example, "
        '\'{"max_seq_lengths": "16k"}\' sets the babilong context length.'
    ),
)
def smoke_test(
    hf_model_path: str,
    framework: str,
    output_path: Path | None,
    print_responses: bool,
    print_cot: bool,
    max_concurrent: int,
    num_questions: int,
    serve_extra_args: str,
    disable_timeouts: bool,
    recipe_path: str | None,
    autoscale_devices: bool,
    override_tasks: tuple[str, ...],
    lm_eval_metadata: str | None,
) -> None:
    """
    Example usage: ./bazelw run smoke-test -- meta-llama/Llama-3.2-1B-Instruct

    This command asks 320 questions against the model behind the given hf_model_path.
    It runs the provided framework (defaulting to MAX serve) in the background,
    and fires off HTTP requests to chat/completions API.
    Note: Only models with a chat template (typically -instruct, -it, -chat, etc.) are supported.

    Accuracy is reported at the end, with higher values being better.
    A 1.0 value means 100% accuracy.

    """
    validate_hf_token()

    if print_cot and not print_responses:
        raise ValueError("--print-cot must be used with --print-responses")

    build_workspace = os.getenv("BUILD_WORKSPACE_DIRECTORY")
    if output_path and build_workspace and not output_path.is_absolute():
        output_path = Path(build_workspace) / output_path

    model = hf_model_path.strip()
    result_dir = None
    if output_path is not None:
        result_dir = output_path / safe_model_name(model)
        result_dir.mkdir(parents=True, exist_ok=True)

    hf_model_path, recipe_path = resolve_model_path(model, recipe_path)
    # A recipe can serve its weights under another name; requests must use it.
    served = hf_model_path
    if recipe_path:
        served = _load_recipe(recipe_path).model.served_model_name or served
    cmd, server_env = get_server_cmd(
        framework,
        hf_model_path,
        serve_extra_args=serve_extra_args,
        recipe_path=recipe_path,
        autoscale_devices=autoscale_devices,
        gpu_spec=get_gpu_name_and_count(),
    )

    if override_tasks:
        tasks = list(override_tasks)
    elif is_vision_model(model):
        tasks = [VISION_TASK, TEXT_TASK]
    else:
        tasks = [TEXT_TASK]

    logger.info(f"Starting server with command:\n {' '.join(cmd)}")
    results = []
    all_samples = []
    if disable_timeouts:
        timeout = sys.maxsize
    else:
        # TODO(GEX-3508): Reduce timeout once model build time is optimized
        timeout = 2700

    metrics_url = _metrics_url(framework)
    with start_server(cmd, timeout, env_overrides=server_env) as server:
        logger.info(f"Server started in {server.startup_time:.2f} seconds")
        write_github_output("startup_time", f"{server.startup_time:.2f}")

        for task in tasks:
            test_single_request(
                URL, served, task, disable_timeouts=disable_timeouts
            )
            result, samples = call_eval(
                URL,
                served,
                task,
                max_concurrent=max_concurrent,
                num_questions=num_questions,
                disable_timeouts=disable_timeouts,
                metrics_url=metrics_url,
                model_alias=model if served != model else None,
                lm_eval_metadata=lm_eval_metadata,
            )

            if print_responses:
                print_samples(samples, print_cot)

            results.append(result)
            all_samples.append(samples)

    if results:
        summary = build_eval_summary(
            results, startup_time_seconds=server.startup_time
        )

        if result_dir is not None:
            write_results(result_dir, summary, results, all_samples, tasks)

        logger.info(pformat(summary, indent=2))

    if result_dir is not None:
        (result_dir / "smoke_status.json").write_text(
            json.dumps({"status": "FINISHED_OK"}) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    smoke_test()
