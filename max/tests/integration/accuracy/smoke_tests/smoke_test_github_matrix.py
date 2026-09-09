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

# /// script
# dependencies = ["click>=8,<9"]
# ///

import json
import re
from collections.abc import Mapping

import click

RUNNERS = {
    "B200": "modrunner-b200-efa",
    "MI355": "modrunner-mi355",
    "2xB200": "modrunner-b200-efa-2x",
    "2xMI355": "modrunner-mi355-2x",
    "4xMI355": "modrunner-mi355-4x",
    "8xB200": "modrunner-b200-efa-8x",
    "8xMI355": "modrunner-mi355-8x",
}

# Framework → GPUs that framework cannot run on.
HW_EX = {
    "vllm": {"MI355", "2xMI355", "4xMI355", "8xMI355"},
    "sglang": {"MI355", "2xMI355", "4xMI355", "8xMI355"},
}

# Tags: skip model on multi-GPU runners.
XL = {"8xB200", "4xMI355", "8xMI355"}
MULTI = {"2xB200", "2xMI355"} | XL
NON_XL = set(RUNNERS) - XL
DISABLE = set(RUNNERS)
B200_2X_ONLY = set(RUNNERS) - {"2xB200"}
B200_8X_ONLY = set(RUNNERS) - {"8xB200"}
# The AMD members of XL. A B200-only model excludes the whole set rather than
# naming one runner, so adding the next AMD runner is a change here and not a
# sweep over every entry that forgot to mention it.
AMD_XL = {"4xMI355", "8xMI355"}

# Model → set of exclusion tags:
#   - framework        (e.g. "max")
#   - gpu              (e.g. "MI355")
#   - framework@gpu    (e.g. "sglang@B200")
#   - use XL           to skip on 8xB200, 4xMI355 and 8xMI355
#   - use MULTI        to skip on all multi-GPU runners
#   - use NON_XL       to skip on everything except 8xB200, 4xMI355 and 8xMI355
#   - use DISABLE      to skip on all runners (temporarily disable a model)
#
# Custom recipe variants (CUSTOM_MODELS, any key with "__") are MAX only and are
# auto-excluded from vllm and sglang by excluded(); no per-entry tag needed.
#
# If you want to add a model to the smoke test:
#   1. Trigger the smoke test job with the model name you want to add:
#   https://github.com/modularml/modular/actions/workflows/pipelineVerification.yaml
#   2. Review the results, and the need for framework/GPU exclusions (if any)
#   3. Add the model to the dictionary below, with the appropriate exclusions
#    3a) For VLMs, add it to the is_vision_model check in smoke_test.py
#    3b) For reasoning models, add it to the is_reasoning_model check in smoke_test.py
# fmt: off
HF_MODELS: Mapping[str, set[str]] = {
    "allenai/Olmo-3-7B-Instruct": MULTI | {"max"},
    "allenai/olmOCR-2-7B-1025-FP8": MULTI | {"sglang"},
    "amd/Kimi-K2.7-Code-MXFP4": NON_XL | {"8xB200", "8xMI355"},
    "nvidia/Kimi-K2.7-Code-NVFP4": NON_XL | AMD_XL,
    # MODELS-1611: M3 is a private arch, so max-ci only.
    "amd/MiniMax-M3-MXFP4": NON_XL | {"8xB200", "8xMI355", "max"},
    "ByteDance-Seed/academic-ds-9B": MULTI | {"max", "max-ci", "sglang@B200", "vllm@B200"},  # SERVOPT-1120
    "deepseek-ai/DeepSeek-V2-Lite-Chat": MULTI | {"max", "max-ci", "vllm@B200"},  # SERVOPT-1120
    "deepseek-ai/DeepSeek-V3.1-Terminus": NON_XL | AMD_XL,
    "google/diffusiongemma-26B-A4B-it": MULTI | {"max", "max-ci"},
    "google/gemma-3-1b-it": MULTI | {"vllm@B200", "MI355"},  # TODO(KERN-3014)
    "google/gemma-3-27b-it": MULTI,
    "google/gemma-4-26B-A4B-it": MULTI,
    "google/gemma-4-31B-it": MULTI,
    "nvidia/Gemma-4-26B-A4B-NVFP4": MULTI | {"MI355"},
    "nvidia/diffusiongemma-26B-A4B-it-NVFP4": MULTI | {"max", "max-ci", "MI355"},
    "nvidia/Gemma-4-31B-IT-NVFP4": XL | {"MI355", "2xMI355"},
    "meta-llama/Llama-3.1-8B-Instruct": MULTI,
    "microsoft/Phi-3.5-mini-instruct": MULTI,
    "microsoft/phi-4": MULTI,
    # MODELS-1611: MXFP8 runs on 8xB200 and 8xMI355 -- the one model that wants
    # the AMD 8x runner, hence the literal 4xMI355 where its neighbours use
    # AMD_XL. max-ci exercises the private M3 arch; sglang serves the HF
    # checkpoint as a reference. vLLM and released MAX are excluded.
    # 8xB200 serves the production MTP recipe, which is the __mtp alias below;
    # sglang still runs the plain checkpoint there as the reference.
    "MiniMaxAI/MiniMax-M3-MXFP8": NON_XL
    | {"4xMI355", "max", "vllm", "max-ci@8xB200"},
    "modularai/MiniMax-M3-MXFP6": NON_XL | {"8xB200", "max"},
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": MULTI | {"vllm"},
    "modularai/Llama-3.1-405B-Instruct-autofp8": NON_XL | {"8xMI355", "max"},
    "nvidia/DeepSeek-V3.1-NVFP4": NON_XL | AMD_XL,
    "OpenGVLab/InternVL3_5-8B-Instruct": MULTI | {"max", "sglang"},
    "Qwen/Qwen2.5-7B-Instruct": MULTI,
    "Qwen/Qwen2.5-VL-7B-Instruct": MULTI,
    "Qwen/Qwen3-8B": MULTI,
    "Qwen/Qwen3-VL-4B-Instruct-FP8": XL | {"MI355", "2xMI355"},  # MI355: no FP8
    "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8": XL | {"MI355", "2xMI355", "max-ci@B200", "sglang@B200"},  # MI355: no FP8, B200: MODELS-1020
    "Qwen/Qwen3.5-9B": MULTI | {"max", "max-ci@MI355"},
    "Qwen/Qwen3.6-27B": MULTI | {"max", "max-ci@MI355"},
    "RedHatAI/gemma-3-27b-it-FP8-dynamic": MULTI,  # TODO(MODELS-1021)
    "nvidia/Llama-3.1-405B-Instruct-NVFP4": NON_XL | AMD_XL | {"max"},
    "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic": NON_XL | {"8xMI355"},
    "openai/gpt-oss-20b": XL | {"2xMI355"},
    "thinkingmachines/Inkling-Small-NVFP4": B200_2X_ONLY,
}

# Models tested with custom MAX recipe presets. MODEL_RECIPES in
# smoke_test.py maps each alias to its reusable recipe config.
CUSTOM_MODELS: Mapping[str, set[str]] = {
    "meta-llama/Llama-3.1-8B-Instruct__modulev3": MULTI,
    "google/gemma-3-27b-it__modulev3": XL,
    "microsoft/Phi-3.5-mini-instruct__modulev3": MULTI,
    "microsoft/phi-4__modulev3": MULTI,
    "deepseek-ai/DeepSeek-V2-Lite-Chat__modulev3": MULTI,
    "nvidia/DeepSeek-V3.1-NVFP4__fp8kv": NON_XL | AMD_XL,
    "nvidia/DeepSeek-V3.1-NVFP4__tpep": NON_XL | AMD_XL,
    "nvidia/DeepSeek-V3.1-NVFP4__tpep_ar": NON_XL | AMD_XL,
    "nvidia/DeepSeek-V3.1-NVFP4__tptp": NON_XL | AMD_XL,
    # TODO(SERVOPT-1168): Support multi-GPU eagle llama
    "meta-llama/Llama-3.1-8B-Instruct__eagle": MULTI,
    "meta-llama/Llama-3.1-8B-Instruct__dflash": MULTI,
    "nvidia/DeepSeek-V3.1-NVFP4__mtp": NON_XL | AMD_XL,
    "nvidia/DeepSeek-V3.1-NVFP4__mtp_tpep": NON_XL | AMD_XL,
    # The experimental_device_graph_synthesis flag is not in a released MAX yet, so
    # max-ci only. Synthesis is single-device (first cut), hence MULTI.
    "google/gemma-4-12B-it__device_graph_synthesis": MULTI | {"max"},
    # DSpark arch is not in a released MAX yet, so max-ci only.
    "google/gemma-4-12B-it__dspark": MULTI | {"max"},
    # Tuned recipes use an FP8 KV cache that does not support MI355.
    "google/gemma-4-26B-A4B-it__tuned": MULTI | {"MI355"},
    "google/gemma-4-31B-it__tuned": MULTI | {"MI355"},
    "nvidia/Gemma-4-26B-A4B-NVFP4__tuned": MULTI | {"MI355"},
    "nvidia/Gemma-4-31B-IT-NVFP4__tuned": MULTI | {"MI355"},
    "nvidia/Kimi-K2.7-Code-NVFP4__modulev3": NON_XL | AMD_XL,
    "meta-llama/Llama-3.1-8B-Instruct__rust_tiered_kvconnector": MULTI | {"MI355"},
    "nvidia/GLM-5.2-NVFP4__mtp_tpep": NON_XL | AMD_XL,
    "RadixArk/GLM-5.3-NVFP4__mtp_tpep": NON_XL | AMD_XL,
    "thinkingmachines/Inkling-Small-NVFP4__mtp": B200_2X_ONLY,
    "MiniMaxAI/MiniMax-M3-MXFP8__mtp": B200_8X_ONLY | {"max"},
}

# Aliases whose recipe ships with a private arch, so it cannot appear in
# MODEL_RECIPES: the private entrypoint resolves it from its own hw_recipes
# table instead.
PRIVATE_RECIPE_MODELS = frozenset({"MiniMaxAI/MiniMax-M3-MXFP8__mtp"})

MODELS: Mapping[str, set[str]] = {**HF_MODELS, **CUSTOM_MODELS}
# fmt: on

NIGHTLY_MODELS = frozenset(
    {
        "google/diffusiongemma-26B-A4B-it",
        "google/gemma-4-12B-it__dspark",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-26B-A4B-it__tuned",
        "google/gemma-4-31B-it",
        "google/gemma-4-31B-it__tuned",
        "nvidia/Gemma-4-26B-A4B-NVFP4",
        "nvidia/Gemma-4-26B-A4B-NVFP4__tuned",
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "nvidia/Gemma-4-31B-IT-NVFP4__tuned",
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "MiniMaxAI/MiniMax-M3-MXFP8",
        "MiniMaxAI/MiniMax-M3-MXFP8__mtp",
        "amd/MiniMax-M3-MXFP4",
        "modularai/MiniMax-M3-MXFP6",
        "nvidia/GLM-5.2-NVFP4__mtp_tpep",
        "amd/Kimi-K2.7-Code-MXFP4",
        "nvidia/Kimi-K2.7-Code-NVFP4",
        "thinkingmachines/Inkling-Small-NVFP4__mtp",
    }
)

TIERS = ("nightly", "all")


def tier_models(tier: str) -> list[str]:
    """Lists the models a tier schedules, before framework/GPU exclusions.

    Args:
        tier: Either ``"nightly"`` for the deployed families or ``"all"`` for
            every entry in ``MODELS``.

    Returns:
        The model keys the tier covers.
    """
    if tier == "all":
        return list(MODELS)
    return [model for model in MODELS if model in NIGHTLY_MODELS]


def excluded(framework: str, gpu: str, model: str) -> bool:
    """Check if a model is excluded from a given framework and/or GPU."""
    if gpu in HW_EX.get(framework, set()):
        return True
    # Custom MAX recipe variants are MAX only; no vLLM/SGLang equivalent.
    if model in CUSTOM_MODELS and framework in {"vllm", "sglang"}:
        return True
    tags = MODELS.get(model, set())
    return framework in tags or gpu in tags or f"{framework}@{gpu}" in tags


def parse_override(raw: str | None) -> list[str]:
    """Parse a comma-separated list of models from the command line."""
    if not raw:
        return []
    parts = re.split(r"[, \n\r]+", raw)
    return [p.strip() for p in parts if p.strip()]


@click.command()
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "max-ci", "max"]),
    required=True,
)
@click.option(
    "--models-override",
    default=None,
    help="Comma list of models; skips exclusions.",
)
@click.option(
    "--tier",
    type=click.Choice(TIERS),
    default="all",
    show_default=True,
    help="Model set: 'nightly' for the deployed families, 'all' for every model.",
)
@click.option("--run-on-b200", is_flag=True)
@click.option("--run-on-mi355", is_flag=True)
@click.option("--run-on-2xb200", is_flag=True)
@click.option("--run-on-2xmi355", is_flag=True)
@click.option("--run-on-4xmi355", is_flag=True)
@click.option("--run-on-8xb200", is_flag=True)
@click.option("--run-on-8xmi355", is_flag=True)
def main(
    framework: str,
    models_override: str | None,
    tier: str,
    run_on_b200: bool,
    run_on_mi355: bool,
    run_on_2xb200: bool,
    run_on_2xmi355: bool,
    run_on_4xmi355: bool,
    run_on_8xb200: bool,
    run_on_8xmi355: bool,
) -> None:
    flags = {
        "B200": run_on_b200,
        "MI355": run_on_mi355,
        "2xB200": run_on_2xb200,
        "2xMI355": run_on_2xmi355,
        "4xMI355": run_on_4xmi355,
        "8xB200": run_on_8xb200,
        "8xMI355": run_on_8xmi355,
    }
    gpus = [gpu for gpu, ok in flags.items() if ok]
    models = parse_override(models_override) or tier_models(tier)
    ignore_exclusions = models_override is not None

    job = []
    for gpu in sorted(gpus):
        for model in sorted(models):
            if ignore_exclusions or not excluded(framework, gpu, model):
                job.append(
                    {
                        "model": model,
                        "runs_on": RUNNERS[gpu],
                        "display_name": f"{gpu} - {model}",
                    }
                )

    print(json.dumps({"include": job}, indent=2))


if __name__ == "__main__":
    main()
