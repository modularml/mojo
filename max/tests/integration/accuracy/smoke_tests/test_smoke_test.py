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

from pytest import MonkeyPatch
from smoke_tests import smoke_test
from smoke_tests.smoke_test import MODEL_RECIPES


def _custom_recipe_keys() -> list[str]:
    return [key for key in MODEL_RECIPES if "__" in key]


def test_model_aliases_contain_exactly_one_double_underscore() -> None:
    for alias in _custom_recipe_keys():
        count = alias.count("__")
        assert count == 1, (
            f"Model alias {alias!r} must contain exactly one '__'"
            f" (found {count})"
        )


def test_all_model_recipes_load() -> None:
    for alias, recipe_path in MODEL_RECIPES.items():
        recipe = smoke_test._load_recipe(recipe_path)
        assert recipe.model.model_path is not None, alias


def test_hf_repos_for_model_includes_draft_model() -> None:
    """Recipes with draft_model expose both base and draft repos."""
    repos = smoke_test.hf_repos_for_model(
        "meta-llama/Llama-3.1-8B-Instruct__eagle"
    )
    assert "meta-llama/Llama-3.1-8B-Instruct" in repos
    assert "atomicapple0/EAGLE-LLaMA3.1-Instruct-8B" in repos


def test_hf_repos_for_model_prefers_recipe_casing() -> None:
    """A lowercased alias still resolves to the recipe's canonical casing.

    The cache is case-sensitive, so the prefetch script's offline probe
    only hits if the repo name matches the cached snapshot exactly. The
    helper adds the recipe-derived `model.model_path` before the alias
    prefix so the casefold dedup keeps the canonical casing.
    """
    repos = smoke_test.hf_repos_for_model(
        "meta-llama/llama-3.1-8b-instruct__eagle"
    )
    assert repos[0] == "meta-llama/Llama-3.1-8B-Instruct"


def test_model_aliases_lookup_is_case_insensitive() -> None:
    for key in MODEL_RECIPES:
        assert MODEL_RECIPES.get(key.lower()) is not None
        assert MODEL_RECIPES.get(key.upper()) is not None


def test_recipe_aliases_preserve_key_model_path_and_speculation() -> None:
    mtp_recipe = smoke_test._load_recipe(
        MODEL_RECIPES["nvidia/DeepSeek-V3.1-NVFP4__mtp"]
    )
    assert mtp_recipe.speculative is not None
    assert mtp_recipe.speculative.num_speculative_tokens == 3

    kimi_recipe = smoke_test._load_recipe(
        MODEL_RECIPES["nvidia/Kimi-K2.7-Code-NVFP4"]
    )
    assert kimi_recipe.model.model_path == "nvidia/Kimi-K2.7-Code-NVFP4"
    assert kimi_recipe.speculative is not None
    assert kimi_recipe.speculative.num_speculative_tokens == 3


def test_recipe_gpu_overrides_scale_matching_parallelism() -> None:
    recipe = smoke_test._load_recipe(
        "max/pipelines/architectures/deepseekV3/recipes/nvfp4_fp8kv_8x_b200.yaml"
    )

    args = smoke_test._recipe_gpu_overrides(recipe, gpu_count=4)

    assert args == [
        "--devices",
        "gpu:0,1,2,3",
        "--data-parallel-degree",
        "4",
        "--ep-size",
        "4",
    ]


def test_recipe_gpu_overrides_preserve_intentional_fixed_parallelism() -> None:
    recipe = smoke_test._load_recipe(
        "max/pipelines/architectures/deepseekV3/recipes/nvfp4_tpep_8x_b200.yaml"
    )

    args = smoke_test._recipe_gpu_overrides(recipe, gpu_count=4)

    assert args == ["--devices", "gpu:0,1,2,3", "--ep-size", "4"]


def test_recipe_gpu_overrides_preserve_single_gpu_recipes() -> None:
    recipe = smoke_test._load_recipe(
        "max/pipelines/architectures/llama3/recipes/llama31_8b_eagle.yaml"
    )

    args = smoke_test._recipe_gpu_overrides(recipe, gpu_count=4)

    assert args == []


def test_8x_recipe_auto_reduces_on_4_gpu_machine(
    monkeypatch: MonkeyPatch,
) -> None:
    """Regression: an 8-GPU recipe on a 4-GPU runner scales down to 4."""
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    # amd/Kimi-K2.7-Code-MXFP4 pins device_specs [0..7] and ep_size 8.
    cmd, _ = smoke_test.get_server_cmd(
        "max", "amd/Kimi-K2.7-Code-MXFP4", gpu_spec=("AMD MI355X", 4)
    )

    assert cmd[cmd.index("--devices") + 1] == "gpu:0,1,2,3"
    assert cmd[cmd.index("--ep-size") + 1] == "4"


def test_no_autoscale_devices_honors_recipe(monkeypatch: MonkeyPatch) -> None:
    """With autoscale off, the recipe's device_specs are left untouched."""
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    cmd, _ = smoke_test.get_server_cmd(
        "max",
        "amd/Kimi-K2.7-Code-MXFP4",
        autoscale_devices=False,
        gpu_spec=("AMD MI355X", 4),
    )

    assert "--devices" not in cmd
    assert "--ep-size" not in cmd


def test_vllm_minimax_keeps_flashinfer_workaround(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    # MiniMaxAI/MiniMax-M2.7 has no MODEL_RECIPES entry anymore (retired from
    # CI), but the "minimax-m2" vLLM workaround this test targets is specific
    # to that architecture family, so pass the still-present recipe directly.
    cmd, env = smoke_test.get_server_cmd(
        "vllm",
        "MiniMaxAI/MiniMax-M2.7",
        recipe_path="max/pipelines/architectures/minimax_m2/recipes/minimax_m2_8x_b200.yaml",
        gpu_spec=("NVIDIA B200", 8),
    )

    assert "--enable-expert-parallel" in cmd
    assert "--enable-chunked-prefill" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.8" in cmd
    assert "--data-parallel-size=8" in cmd
    assert "--attention-backend" in cmd
    assert "FLASH_ATTN" in cmd
    assert env["VLLM_USE_FLASHINFER_MOE_FP8"] == "0"


def test_vllm_uses_tp_for_recipe_default_data_parallel_degree(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    cmd, _ = smoke_test.get_server_cmd(
        "vllm",
        "nvidia/DeepSeek-V3.1-NVFP4__tpep",
        gpu_spec=("NVIDIA B200", 8),
    )

    assert "--enable-expert-parallel" in cmd
    assert "--tensor-parallel-size=8" in cmd
    assert "--data-parallel-size=8" not in cmd


def test_sglang_uses_tp_for_recipe_with_tensor_parallel_attention(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    cmd, _ = smoke_test.get_server_cmd(
        "sglang",
        "nvidia/DeepSeek-V3.1-NVFP4__tpep",
        gpu_spec=("NVIDIA B200", 8),
    )

    assert "--tp-size=8" in cmd
    assert "--expert-parallel-size" in cmd
    assert "8" in cmd
    assert "--mem-fraction-static" in cmd
    assert "0.8" in cmd
    assert "--data-parallel-size=8" not in cmd
    assert "--enable-dp-attention" not in cmd


def test_sglang_uses_data_parallel_attention_for_recipe_dp(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    cmd, _ = smoke_test.get_server_cmd(
        "sglang",
        "nvidia/DeepSeek-V3.1-NVFP4__fp8kv",
        gpu_spec=("NVIDIA B200", 8),
    )

    assert "--data-parallel-size=8" in cmd
    assert "--enable-dp-attention" in cmd
    assert "--expert-parallel-size" in cmd
    assert "8" in cmd
    assert "--mem-fraction-static" in cmd
    assert "0.8" in cmd
    assert "--tp-size=8" not in cmd


def test_sglang_uses_recipe_memory_cap(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    cmd, _ = smoke_test.get_server_cmd(
        "sglang",
        "meta-llama/Llama-3.1-8B-Instruct__dflash",
        gpu_spec=("NVIDIA B200", 8),
    )

    assert "--mem-fraction-static" in cmd
    assert "0.75" in cmd


def test_max_get_server_cmd_recipe_alias_resolves_yaml(
    monkeypatch: MonkeyPatch,
) -> None:
    """``get_server_cmd`` must load built-in recipe YAML without importing ``max``.

    Serve smoke CI runs the driver under ``uv run`` without the Modular package;
    this path used to fail with ``ModuleNotFoundError: max`` when resolving
    ``MODEL_RECIPES`` aliases.
    """
    monkeypatch.setattr(smoke_test, "_inside_bazel", lambda: False)

    alias = "microsoft/phi-4__modulev3"
    recipe_path = MODEL_RECIPES[alias]
    cmd, _ = smoke_test.get_server_cmd(
        "max", alias, gpu_spec=("NVIDIA L40S", 1)
    )

    assert cmd[:3] == [
        ".venv-serve/bin/max",
        "serve",
        "--pretty-print-config",
    ]
    assert "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "8000"
    assert "--config-file" in cmd
    cfg_idx = cmd.index("--config-file")
    assert cmd[cfg_idx + 1] == recipe_path
    assert "--trust-remote-code" not in cmd


def test_merge_serve_extra_args_appends_when_absent() -> None:
    args = ["prog", "model", "--framework", "max-ci"]
    merged = smoke_test.merge_serve_extra_args(
        args, "--kv-connector-config=tiered.json"
    )
    assert merged == args + [
        "--serve-extra-args",
        "--kv-connector-config=tiered.json",
    ]


def test_merge_serve_extra_args_splices_into_two_token_form() -> None:
    merged = smoke_test.merge_serve_extra_args(
        ["prog", "--serve-extra-args", "--max-batch-size=16"],
        "--kv-connector-config=tiered.json",
    )
    assert merged == [
        "prog",
        "--serve-extra-args",
        "--kv-connector-config=tiered.json --max-batch-size=16",
    ]


def test_merge_serve_extra_args_splices_into_equals_form() -> None:
    merged = smoke_test.merge_serve_extra_args(
        ["prog", "--serve-extra-args=--max-batch-size=16"],
        "--kv-connector-config=tiered.json",
    )
    assert merged == [
        "prog",
        "--serve-extra-args=--kv-connector-config=tiered.json --max-batch-size=16",
    ]


def test_merge_serve_extra_args_caller_value_goes_last_so_it_wins() -> None:
    """The caller's value lands after the merged-in one, so the serve CLI's
    last-wins parsing gives an explicit caller override of the same flag
    precedence."""
    merged = smoke_test.merge_serve_extra_args(
        ["prog", "--serve-extra-args", "--kv-connector-config=null.json"],
        "--kv-connector-config=tiered.json",
    )
    assert (
        merged[2]
        == "--kv-connector-config=tiered.json --kv-connector-config=null.json"
    )


def test_merge_serve_extra_args_does_not_mutate_input() -> None:
    args = ["prog", "--serve-extra-args", "--max-batch-size=16"]
    snapshot = list(args)
    smoke_test.merge_serve_extra_args(args, "--kv-connector-config=tiered.json")
    assert args == snapshot
