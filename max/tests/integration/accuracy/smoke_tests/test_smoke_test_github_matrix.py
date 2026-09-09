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

import json

from click.testing import CliRunner
from smoke_tests import smoke_test, smoke_test_github_matrix


def test_custom_model_keys_have_dunder() -> None:
    bad = [k for k in smoke_test_github_matrix.CUSTOM_MODELS if "__" not in k]
    assert not bad, (
        f"CUSTOM_MODELS keys must contain '__' to separate the base model "
        f"from the alias suffix: {bad}"
    )


def test_custom_models_defined_in_model_aliases() -> None:
    missing = [
        k
        for k in smoke_test_github_matrix.CUSTOM_MODELS
        if k not in smoke_test.MODEL_RECIPES
        and k not in smoke_test_github_matrix.PRIVATE_RECIPE_MODELS
    ]
    assert not missing, (
        f"CUSTOM_MODELS keys must have a corresponding entry in "
        f"smoke_test.MODEL_RECIPES: {missing}"
    )


def test_model_aliases_in_custom_models() -> None:
    missing = [
        k
        for k in smoke_test.MODEL_RECIPES
        if "__" in k and k not in smoke_test_github_matrix.CUSTOM_MODELS
    ]
    assert not missing, (
        f"Custom MODEL_RECIPES keys must have a corresponding entry in "
        f"smoke_test_github_matrix.CUSTOM_MODELS: {missing}"
    )


def test_nightly_models_are_known_models() -> None:
    unknown = sorted(
        smoke_test_github_matrix.NIGHTLY_MODELS
        - set(smoke_test_github_matrix.MODELS)
    )
    assert not unknown, (
        f"NIGHTLY_MODELS entries must also appear in MODELS: {unknown}"
    )


def test_nightly_tier_is_a_subset_of_all() -> None:
    """The nightly tier narrows the full matrix; it never adds to it."""
    for gpu in smoke_test_github_matrix.RUNNERS:
        flag = f"--run-on-{gpu.lower()}"
        tiers: dict[str, set[str]] = {}
        for tier in smoke_test_github_matrix.TIERS:
            result = CliRunner().invoke(
                smoke_test_github_matrix.main,
                ["--framework", "max-ci", "--tier", tier, flag],
            )
            assert result.exit_code == 0
            tiers[tier] = {
                job["model"] for job in json.loads(result.output)["include"]
            }
        assert tiers["nightly"] <= tiers["all"], gpu


def test_nightly_8xb200_pinned() -> None:
    """Pins the nightly 8xB200 set; widen it only deliberately."""
    result = CliRunner().invoke(
        smoke_test_github_matrix.main,
        ["--framework", "max-ci", "--tier", "nightly", "--run-on-8xb200"],
    )
    assert result.exit_code == 0
    scheduled = {job["model"] for job in json.loads(result.output)["include"]}
    assert scheduled == {
        "MiniMaxAI/MiniMax-M3-MXFP8__mtp",
        "nvidia/GLM-5.2-NVFP4__mtp_tpep",
        "nvidia/Kimi-K2.7-Code-NVFP4",
    }


def test_8xmi355_stays_opt_in() -> None:
    """8xMI355 nodes are scarce, so landing there must be a deliberate choice.

    Pinning the schedule means a model added without an 8xMI355 exclusion fails
    here instead of quietly consuming the runner. Widen the expectation only
    when a model is meant to run on it.
    """
    expected = {
        "max-ci": {
            "MiniMaxAI/MiniMax-M3-MXFP8",
            "modularai/MiniMax-M3-MXFP6",
        },
        "max": set(),
        "vllm": set(),
        "sglang": set(),
    }
    for framework, models in expected.items():
        result = CliRunner().invoke(
            smoke_test_github_matrix.main,
            ["--framework", framework, "--run-on-8xmi355"],
        )
        assert result.exit_code == 0
        scheduled = {
            job["model"] for job in json.loads(result.output)["include"]
        }
        assert scheduled == models, f"{framework} on 8xMI355"


def test_smoke_test_github_matrix_b200_max_ci() -> None:
    runner = CliRunner()
    result = runner.invoke(
        smoke_test_github_matrix.main,
        ["--framework", "max-ci", "--run-on-b200"],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert "include" in output
    assert len(output["include"]) > 0
