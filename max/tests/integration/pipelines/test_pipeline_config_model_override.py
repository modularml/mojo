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

"""Regression tests for CLI overrides reaching ``MAXModelConfig`` construction.

``test_main_override_revision_used_for_offline_cache_lookup``: under
``HF_HUB_OFFLINE`` the cache is keyed by revision, so any HF lookup that
uses the dataclass-default revision ``"main"`` will fail when only the
pinned SHA is pre-downloaded. ``main.*`` and ``draft.*`` overrides must
therefore reach ``MAXModelConfig`` construction before
``HuggingFaceRepo.__post_init__`` resolves the offline cache.
"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max.pipelines import PipelineArgs, PipelineConfig
from max.pipelines.lib import MAXModelConfig
from max.pipelines.lib.model_manifest import ModelManifest

GENERATE_LOCAL_PATH = "max.pipelines.weights.hf_utils.generate_local_model_path"
HF_OFFLINE = "huggingface_hub.constants.HF_HUB_OFFLINE"

PINNED_SHA = "abcdef123"


@pytest.fixture(autouse=True)
def _skip_repo_access_check() -> Iterator[None]:
    """Keep MAXModelConfig construction offline for placeholder repos:
    ``__init__`` eagerly builds the HuggingFace repo handles. Tests that
    stub ``generate_local_model_path`` themselves override this inner patch."""
    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            GENERATE_LOCAL_PATH,
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
    ):
        yield


def test_main_override_revision_used_for_offline_cache_lookup() -> None:
    revisions_seen: list[str] = []

    def stub(repo_id: str, revision: str) -> str:
        revisions_seen.append(revision)
        return f"/fake/cache/{repo_id}__{revision}"

    with patch(HF_OFFLINE, True), patch(GENERATE_LOCAL_PATH, side_effect=stub):
        # Construction may raise downstream once it tries to load weights
        # from the fake path; what we care about is which revisions were
        # asked for during HuggingFaceRepo construction.
        try:
            PipelineConfig.from_args(
                PipelineArgs.from_flat_kwargs(
                    model_path="some/repo",
                    model_override=[
                        f"main.huggingface_model_revision={PINNED_SHA}",
                        f"main.huggingface_weight_revision={PINNED_SHA}",
                    ],
                )
            )
        except Exception:
            pass

    assert set(revisions_seen) == {PINNED_SHA}, (
        f"Expected every offline lookup to use the pinned SHA "
        f"{PINNED_SHA!r}; got {revisions_seen}"
    )


def test_config_file_model_override_revision_reaches_construction() -> None:
    # A --config-file recipe supplies the main model as a dict (the ``model``
    # kwarg) and pins the revision separately via --model-override. The override
    # must be folded into the dict before MAXModelConfig is constructed, because
    # the constructor eagerly builds the HuggingFaceRepo and — under
    # HF_HUB_OFFLINE — a repo cached only at the pinned SHA has no ``main``
    # snapshot. Regression for the serve-smoke offline failure (MXF-517).
    revisions_seen: list[str] = []

    def stub(repo_id: str, revision: str) -> str:
        revisions_seen.append(revision)
        return f"/fake/cache/{repo_id}__{revision}"

    with patch(HF_OFFLINE, True), patch(GENERATE_LOCAL_PATH, side_effect=stub):
        try:
            PipelineConfig.from_args(
                PipelineArgs.from_flat_kwargs(
                    model={"model_path": "some/repo"},
                    model_override=[
                        f"main.huggingface_model_revision={PINNED_SHA}",
                        f"main.huggingface_weight_revision={PINNED_SHA}",
                    ],
                )
            )
        except Exception:
            pass

    assert revisions_seen, (
        "expected HuggingFaceRepo construction to resolve a revision offline"
    )
    assert set(revisions_seen) == {PINNED_SHA}, (
        f"Expected every offline lookup to use the pinned SHA "
        f"{PINNED_SHA!r}; got {revisions_seen}"
    )


def test_data_parallel_degree_cli_flag_reaches_config() -> None:
    with (
        patch(HF_OFFLINE, True),
        patch(
            GENERATE_LOCAL_PATH,
            return_value="/fake/cache/some/repo",
        ),
    ):
        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                model_path="some/repo",
                data_parallel_degree=4,
            )
        )

    assert config.model is not None
    assert config.model.data_parallel_degree == 4


def test_data_parallel_degree_cli_override_of_default_value_applied() -> None:
    # data_parallel_degree defaults to 1, so this exercises the merge into an
    # already-populated "main" model (e.g. loaded via --config-file): the
    # explicit override must win even though it matches the field default,
    # rather than being mistaken for "flag never passed".
    # "some/repo" is a placeholder; keep MAXModelConfig construction (which now
    # eagerly builds the repo handles and loads the HF config) offline.
    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch("huggingface_hub.file_exists", return_value=False),
        # architectures=None keeps the architecture name undeterminable, so
        # construction-time resolution skips instead of rejecting the
        # placeholder repo as an unknown architecture.
        patch(
            "max.pipelines.lib.config.model_config.load_huggingface_config",
            return_value=MagicMock(architectures=None),
        ),
    ):
        manifest = ModelManifest(
            {
                "main": MAXModelConfig(
                    model_path="some/repo", data_parallel_degree=8
                )
            }
        )

        config = PipelineConfig.from_args(
            PipelineArgs.from_flat_kwargs(
                models=manifest, data_parallel_degree=1
            )
        )

    assert config.model is not None
    assert config.model.data_parallel_degree == 1
