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

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import max.tests.integration.tools.debugging_utils as dbg
import pytest
import transformers
from max.experimental.nn import Module as ModuleV3
from max.nn.layer import Module
from max.pipelines.lib.config.model_config import MAXModelConfig


@pytest.fixture(autouse=True)
def _skip_repo_access_check() -> Iterator[None]:
    """These tests build ``MAXModelConfig`` with placeholder repos.
    ``__init__`` eagerly builds the ``HuggingFaceRepo`` handles (running the HF
    existence check) and eagerly loads the HF config; no-op both so
    construction stays offline. These tests set ``huggingface_config``
    themselves, so the mocked load value is unused."""
    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
        patch("huggingface_hub.file_exists", return_value=False),
        patch(
            "max.pipelines.lib.config.model_config.load_huggingface_config",
            return_value=MagicMock(),
        ),
    ):
        yield


def test_apply_v3_hooks() -> None:
    """Test apply_v3_hooks patches and restores ModuleV3.__call__ and .compile."""
    orig_call = ModuleV3.__call__
    orig_compile = ModuleV3.compile
    with dbg.apply_max_hooks(output_directory=None) as hook:
        with dbg.apply_v3_hooks(hook):
            assert ModuleV3.__call__ is not orig_call
            assert ModuleV3.compile is not orig_compile
    assert ModuleV3.__call__ is orig_call
    assert ModuleV3.compile is orig_compile


def test_apply_max_hooks_without_output_dir() -> None:
    """Test apply_max_hooks creates and cleans up hook without output directory."""
    orig_infer_init = dbg.InferenceSession.__init__
    with dbg.apply_max_hooks(output_directory=None) as hook:
        assert isinstance(hook, dbg.PrintHook)
        assert dbg.InferenceSession.__init__ == orig_infer_init
    assert dbg.InferenceSession.__init__ == orig_infer_init


def test_apply_max_hooks_with_output_dir() -> None:
    """Test apply_max_hooks patches InferenceSession when output_directory is provided."""
    orig_infer_init = dbg.InferenceSession.__init__
    with TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir)
        with dbg.apply_max_hooks(output_directory=output_path) as hook:
            assert isinstance(hook, dbg.PrintHook)
            assert dbg.InferenceSession.__init__ != orig_infer_init
        assert dbg.InferenceSession.__init__ == orig_infer_init


def test_debug_context_with_hf_overrides() -> None:
    """Test debug_context with HF config overrides."""
    base_cfg = transformers.AutoConfig.for_model("gpt2")
    orig_prop = MAXModelConfig.huggingface_config
    orig_module_load = Module.load_state_dict

    real_cfg = MAXModelConfig(model_path="dummy/text")
    real_cfg._huggingface_config = base_cfg

    with dbg.debug_context(
        output_directory=None,
        hf_config_overrides={"n_embd": 3072},
    ):
        assert MAXModelConfig.huggingface_config is not orig_prop
        cfg_in_ctx = real_cfg.huggingface_config
        assert cfg_in_ctx is not None
        assert cfg_in_ctx.n_embd == 3072

        assert Module.load_state_dict != orig_module_load

    assert MAXModelConfig.huggingface_config is orig_prop
    assert Module.load_state_dict == orig_module_load


def test_debug_context_without_hf_overrides() -> None:
    """Test debug_context without HF config overrides."""
    base_cfg = transformers.AutoConfig.for_model("gpt2")
    orig_prop = MAXModelConfig.huggingface_config
    orig_module_load = Module.load_state_dict

    real_cfg = MAXModelConfig(model_path="dummy/text")
    real_cfg._huggingface_config = base_cfg
    with dbg.debug_context(
        output_directory=None,
        hf_config_overrides=None,
    ):
        assert MAXModelConfig.huggingface_config is orig_prop
        cfg_in_ctx = real_cfg.huggingface_config
        assert cfg_in_ctx is not None
        assert cfg_in_ctx.n_embd == base_cfg.n_embd  # unchanged
        assert Module.load_state_dict != orig_module_load
    assert MAXModelConfig.huggingface_config is orig_prop
    assert Module.load_state_dict == orig_module_load


def test_debug_context_with_output_directory(tmp_path: Path) -> None:
    """Test debug_context with output directory patches InferenceSession."""
    orig_infer_init = dbg.InferenceSession.__init__

    with dbg.debug_context(
        output_directory=tmp_path,
        hf_config_overrides=None,
    ):
        assert dbg.InferenceSession.__init__ != orig_infer_init
    assert dbg.InferenceSession.__init__ == orig_infer_init
