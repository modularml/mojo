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

"""Ensure proper errors when loading models that require trust_remote_code."""

import json
from pathlib import Path

import pytest
from max.pipelines import PIPELINE_REGISTRY
from max.pipelines.weights.hf_utils import HuggingFaceRepo


def test_custom_model_type_surfaces_autoconfig_error(tmp_path: Path) -> None:
    """Config with custom code should raise, not silently degrade."""
    config = {
        "model_type": "not_a_real_type",
        "auto_map": {"AutoConfig": "custom_config.CustomConfig"},
        "architectures": ["CustomModelForCausalLM"],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    repo = HuggingFaceRepo(repo_id=str(tmp_path))

    with pytest.raises(ValueError, match="trust_remote_code"):
        PIPELINE_REGISTRY.get_active_huggingface_config(repo)
