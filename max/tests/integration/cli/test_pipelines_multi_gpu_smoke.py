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

import logging

import pytest
from max._entrypoints import pipelines

# Keep original constants for non-LoRA tests
REPO_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"
logger = logging.getLogger("max.pipelines")


def test_pipelines_multi_gpu_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    # Use HuggingFace repo ID directly to ensure we have access to all weight formats
    local_model_path = REPO_ID

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--devices=gpu:0,1,2,3",
                "--max-batch-size=1",
                "--max-new-tokens=32",
                "--no-use-subgraphs",
                "--max-length=512",
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_pipelines_multi_gpu_smoke_with_subgraphs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Use HuggingFace repo ID directly to ensure we have access to all weight formats
    local_model_path = REPO_ID

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--devices=gpu:0,1,2,3",
                "--max-batch-size=1",
                "--max-new-tokens=32",
                "--max-length=512",
                "--use-subgraphs",
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0
