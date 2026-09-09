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
import os

import pytest
from _cli_pipeline_flags import pipeline_flags
from max._entrypoints import pipelines
from test_common.graph_utils import is_h100_h200
from test_common.lora_utils import (
    create_multiple_test_lora_adapters,
    create_test_lora_adapter,
)

# Keep original constants for non-LoRA tests
REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
logger = logging.getLogger("max.pipelines")


@pytest.mark.skipif(is_h100_h200(), reason="AITLIB-342: Failing on H100")
def test_pipelines_cli__smollm_bfloat16(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Bazel hands the precompiled artifacts over in the environment; the
    # pipeline is told about them explicitly.
    precompiled = os.environ.get("PRECOMPILED_MEFS_DIR")
    reuse_flags = ["--precompiled-mefs", precompiled] if precompiled else []

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--prompt",
                "Why is the sky blue",
                "--top-k=1",
                *reuse_flags,
                *pipeline_flags("smollm"),
            ]
        )
    captured = capsys.readouterr()
    # `generate` prints this summary only after it has produced tokens. Checking
    # for it first means a run that exits early cannot satisfy the content check
    # below off the back of the prompt being echoed.
    assert "Output size:" in captured.out, (
        f"the CLI produced no generation:\n{(captured.out + captured.err)[-4000:]}"
    )
    # Deliberately excludes "blue": it appears in the prompt, so it cannot
    # distinguish a real answer from an echo.
    assert any(
        word in captured.out.lower()
        for word in ["light", "scatter", "atmosphere", "wavelength"]
    ), captured.out


@pytest.mark.skip("LoRA doesn't work with generate entrypoint. E2EOPT-457")
def test_pipelines_cli__smollm_with_lora(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test SmolLM2 with LoRA adapter via CLI."""
    lora_path = create_test_lora_adapter(prefix="cli_test")
    model_path = REPO_ID

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                model_path,
                "--prompt",
                "What is machine learning?",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--devices=gpu",
                "--enable-lora",
                "-max-num-loras=2",
                "--max-lora-rank=16",
                f"--lora-paths={lora_path}",
                "--max-new-tokens=50",
                "--top-k=1",
            ]
        )
    captured = capsys.readouterr()

    # Verify output contains response about machine learning
    assert len(captured.out) > 0
    # Common words that might appear in ML explanation
    assert any(
        word in captured.out.lower()
        for word in ["learn", "data", "algorithm", "model"]
    )


@pytest.mark.skip("LoRA doesn't work with generate entrypoint. E2EOPT-457")
def test_pipelines_cli__smollm_with_multiple_loras(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Test SmolLM2 with multiple LoRA adapters via CLI."""

    # Create multiple LoRA adapters
    lora_adapter_paths = create_multiple_test_lora_adapters(
        num_adapters=2, prefix="cli_multi_test"
    )
    model_path = REPO_ID

    # Multiple LoRA adapters with names
    lora_paths = [
        f"adapter1={lora_adapter_paths[0]}",
        f"adapter2={lora_adapter_paths[1]}",
    ]

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                model_path,
                "--prompt",
                "Explain quantum computing",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--devices=gpu",
                "--enable-lora",
                "--max-num-loras=3",
                "--max-lora-rank=16",
                f"--lora-paths={lora_paths[0]}",
                f"--lora-paths={lora_paths[1]}",
                "--max-new-tokens=100",
                "--top-k=1",
            ]
        )
    captured = capsys.readouterr()

    # Verify output contains response about quantum computing
    assert len(captured.out) > 0
    assert any(
        word in captured.out.lower()
        for word in ["quantum", "computer", "qubit", "state"]
    )
