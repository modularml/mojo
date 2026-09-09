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

logger = logging.getLogger("max.pipelines")


@pytest.mark.skipif(is_h100_h200(), reason="AITLIB-342: Failing on H100")
def test_pipelines_cli__smollm_bfloat16_with_structured_output_enabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Bazel hands the artifacts directory over in the environment; the pipeline
    # is told about it explicitly.
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
                *pipeline_flags("smollm-structured-output"),
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
