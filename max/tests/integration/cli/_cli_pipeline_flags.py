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

"""The pipeline flags shared by the GPU CLI tests and their CPU precompilers.

Each flag set names one pipeline configuration. The precompiling and the
executing run must build the same graphs, so both sides read the set from here
rather than spelling the flags out; a flag changed on one side only would leave
the precompiled artifacts unusable.
"""

from __future__ import annotations

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Flags that distinguish each configuration from the shared base below. Adding a
# set here is most of what a new precompiled CLI test needs.
_EXTRA_FLAGS = {
    "smollm": [],
    # Enabling structured output server-wide without a JSON schema must not
    # change the outputs of the base chat experience. It does add a second,
    # bitmask-aware sampler graph, so this configuration's artifacts differ.
    "smollm-structured-output": ["--enable-structured-output"],
}

FLAG_SETS = tuple(_EXTRA_FLAGS)


def pipeline_flags(flag_set: str) -> list[str]:
    """Returns the pipeline-config flags for one configuration.

    Per-request sampling params (``--top-k`` and friends) are deliberately
    absent: they reach the pipeline as a ``SamplingParams`` per request rather
    than as graph structure, and ``warm-cache`` does not accept them.

    Args:
        flag_set: One of :data:`FLAG_SETS`.

    Returns:
        The CLI flags for the requested configuration.

    Raises:
        ValueError: If ``flag_set`` is not a known configuration.
    """
    if flag_set not in _EXTRA_FLAGS:
        raise ValueError(
            f"unknown flag set {flag_set!r}; expected one of {FLAG_SETS}"
        )

    return [
        "--model-path",
        REPO_ID,
        "--trust-remote-code",
        "--device-memory-utilization=0.1",
        "--quantization-encoding=bfloat16",
        "--devices=gpu",
        *_EXTRA_FLAGS[flag_set],
    ]
