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

"""Correctness test for the unfused rope+store path.

Verifies that ``_fuse_rope_and_store=False`` produces the same output as
the fused path by comparing both against the HuggingFace torch reference.
"""

from __future__ import annotations

import pytest
from layer_mefs import create_runner
from max.pipelines.context import TextContext
from testbed.harnesses.attention_with_rope import AttentionWithRopeStaticParams
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner
from testbed.specs import ATTENTION_UNFUSED_ROPE

_CORRECTNESS_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=11),
    AttentionDynamicParams(batch_size=1, seq_len=128),
]


@pytest.fixture(scope="module")
def unfused_runner() -> LayerTestRunner[
    AttentionWithRopeStaticParams,
    AttentionDynamicParams,
    list[TextContext],
]:
    return create_runner(ATTENTION_UNFUSED_ROPE)


def test_unfused_correctness(
    unfused_runner: LayerTestRunner[
        AttentionWithRopeStaticParams,
        AttentionDynamicParams,
        list[TextContext],
    ],
) -> None:
    """Unfused rope+store path matches torch reference."""
    results = unfused_runner.correctness(
        _CORRECTNESS_SHAPES, atol=0.0625, rtol=0.016, cos_threshold=0.001
    )
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"
