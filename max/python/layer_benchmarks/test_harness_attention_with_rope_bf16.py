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

"""Harness tests for AttentionWithRope (bf16)."""

from __future__ import annotations

import pytest
from layer_mefs import create_runner
from max.pipelines.context import TextContext
from testbed.harnesses.attention_with_rope import AttentionWithRopeStaticParams
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner
from testbed.specs import ATTENTION_WITH_ROPE_BF16

_SMOKE_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=8192, ctx_len=8192),
    AttentionDynamicParams(batch_size=1, seq_len=1, ctx_len=8192),
]


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    AttentionWithRopeStaticParams,
    AttentionDynamicParams,
    list[TextContext],
]:
    return create_runner(ATTENTION_WITH_ROPE_BF16)


def test_benchmark_smoke(
    runner: LayerTestRunner[
        AttentionWithRopeStaticParams,
        AttentionDynamicParams,
        list[TextContext],
    ],
) -> None:
    results = runner.benchmark(_SMOKE_SHAPES, iterations=1, warmup=1)
    for _label, stats in results:
        assert stats.mean_ms > 0.0


def test_correctness(
    runner: LayerTestRunner[
        AttentionWithRopeStaticParams,
        AttentionDynamicParams,
        list[TextContext],
    ],
) -> None:
    # Correctness only works for prefill (ctx_len=0), batch_size=1.
    shapes = [
        AttentionDynamicParams(batch_size=1, seq_len=11),
        AttentionDynamicParams(batch_size=1, seq_len=128),
    ]
    results = runner.correctness(
        shapes, atol=0.0625, rtol=0.016, cos_threshold=0.001
    )
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"
