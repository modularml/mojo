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

"""Harness tests for Qwen2.5VL decoder attention (Qwen2.5-VL-3B-like config)."""

from __future__ import annotations

import pytest
from layer_mefs import create_runner
from max.pipelines.context import TextContext
from testbed.harnesses.qwen2_5vl_attention import Qwen25VLAttentionStaticParams
from testbed.harnesses.ragged_attention_harness import AttentionDynamicParams
from testbed.runner import LayerTestRunner
from testbed.specs import QWEN2_5VL_ATTENTION

_SMOKE_SHAPES = [
    AttentionDynamicParams(batch_size=1, seq_len=11),
    AttentionDynamicParams(batch_size=1, seq_len=128),
    AttentionDynamicParams(batch_size=1, seq_len=1, ctx_len=128),
]


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    Qwen25VLAttentionStaticParams,
    AttentionDynamicParams,
    list[TextContext],
]:
    return create_runner(QWEN2_5VL_ATTENTION)


def test_benchmark_smoke(
    runner: LayerTestRunner[
        Qwen25VLAttentionStaticParams,
        AttentionDynamicParams,
        list[TextContext],
    ],
) -> None:
    results = runner.benchmark(_SMOKE_SHAPES, iterations=1, warmup=1)
    for _label, stats in results:
        assert stats.mean_ms > 0.0
