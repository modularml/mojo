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

"""Harness tests for WAN TextEncoder (correctness + benchmark).

By default, correctness runs (compares MAX graph vs HF torch reference).
Set ``TEXT_ENC_RUN_BENCHMARK=1`` to also run the MAX benchmark (requires
a dedicated GPU memory budget for the HF torch model to coexist, or
disable correctness via ``TEXT_ENC_RUN_CORRECTNESS=0``).

Environment variables:

* Model size: ``TEXT_ENC_VOCAB_SIZE``, ``TEXT_ENC_D_MODEL``,
  ``TEXT_ENC_D_KV``, ``TEXT_ENC_D_FF``, ``TEXT_ENC_NUM_LAYERS``,
  ``TEXT_ENC_NUM_HEADS``, ``TEXT_ENC_EMBED_SEQ_LEN``
* Correctness: ``TEXT_ENC_RUN_CORRECTNESS=0`` to disable (default on)
* Benchmark: ``TEXT_ENC_RUN_BENCHMARK=1`` to enable (default off),
  ``TEXT_ENC_ITERATIONS`` (default 50), ``TEXT_ENC_WARMUP`` (default 10)

Example (production WAN 2.2 config, benchmark only)::

    ./bazelw test //max/python/layer_benchmarks:harness_text_encoder \\
      --test_output=streamed --curses=no --noshow_progress \\
      --test_timeout=600 \\
      --test_env=TEXT_ENC_VOCAB_SIZE=256384 \\
      --test_env=TEXT_ENC_D_MODEL=4096 \\
      --test_env=TEXT_ENC_D_KV=64 \\
      --test_env=TEXT_ENC_D_FF=10240 \\
      --test_env=TEXT_ENC_NUM_LAYERS=24 \\
      --test_env=TEXT_ENC_NUM_HEADS=64 \\
      --test_env=TEXT_ENC_RUN_CORRECTNESS=0 \\
      --test_env=TEXT_ENC_RUN_BENCHMARK=1
"""

from __future__ import annotations

import os

import pytest
from benchmark_utils import print_results_table
from layer_mefs import create_runner
from testbed.correctness import print_correctness_report
from testbed.harnesses.text_encoder import (
    TextEncoderDynamicParams,
    TextEncoderStaticParams,
)
from testbed.runner import LayerTestRunner
from testbed.specs import TEXT_ENCODER


def _env_int(name: str, default: int) -> int:
    """Read an integer from an environment variable, falling back to default."""
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _make_static_params() -> TextEncoderStaticParams:
    """Build static params from env vars, defaulting to the spec's smoke config.

    Defaulting to the spec keeps an un-overridden run identical to the graph the
    CPU build action precompiled, so it can skip the compile.
    """
    smoke = TEXT_ENCODER.static_params
    return TextEncoderStaticParams(
        vocab_size=_env_int("TEXT_ENC_VOCAB_SIZE", smoke.vocab_size),
        d_model=_env_int("TEXT_ENC_D_MODEL", smoke.d_model),
        d_kv=_env_int("TEXT_ENC_D_KV", smoke.d_kv),
        d_ff=_env_int("TEXT_ENC_D_FF", smoke.d_ff),
        num_layers=_env_int("TEXT_ENC_NUM_LAYERS", smoke.num_layers),
        num_heads=_env_int("TEXT_ENC_NUM_HEADS", smoke.num_heads),
        embed_seq_len=_env_int("TEXT_ENC_EMBED_SEQ_LEN", smoke.embed_seq_len),
    )


_SMOKE_SHAPES = [
    # All tokens real, full 512 seq len (typical WAN input).
    TextEncoderDynamicParams(batch_size=1, seq_len=512),
    # Padded input: only 100 real tokens out of 512.
    TextEncoderDynamicParams(batch_size=1, seq_len=512, num_real_tokens=100),
]

_RUN_CORRECTNESS = os.environ.get("TEXT_ENC_RUN_CORRECTNESS", "1") == "1"
_RUN_BENCHMARK = os.environ.get("TEXT_ENC_RUN_BENCHMARK", "0") == "1"


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    TextEncoderStaticParams, TextEncoderDynamicParams, None
]:
    params = _make_static_params()
    print(f"\nTextEncoder config: {params}")
    return create_runner(TEXT_ENCODER, params)


@pytest.mark.skipif(
    not _RUN_CORRECTNESS,
    reason="Set TEXT_ENC_RUN_CORRECTNESS=0 to skip correctness.",
)
def test_correctness(
    runner: LayerTestRunner[
        TextEncoderStaticParams, TextEncoderDynamicParams, None
    ],
) -> None:
    results = runner.correctness(
        _SMOKE_SHAPES, atol=1e-2, rtol=1e-2, cos_threshold=0.001
    )
    print_correctness_report(results)
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"


@pytest.mark.skipif(
    not _RUN_BENCHMARK,
    reason="Set TEXT_ENC_RUN_BENCHMARK=1 to run the benchmark.",
)
def test_benchmark(
    runner: LayerTestRunner[
        TextEncoderStaticParams, TextEncoderDynamicParams, None
    ],
) -> None:
    iterations = _env_int("TEXT_ENC_ITERATIONS", 50)
    warmup = _env_int("TEXT_ENC_WARMUP", 10)
    results = runner.benchmark(
        _SMOKE_SHAPES, iterations=iterations, warmup=warmup
    )
    print_results_table("TextEncoder", results)
    for _label, stats in results:
        assert stats.mean_ms > 0.0
