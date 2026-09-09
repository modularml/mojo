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

"""Harness tests for the Wan VAE AttentionBlock (correctness + benchmark).

Runs a correctness comparison between the MAX ``AttentionBlock`` and a
hand-rolled torch reference that mirrors diffusers' single-head
``WanAttentionBlock``. Small smoke defaults cover the common regression
(multi-head vs. single-head weight interpretation); production shapes
can be dialled in via env vars.

Environment variables:

* ``WAN_VAE_DIM`` (default ``96``) — channel dim.
* ``WAN_VAE_BATCH`` (default ``1``), ``WAN_VAE_FRAMES`` (default ``2``),
  ``WAN_VAE_H``, ``WAN_VAE_W`` (default ``8`` each) — shape knobs.
* ``WAN_VAE_RUN_CORRECTNESS`` (default ``1``) — toggle correctness test.
* ``WAN_VAE_RUN_BENCHMARK`` (default ``0``) — toggle benchmark test.
* ``WAN_VAE_ITERATIONS`` (default ``50``), ``WAN_VAE_WARMUP``
  (default ``10``) — benchmark knobs.

Example — 720p mid-block shape on the remote B200 runner::

    bt-b200 //max/python/layer_benchmarks:harness_wan_vae_attention \\
      --test_output=streamed --curses=no --noshow_progress \\
      --test_timeout=600 \\
      --test_env=WAN_VAE_DIM=384 \\
      --test_env=WAN_VAE_FRAMES=4 \\
      --test_env=WAN_VAE_H=90 \\
      --test_env=WAN_VAE_W=160
"""

from __future__ import annotations

import os

import pytest
from benchmark_utils import print_results_table
from layer_mefs import create_runner
from testbed.correctness import print_correctness_report
from testbed.harnesses.wan_vae_attention import (
    WanVaeAttentionDynamicParams,
    WanVaeAttentionStaticParams,
)
from testbed.runner import LayerTestRunner
from testbed.specs import WAN_VAE_ATTENTION


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val is not None else default


def _make_static_params() -> WanVaeAttentionStaticParams:
    # Defaulting to the spec keeps an un-overridden run identical to the graph
    # the CPU build action precompiled, so it can skip the compile.
    return WanVaeAttentionStaticParams(
        dim=_env_int("WAN_VAE_DIM", WAN_VAE_ATTENTION.static_params.dim),
    )


def _make_shapes() -> list[WanVaeAttentionDynamicParams]:
    return [
        WanVaeAttentionDynamicParams(
            batch_size=_env_int("WAN_VAE_BATCH", 1),
            num_frames=_env_int("WAN_VAE_FRAMES", 2),
            height=_env_int("WAN_VAE_H", 8),
            width=_env_int("WAN_VAE_W", 8),
        )
    ]


_RUN_CORRECTNESS = os.environ.get("WAN_VAE_RUN_CORRECTNESS", "1") == "1"
_RUN_BENCHMARK = os.environ.get("WAN_VAE_RUN_BENCHMARK", "0") == "1"


@pytest.fixture(scope="module")
def runner() -> LayerTestRunner[
    WanVaeAttentionStaticParams, WanVaeAttentionDynamicParams, None
]:
    params = _make_static_params()
    print(f"\nWanVaeAttention config: {params}")
    return create_runner(WAN_VAE_ATTENTION, params)


@pytest.mark.skipif(
    not _RUN_CORRECTNESS,
    reason="Set WAN_VAE_RUN_CORRECTNESS=0 to skip correctness.",
)
def test_correctness(
    runner: LayerTestRunner[
        WanVaeAttentionStaticParams, WanVaeAttentionDynamicParams, None
    ],
) -> None:
    shapes = _make_shapes()
    # bf16 attention output has real per-element drift (softmax +
    # QKV matmul + scale) that grows with seq_len; an allclose
    # gate isn't meaningful at this dtype, so widen atol/rtol and
    # gate on cos_dist instead. On remote B200 the patched layer
    # measures cos_dist ~1.5e-4 at 720p; the multi-head regression
    # measures cos_dist ~0.15-0.62, a ~1000x signal.
    results = runner.correctness(shapes, atol=2.5, rtol=0.5, cos_threshold=1e-2)
    print_correctness_report(results)
    for r in results:
        assert r.passed, f"Correctness failed for {r.label}: {r}"


@pytest.mark.skipif(
    not _RUN_BENCHMARK,
    reason="Set WAN_VAE_RUN_BENCHMARK=1 to run the benchmark.",
)
def test_benchmark(
    runner: LayerTestRunner[
        WanVaeAttentionStaticParams, WanVaeAttentionDynamicParams, None
    ],
) -> None:
    shapes = _make_shapes()
    iterations = _env_int("WAN_VAE_ITERATIONS", 50)
    warmup = _env_int("WAN_VAE_WARMUP", 10)
    results = runner.benchmark(shapes, iterations=iterations, warmup=warmup)
    print_results_table("WanVaeAttention", results)
    for _label, stats in results:
        assert stats.mean_ms > 0.0
