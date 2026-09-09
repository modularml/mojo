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

# AMD MLA (Multi-head Latent Attention) benchmark comparing MAX against aiter.
# Supports prefill and decode in both bfloat16 and float8_e4m3fn (FP8).
# Model shape defaults to Kimi K2.5 (num_q_heads=64, qk_nope=128, qk_rope=64,
# kv_lora_rank=512, v_head_dim=128). DeepSeek-V3 shape available via
# `--num_q_heads 128`.
#
# Run via kbench: kbench bench_amd_mla_decode.yaml (or bench_amd_mla_prefill.yaml)

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Configure aiter JIT environment before any aiter imports. When run via
# kbench (file: mode), the Bazel env vars aren't applied, so set them as
# fallbacks.
if "AITER_JIT_DIR" not in os.environ:
    _ws = os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd())
    os.environ["AITER_JIT_DIR"] = os.path.join(
        _ws, ".derived", "aiter_jit_cache"
    )
if "/usr/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = (
        "/usr/bin:/bin:/usr/local/bin:/opt/rocm/bin:"
        + os.environ.get("PATH", "")
    )

import numpy as np
import torch
from bencher_utils import Bench, ThroughputMeasure
from max._kv_cache_ops import mla_dispatch_args_scalar
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flare_mla_decode_ragged,
    flare_mla_prefill_ragged,
)
from max.nn.kv_cache import MLAKVCacheParams, PagedCacheValues

_aiter: types.ModuleType | None
_aiter_mla: types.ModuleType | None
try:
    import aiter as _aiter
    from aiter import mla as _aiter_mla
except (ImportError, Exception) as e:
    print(f"Warning: aiter not available: {e}")
    _aiter = None
    _aiter_mla = None

# MI355X L2 cache size (256 MB).
_L2_CACHE_SIZE_BYTES = int(256e6)
# MAX MLA kernels are only validated for page_size=128.
_MAX_PAGE_SIZE = 128
# Number of rotating KV copies for the graph-replay decode benches. The total
# footprint (ncopies * KV) must exceed the 256 MB L2 so each replay reads cold
# HBM. Overridable via `--ncopies`.
_NCOPIES = 16
# aiter decode scheduling: True = persistent `_ps` asm kernel (default,
# vllm/sglang production); False = non-persistent split-K + Triton reduce.
# Toggled via `--aiter-sched`.
_AITER_PERSISTENT = True


@dataclass
class Config:
    """MLA shape parameters. Defaults to Kimi K2.5 (num_q_heads=64);
    DeepSeek-V3 uses num_q_heads=128, same other dims."""

    num_q_heads: int = 64
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    kv_lora_rank: int = 512
    v_head_dim: int = 128

    @property
    def qk_head_dim(self) -> int:
        # Compressed Q head_dim (also equals KV cache latent+rope dim).
        return self.kv_lora_rank + self.qk_rope_head_dim  # 576

    @property
    def qk_prefill_head_dim(self) -> int:
        # Decompressed Q head_dim used by flare_mla_prefill_ragged's input.
        return self.qk_nope_head_dim + self.qk_rope_head_dim  # 192


def _bench_cuda_events(
    fn: Callable[[], Any],
    num_warmups: int = 50,
    num_iters: int = 100,
    flush_l2: bool = True,
) -> float:
    """Benchmark using CUDA events. Returns median time in seconds."""
    torch.cuda.synchronize()
    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    # Allocate the L2-flush buffer once and reuse it across iterations.
    flush_buffer = (
        torch.empty(_L2_CACHE_SIZE_BYTES // 4, dtype=torch.int, device="cuda")
        if flush_l2
        else None
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times: list[float] = []
    for _ in range(num_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2] / 1e3  # seconds


def _time_replays(
    replay: Callable[[], Any], max_iters: int = 300, target_s: float = 0.03
) -> float:
    """Time a back-to-back replay loop (s/iter), auto-scaling the iteration
    count to the kernel size: a quick 5-replay probe picks `iters` so the
    timed loop runs ~`target_s`, clamped to [20, max_iters]. Keeps tiny decode
    kernels well-averaged AND huge O(seqlen^2) prefill kernels tractable."""
    torch.cuda.synchronize()
    for _ in range(5):
        replay()
    torch.cuda.synchronize()
    es = torch.cuda.Event(enable_timing=True)
    ee = torch.cuda.Event(enable_timing=True)
    es.record()
    for _ in range(5):
        replay()
    ee.record()
    torch.cuda.synchronize()
    per = es.elapsed_time(ee) / 1e3 / 5  # s/replay
    iters = (
        max(20, min(max_iters, int(target_s / per))) if per > 0 else max_iters
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def _time_cuda_graph(call: Callable[[], Any], max_iters: int = 300) -> float:
    """Capture `call` into a CUDA graph and time auto-scaled back-to-back
    replay (s/iter). Replay removes per-call Python/torch dispatch, so the loop
    is GPU-bound and measures the kernel(s). Used for the aiter benches (a plain
    Python loop is CPU-bound on aiter's per-call dispatch). Raises on capture
    failure."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    torch.cuda.synchronize()
    return _time_replays(g.replay, max_iters=max_iters)


def _compute_prefill_flops(
    config: Config, batch_size: int, qkv_len: int, causal: bool
) -> int:
    """FLOPS for MLA prefill decompressed path:
    Q @ K^T (qk_head_dim) + softmax(P) @ V (v_head_dim), causal halves s*s.
    """
    div = 2 if causal else 1
    qk = (
        2
        * batch_size
        * config.num_q_heads
        * qkv_len
        * qkv_len
        * config.qk_prefill_head_dim
        // div
    )
    pv = (
        2
        * batch_size
        * config.num_q_heads
        * qkv_len
        * qkv_len
        * config.v_head_dim
        // div
    )
    return qk + pv


def _compute_decode_bytes(
    config: Config, batch_size: int, cache_len: int, dtype: torch.dtype
) -> int:
    """Bytes moved for MLA decode (memory-bound). Reads Q + compressed KV
    cache, writes BF16 output."""

    def _bpe(dt: torch.dtype) -> int:
        if dt in (torch.float8_e4m3fn, torch.float8_e5m2):
            return 1
        if dt in (torch.bfloat16, torch.float16):
            return 2
        return 4

    kv_bpe = _bpe(dtype)
    q_bpe = _bpe(dtype)
    out_bpe = 2  # output is always bf16

    # Q: [batch, num_q_heads, qk_head_dim]
    q_bytes = batch_size * config.num_q_heads * config.qk_head_dim * q_bpe
    # KV compressed: [batch, cache_len, qk_head_dim]
    kv_bytes = batch_size * cache_len * config.qk_head_dim * kv_bpe
    # Out: [batch, num_q_heads, v_head_dim]
    o_bytes = batch_size * config.num_q_heads * config.kv_lora_rank * out_bpe
    return q_bytes + kv_bytes + o_bytes


# ----------------------------------------------------------------------------
# MAX backends.
# ----------------------------------------------------------------------------


def _auto_ncopies(kv_bytes_per_copy: int) -> int:
    """Rotating-copy count so the working set exceeds the L2 (=> cold HBM
    reads), capped at 16. At large (batch, cache_len) one copy already
    exceeds L2 so 2 suffice (bounds memory + capture time); at small sizes
    the cap of 16 is used (those are latency-bound, not bandwidth-bound, so
    the residual L2 warmth is immaterial and identical for both engines)."""
    return max(
        2,
        min(
            16,
            math.ceil(1.5 * _L2_CACHE_SIZE_BYTES / max(1, kv_bytes_per_copy)),
        ),
    )


def bench_max_decode(
    batch_size: int,
    cache_len: int,
    config: Config,
    dtype: torch.dtype,
    num_iters: int,
    ncopies: int | None = None,
) -> tuple[float, int] | None:
    """MAX flare_mla_decode_ragged benchmark (paged KV cache).

    Q and KV share the same dtype (the AMD MLA kernel's MMA path cannot
    mix FP8/BF16). For FP8: Q=FP8, KV=FP8, output=BF16 — matching the
    `mla_decode_branch_fp8` production path where `mla_decode_input_buf`
    is allocated at the KV cache dtype.
    """
    is_fp8 = dtype == torch.float8_e4m3fn
    kv_bpe = 1 if is_fp8 else 2
    kv_bytes_per_copy = batch_size * cache_len * config.qk_head_dim * kv_bpe
    max_dtype = torch_dtype_to_max(dtype)

    kv_params = MLAKVCacheParams(
        dtype=max_dtype,
        head_dim=config.qk_head_dim,
        num_layers=1,
        page_size=_MAX_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=config.num_q_heads,
    )

    # Allocate one extra page of slack: the ragged kernel path has
    # `_is_cache_length_accurate=False`, so it reads `num_keys = cache_len + 1`
    # during decode. The extra page is dummy data but prevents OOB lookup
    # for the last token.
    num_blocks_per_seq = (cache_len + 1 + _MAX_PAGE_SIZE - 1) // _MAX_PAGE_SIZE

    # KV cache blocks: [num_pages, 1, 1, page_size, 1, qk_head_dim]
    blocks_torch = torch.randn(
        batch_size * num_blocks_per_seq,
        1,
        1,
        _MAX_PAGE_SIZE,
        1,
        config.qk_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if is_fp8:
        blocks_torch = blocks_torch.to(dtype)

    lut_torch = (
        torch.arange(
            batch_size * num_blocks_per_seq, dtype=torch.int32, device="cuda"
        )
        .reshape(batch_size, num_blocks_per_seq)
        .to(torch.uint32)
    )
    cache_lengths_torch = torch.full(
        (batch_size,), cache_len, dtype=torch.uint32, device="cuda"
    )
    max_prompt_length_torch = torch.tensor(
        [1], dtype=torch.uint32, device="cpu"
    )
    max_cache_length_torch = torch.tensor(
        [cache_len], dtype=torch.uint32, device="cpu"
    )

    # FP8 tensors can't be DLPack'd directly; round-trip via uint8.
    if is_fp8:
        blocks_max = Buffer.from_dlpack(blocks_torch.view(torch.uint8)).view(
            max_dtype
        )
    else:
        blocks_max = Buffer.from_dlpack(blocks_torch)
    lut_max = Buffer.from_dlpack(lut_torch)
    cache_lengths_max = Buffer.from_dlpack(cache_lengths_torch)
    max_prompt_length_max = Buffer.from_dlpack(max_prompt_length_torch)
    max_cache_length_max = Buffer.from_dlpack(max_cache_length_torch)

    q_type = TensorType(
        max_dtype,
        shape=["total_tokens", config.num_q_heads, config.qk_head_dim],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["batch_plus_1"], device=DeviceRef.GPU()
    )
    blocks_type = BufferType(
        max_dtype,
        shape=[
            "total_num_pages",
            1,
            1,
            _MAX_PAGE_SIZE,
            1,
            config.qk_head_dim,
        ],
        device=DeviceRef.GPU(),
    )
    cache_lengths_type = TensorType(
        DType.uint32, shape=["batch"], device=DeviceRef.GPU()
    )
    lookup_table_type = TensorType(
        DType.uint32, shape=["batch", "max_num_pages"], device=DeviceRef.GPU()
    )
    max_prompt_length_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )
    max_cache_length_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )
    scalar_args_type = TensorType(
        DType.int64, shape=[3], device=DeviceRef.GPU()
    )

    # Rotating KV copies: the total working set must exceed the 256 MB L2 so
    # each buffer is read cold from HBM. The `ncopies` distinct buffers are
    # chained into ONE device-graph below (op i reads buffer i).
    ncopies = _NCOPIES if ncopies is None else ncopies
    if ncopies <= 0:
        ncopies = _auto_ncopies(kv_bytes_per_copy)

    keepalive: list[Any] = []
    blocks_bufs: list[Buffer] = [blocks_max]
    for _ in range(ncopies - 1):
        bt = torch.randn(
            batch_size * num_blocks_per_seq,
            1,
            1,
            _MAX_PAGE_SIZE,
            1,
            config.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        if is_fp8:
            bt = bt.to(dtype)
            bb = Buffer.from_dlpack(bt.view(torch.uint8)).view(max_dtype)
        else:
            bb = Buffer.from_dlpack(bt)
        keepalive.append(bt)
        blocks_bufs.append(bb)

    # One device-graph chaining `ncopies` decode ops, op i reading a distinct
    # rotating KV buffer. Replaying this single graph sweeps all `ncopies` cold
    # buffers (total > L2) in ONE launch, giving a cold-HBM kernel measurement
    # free of the ~4-6us per-replay launch gap a 1-op replay loop pays each
    # iteration. Reported latency = whole-graph time / ncopies (per-op).
    session = InferenceSession(devices=[Accelerator()])
    with Graph(
        "mla_decode_max_chain",
        input_types=[
            q_type,
            input_row_offsets_type,
            *([blocks_type] * ncopies),
            cache_lengths_type,
            lookup_table_type,
            max_prompt_length_type,
            max_cache_length_type,
            scalar_args_type,
        ],
    ) as graph:
        ins = graph.inputs
        q, row_offsets = ins[0], ins[1]
        blocks_in = ins[2 : 2 + ncopies]
        (
            cache_lens,
            lut,
            max_prompt_len,
            max_cache_len,
            scalar_args,
        ) = ins[2 + ncopies : 7 + ncopies]
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
        results = [
            flare_mla_decode_ragged(
                kv_params,
                q.tensor,
                row_offsets.tensor,
                PagedCacheValues(
                    bl.buffer,
                    cache_lens.tensor,
                    lut.tensor,
                    max_prompt_len.tensor,
                    max_cache_len.tensor,
                ),
                layer_idx,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=1.0 / math.sqrt(config.qk_prefill_head_dim),
                scalar_args=scalar_args.tensor,
                qk_rope_dim=config.qk_rope_head_dim,
            )
            for bl in blocks_in
        ]
        graph.output(*results)

    model = session.load(graph)

    total_tokens = batch_size  # decode: q_len_per_request=1
    q_torch = torch.randn(
        total_tokens,
        config.num_q_heads,
        config.qk_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if is_fp8:
        q_torch = q_torch.to(dtype)
        q_input = Buffer.from_dlpack(q_torch.view(torch.uint8)).view(max_dtype)
    else:
        q_input = q_torch

    input_row_offsets = torch.arange(
        0,
        total_tokens + 1,
        1,
        dtype=torch.int32,
        device="cuda",
    ).to(torch.uint32)

    device = Accelerator()
    scalar_args_gpu = Buffer.from_numpy(
        np.array(
            mla_dispatch_args_scalar(
                batch_size,
                cache_len,
                1,
                config.num_q_heads,
                is_fp8,
                device,
            ),
            dtype=np.int64,
        )
    ).to(device)

    # Build the capture/replay input tuple: q, the `ncopies` distinct cold KV
    # buffers, then the shared ragged metadata. The single captured graph runs
    # all `ncopies` decode ops back-to-back over the distinct buffers, so one
    # replay is a cold-HBM, dispatch-free measurement of `ncopies` decodes.
    q_buf = q_input if is_fp8 else Buffer.from_dlpack(q_torch)
    ro_buf = Buffer.from_dlpack(input_row_offsets)
    graph_inputs = (
        q_buf,
        ro_buf,
        *blocks_bufs,
        cache_lengths_max,
        lut_max,
        max_prompt_length_max,
        max_cache_length_max,
        scalar_args_gpu,
    )
    outs = model.capture(0, *graph_inputs)
    keepalive.append(outs)
    # Validate the captured graph matches eager execution.
    model.debug_verify_replay(0, *graph_inputs)

    nrun = max(num_iters, 200)
    torch.cuda.synchronize()
    for _ in range(50):
        model.replay(0, *graph_inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nrun):
        model.replay(0, *graph_inputs)
    end.record()
    torch.cuda.synchronize()
    # Per-op latency = whole chained-graph time / number of chained ops.
    per_op_s = start.elapsed_time(end) / 1e3 / nrun / ncopies

    working_set_bytes = kv_bytes_per_copy * ncopies
    kv_mb = kv_bytes_per_copy / (1024.0 * 1024.0)
    cold_state = (
        "cold"
        if working_set_bytes > _L2_CACHE_SIZE_BYTES
        else f"WARM: working set < {_L2_CACHE_SIZE_BYTES / 1e6:.0f}MB L2"
    )
    print(
        f"[MAX chained device-graph] chain={ncopies} "
        f"kv_per_copy~{kv_mb:.1f}MB working_set~{kv_mb * ncopies:.1f}MB ({cold_state}) "
        f"| per-op {per_op_s * 1e6:.2f}us"
    )
    keepalive.clear()
    return per_op_s, _compute_decode_bytes(config, batch_size, cache_len, dtype)


def bench_max_prefill(
    batch_size: int,
    qkv_len: int,
    config: Config,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """MAX flare_mla_prefill_ragged benchmark.

    All KV tokens live in the current ragged input (no existing cache) so the
    measurement is a pure prefill attention step. The kv_collection is still
    constructed because the kernel signature requires it.
    """
    is_fp8 = dtype == torch.float8_e4m3fn
    max_dtype = torch_dtype_to_max(dtype)

    kv_params = MLAKVCacheParams(
        dtype=max_dtype,
        head_dim=config.qk_head_dim,
        num_layers=1,
        page_size=_MAX_PAGE_SIZE,
        devices=[DeviceRef.GPU()],
        num_q_heads=config.num_q_heads,
    )

    # Minimal KV-cache inputs (no previously cached tokens).
    num_blocks_per_seq = max(
        1,
        (qkv_len + _MAX_PAGE_SIZE - 1) // _MAX_PAGE_SIZE,
    )
    total_num_pages = batch_size * num_blocks_per_seq
    blocks_torch = torch.zeros(
        total_num_pages,
        1,
        1,
        _MAX_PAGE_SIZE,
        1,
        config.qk_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if is_fp8:
        blocks_torch = blocks_torch.to(dtype)
    lut_torch = (
        torch.arange(total_num_pages, dtype=torch.int32, device="cuda")
        .reshape(batch_size, num_blocks_per_seq)
        .to(torch.uint32)
    )
    cache_lengths_torch = torch.zeros(
        batch_size, dtype=torch.uint32, device="cuda"
    )
    # max_prompt_length: [max_q_len], max_cache_length: [max_cache_len]
    max_prompt_length_torch = torch.tensor(
        [qkv_len], dtype=torch.uint32, device="cpu"
    )
    max_cache_length_torch = torch.tensor([0], dtype=torch.uint32, device="cpu")

    if is_fp8:
        blocks_max = Buffer.from_dlpack(blocks_torch.view(torch.uint8)).view(
            max_dtype
        )
    else:
        blocks_max = Buffer.from_dlpack(blocks_torch)
    lut_max = Buffer.from_dlpack(lut_torch)
    cache_lengths_max = Buffer.from_dlpack(cache_lengths_torch)
    max_prompt_length_max = Buffer.from_dlpack(max_prompt_length_torch)
    max_cache_length_max = Buffer.from_dlpack(max_cache_length_torch)

    q_type = TensorType(
        max_dtype,
        shape=[
            "total_tokens",
            config.num_q_heads,
            config.qk_prefill_head_dim,
        ],
        device=DeviceRef.GPU(),
    )
    k_type = TensorType(
        max_dtype,
        shape=[
            "total_tokens",
            config.num_q_heads,
            config.qk_nope_head_dim,
        ],
        device=DeviceRef.GPU(),
    )
    v_type = TensorType(
        max_dtype,
        shape=["total_tokens", config.num_q_heads, config.v_head_dim],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["batch_plus_1"], device=DeviceRef.GPU()
    )
    buffer_row_offsets_type = TensorType(
        DType.uint32, shape=["batch_plus_1"], device=DeviceRef.GPU()
    )
    cache_offsets_type = TensorType(
        DType.uint32, shape=["batch_plus_1"], device=DeviceRef.GPU()
    )
    blocks_type = BufferType(
        max_dtype,
        shape=[
            "total_num_pages",
            1,
            1,
            _MAX_PAGE_SIZE,
            1,
            config.qk_head_dim,
        ],
        device=DeviceRef.GPU(),
    )
    cache_lengths_type = TensorType(
        DType.uint32, shape=["batch"], device=DeviceRef.GPU()
    )
    lookup_table_type = TensorType(
        DType.uint32, shape=["batch", "max_num_pages"], device=DeviceRef.GPU()
    )
    max_prompt_length_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )
    max_cache_length_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )

    session = InferenceSession(devices=[Accelerator()])
    with Graph(
        "mla_prefill_max",
        input_types=[
            q_type,
            k_type,
            v_type,
            input_row_offsets_type,
            buffer_row_offsets_type,
            cache_offsets_type,
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_prompt_length_type,
            max_cache_length_type,
        ],
    ) as graph:
        (
            q,
            k,
            v,
            row_offsets,
            buffer_row_offsets,
            cache_offsets,
            blocks,
            cache_lens,
            lut,
            max_prompt_len,
            max_cache_len,
        ) = graph.inputs
        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lens.tensor,
            lut.tensor,
            max_prompt_len.tensor,
            max_cache_len.tensor,
        )
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
        result = flare_mla_prefill_ragged(
            kv_params,
            q.tensor,
            k.tensor,
            v.tensor,
            row_offsets.tensor,
            buffer_row_offsets.tensor,
            cache_offsets.tensor,
            kv_collection,
            layer_idx,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=1.0 / math.sqrt(config.qk_prefill_head_dim),
            qk_rope_dim=config.qk_rope_head_dim,
            output_dtype=DType.bfloat16,
        )
        graph.output(result)

    model = session.load(graph)

    total_tokens = batch_size * qkv_len
    q_torch = torch.randn(
        total_tokens,
        config.num_q_heads,
        config.qk_prefill_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_torch = torch.randn(
        total_tokens,
        config.num_q_heads,
        config.qk_nope_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_torch = torch.randn(
        total_tokens,
        config.num_q_heads,
        config.v_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if is_fp8:
        q_torch = q_torch.to(dtype)
        k_torch = k_torch.to(dtype)
        v_torch = v_torch.to(dtype)
        q_input = Buffer.from_dlpack(q_torch.view(torch.uint8)).view(max_dtype)
        k_input = Buffer.from_dlpack(k_torch.view(torch.uint8)).view(max_dtype)
        v_input = Buffer.from_dlpack(v_torch.view(torch.uint8)).view(max_dtype)
    else:
        q_input = q_torch
        k_input = k_torch
        v_input = v_torch

    row_offsets = torch.arange(
        0,
        total_tokens + 1,
        qkv_len,
        dtype=torch.int32,
        device="cuda",
    ).to(torch.uint32)
    buffer_row_offsets = row_offsets.clone()
    cache_offsets = torch.zeros(
        batch_size + 1, dtype=torch.uint32, device="cuda"
    )

    def run_kernel() -> Any:
        return model.execute(
            q_input if is_fp8 else q_torch.detach(),
            k_input if is_fp8 else k_torch.detach(),
            v_input if is_fp8 else v_torch.detach(),
            row_offsets.detach(),
            buffer_row_offsets.detach(),
            cache_offsets.detach(),
            blocks_max,
            cache_lengths_max,
            lut_max,
            max_prompt_length_max,
            max_cache_length_max,
        )[0]

    # MAX native device-graph (HIP graph) capture/replay -> removes the
    # per-call `model.execute` dispatch (same fix as decode). Prefill is
    # compute-bound, so a single captured graph replayed back-to-back (warm L2
    # is fine) measures the kernel. Eager fallback if capture is unsupported.
    q_buf = q_input if is_fp8 else Buffer.from_dlpack(q_torch)
    k_buf = k_input if is_fp8 else Buffer.from_dlpack(k_torch)
    v_buf = v_input if is_fp8 else Buffer.from_dlpack(v_torch)
    g_inputs = (
        q_buf,
        k_buf,
        v_buf,
        Buffer.from_dlpack(row_offsets),
        Buffer.from_dlpack(buffer_row_offsets),
        Buffer.from_dlpack(cache_offsets),
        blocks_max,
        cache_lengths_max,
        lut_max,
        max_prompt_length_max,
        max_cache_length_max,
    )
    try:
        outs = model.capture(0, *g_inputs)
        model.debug_verify_replay(0, *g_inputs)

        graph_s = _time_replays(lambda: model.replay(0, *g_inputs))
        print(
            f"[MAX prefill device-graph] replay {graph_s * 1e6:.2f}us "
            "(dispatch removed)"
        )
        _ = outs
        return graph_s, _compute_prefill_flops(
            config, batch_size, qkv_len, True
        )
    except Exception as e:
        print(f"  (MAX prefill device-graph capture FAILED, eager: {e})")
        time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
        return time_s, _compute_prefill_flops(config, batch_size, qkv_len, True)


# ----------------------------------------------------------------------------
# aiter backends.
# ----------------------------------------------------------------------------


def _make_aiter_paged_kv(
    batch_size: int,
    max_kv_len: int,
    page_size: int,
    kv_head_dim: int,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build aiter paged KV cache inputs for a uniform batch.

    Returns (kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens).
    kv_buffer shape: [num_pages, page_size, 1, kv_head_dim]
    """
    num_pages_per_seq = (max_kv_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_seq

    kv_buffer_bf16 = torch.randn(
        num_pages,
        page_size,
        1,
        kv_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if dtype == torch.float8_e4m3fn:
        kv_buffer = kv_buffer_bf16.to(dtype)
    else:
        kv_buffer = kv_buffer_bf16.to(dtype)

    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * num_pages_per_seq,
        num_pages_per_seq,
        dtype=torch.int32,
        device="cuda",
    )
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    last_page = max_kv_len - (num_pages_per_seq - 1) * page_size
    kv_last_page_lens = torch.full(
        (batch_size,),
        last_page,
        dtype=torch.int32,
        device="cuda",
    )
    # qo_indptr is filled per-benchmark (depends on q length).
    qo_indptr = torch.empty(
        batch_size + 1,
        dtype=torch.int32,
        device="cuda",
    )
    return kv_buffer, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens


def _build_mla_persistent_metadata(
    batch_size: int,
    cache_len: int,
    num_q_heads: int,
    qo_indptr: torch.Tensor,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    max_seqlen_qo: int = 1,
    kv_granularity: int = 16,
) -> tuple[dict[str, torch.Tensor], int]:
    """Allocate and populate aiter's MLA persistent-scheduling metadata.

    Returns (metadata_kwargs, max_split_per_batch). Values mirror the
    `fast_mode=True` path used by the aiter op_tests reference.

    `kv_granularity=16` matches vllm + sglang production (both use 16 for
    decode on page_size=1). Older benches used 128 which under-utilized
    splits; leaving it configurable for prefill paths that want 128.
    """
    assert _aiter is not None
    # aiter v0.1.10+ defaults `num_kv_splits=32`; the info shape alloc
    # must agree with `max_split_per_batch` at the runtime call site.
    # For nhead>=128 the op_test caps splits to `ceil(cu_num/batch)` to
    # keep the reduce buffer small — over-provisioning GPU-faults on
    # "Write access to a read-only page" in the reduce kernel.
    cu_num = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    max_split_per_batch = 64
    if num_q_heads >= 128:
        max_split_per_batch = min(
            (cu_num + batch_size - 1) // batch_size, max_split_per_batch
        )

    (
        (m_meta_size, m_meta_dtype),
        (m_indptr_size, m_indptr_dtype),
        (m_info_size, m_info_dtype),
        (r_indptr_size, r_indptr_dtype),
        (r_final_size, r_final_dtype),
        (r_partial_size, r_partial_dtype),
    ) = _aiter.get_mla_metadata_info_v1(
        batch_size,
        max_seqlen_qo,
        num_q_heads,
        q_dtype,
        kv_dtype,
        is_sparse=0,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
    )
    work_meta = torch.empty(m_meta_size, dtype=m_meta_dtype, device="cuda")
    work_indptr = torch.empty(
        m_indptr_size, dtype=m_indptr_dtype, device="cuda"
    )
    work_info = torch.empty(m_info_size, dtype=m_info_dtype, device="cuda")
    reduce_indptr = torch.empty(
        r_indptr_size, dtype=r_indptr_dtype, device="cuda"
    )
    reduce_final_map = torch.empty(
        r_final_size, dtype=r_final_dtype, device="cuda"
    )
    reduce_partial_map = torch.empty(
        r_partial_size, dtype=r_partial_dtype, device="cuda"
    )

    # Metadata consumes seqlens_kv_indptr in TOKEN units (not pages).
    seqlens_kv_tokens = torch.arange(
        0,
        (batch_size + 1) * cache_len,
        cache_len,
        dtype=torch.int32,
        device="cuda",
    )
    # kv_last_page_lens = ones matches vllm/sglang production. Required
    # positional arg in aiter v0.1.10+; v0.1.7 did not take this.
    kv_last_page_lens_meta = torch.ones(
        batch_size,
        dtype=torch.int32,
        device="cuda",
    )

    # Runtime positional order: (..., work_metadata_ptrs, work_info_set, work_indptr, ...).
    # is_causal=False for decode — aiter's op_test uses False here even for
    # causal attention (qseqlen=1 makes the mask trivial).
    _aiter.get_mla_metadata_v1(
        qo_indptr,
        seqlens_kv_tokens,
        kv_last_page_lens_meta,
        num_q_heads,
        1,
        False,
        work_meta,
        work_info,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=kv_granularity,
        max_seqlen_qo=max_seqlen_qo,
        uni_seqlen_qo=max_seqlen_qo,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )
    return (
        {
            "work_meta_data": work_meta,
            "work_indptr": work_indptr,
            "work_info_set": work_info,
            "reduce_indptr": reduce_indptr,
            "reduce_final_map": reduce_final_map,
            "reduce_partial_map": reduce_partial_map,
        },
        max_split_per_batch,
    )


def bench_aiter_decode(
    batch_size: int,
    cache_len: int,
    config: Config,
    dtype: torch.dtype,
    num_iters: int,
    ncopies: int | None = None,
) -> tuple[float, int] | None:
    """aiter MLA decode (persistent FP8 `_ps` gfx950 kernel), CUDA-graph only.

    Single timing path, mirroring the MAX device-graph path so the two engines
    are compared kernel-to-kernel with NO per-call host dispatch and a
    REALISTIC cold-L2 read:
      * ``ncopies`` independent KV buffers (total > 256 MB MI355 L2) -> each
        replay reads a cold buffer from HBM. No separate flush kernel.
      * ``torch.cuda.CUDAGraph`` capture + replay removes aiter's ~17 us
        per-call Python/torch dispatch (a plain Python loop is CPU-bound and
        would measure dispatch, not the kernel). N graphs (one per KV copy)
        replayed round-robin = rotating + cold. Reports cold us/call.

    KV layout = page_size=1 (vllm/sglang production).
    """
    aiter_mla_mod = _aiter_mla
    if aiter_mla_mod is None or _aiter is None:
        print("aiter not available, skipping bench_aiter_decode")
        return None
    is_fp8 = dtype == torch.float8_e4m3fn
    kv_bpe = 1 if is_fp8 else 2
    kv_bytes_per_copy = batch_size * cache_len * config.qk_head_dim * kv_bpe
    ncopies = _NCOPIES if ncopies is None else ncopies
    if ncopies <= 0:
        ncopies = _auto_ncopies(kv_bytes_per_copy)

    total_pages = batch_size * cache_len
    kv_indptr = torch.arange(
        0,
        (batch_size + 1) * cache_len,
        cache_len,
        dtype=torch.int32,
        device="cuda",
    )
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    # N KV copies -> working set > L2 so each replay reads cold HBM.
    kv_buffers = [
        torch.randn(
            total_pages,
            1,
            1,
            config.qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        ).to(dtype)
        for _ in range(ncopies)
    ]
    qo_indptr = torch.arange(
        0, batch_size + 1, 1, dtype=torch.int32, device="cuda"
    )
    q_torch = torch.randn(
        batch_size,
        config.num_q_heads,
        config.qk_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(dtype)
    # One output buffer per chained call so the chained CUDA graph has no false
    # write-after-write dependency between consecutive calls (which would
    # serialize them, inflating per-op latency). Output is the compressed latent
    # space [batch, heads, kv_lora_rank]; the W_UV up-projection happens OUTSIDE
    # the attention kernel.
    o_list = [
        torch.empty(
            batch_size,
            config.num_q_heads,
            config.kv_lora_rank,
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(ncopies)
    ]
    sm_scale = 1.0 / math.sqrt(config.qk_prefill_head_dim)

    scale_kwargs: dict[str, torch.Tensor] = {}
    if is_fp8:
        scale_kwargs["q_scale"] = torch.ones(
            [1], dtype=torch.float32, device="cuda"
        )
        scale_kwargs["kv_scale"] = torch.ones(
            [1], dtype=torch.float32, device="cuda"
        )

    # aiter dispatches persistent vs non-persistent on whether work_meta_data
    # is passed (mla.py: `persistent_mode = work_meta_data is not None`).
    # Persistent = the `_ps` gfx950 asm kernel; non-persistent = split-K
    # stage1 asm + Triton stage2 reduce (it self-computes num_kv_splits).
    if _AITER_PERSISTENT:
        metadata, max_split_per_batch = _build_mla_persistent_metadata(
            batch_size=batch_size,
            cache_len=cache_len,
            num_q_heads=config.num_q_heads,
            qo_indptr=qo_indptr,
            q_dtype=dtype,
            kv_dtype=dtype,
            kv_granularity=16,
        )

        def call(buf: torch.Tensor, out: torch.Tensor) -> None:
            aiter_mla_mod.mla_decode_fwd(
                q_torch,
                buf,
                out,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                max_seqlen_q=1,
                sm_scale=sm_scale,
                num_kv_splits=max_split_per_batch,
                **metadata,
                **scale_kwargs,
            )

    else:

        def call(buf: torch.Tensor, out: torch.Tensor) -> None:
            # No work_meta_data => non-persistent path.
            aiter_mla_mod.mla_decode_fwd(
                q_torch,
                buf,
                out,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_lens,
                max_seqlen_q=1,
                sm_scale=sm_scale,
                **scale_kwargs,
            )

    nrun = max(num_iters, 200)
    # One CUDA graph chaining `ncopies` decode calls, call i reading a distinct
    # rotating KV buffer. Replaying it sweeps all `ncopies` cold buffers
    # (total > L2) in ONE launch -> cold-HBM, free of the per-replay launch gap
    # a 1-call replay loop pays each iteration. per-op = whole-graph / ncopies.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            for b in range(ncopies):
                call(kv_buffers[b], o_list[b])
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gchain = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gchain):
        for b in range(ncopies):
            call(kv_buffers[b], o_list[b])
    torch.cuda.synchronize()
    for _ in range(50):
        gchain.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(nrun):
        gchain.replay()
    end.record()
    torch.cuda.synchronize()
    per_op_s = start.elapsed_time(end) / 1e3 / nrun / ncopies

    working_set_bytes = kv_bytes_per_copy * ncopies
    kv_mb = kv_bytes_per_copy / (1024.0 * 1024.0)
    cold_state = (
        "cold"
        if working_set_bytes > _L2_CACHE_SIZE_BYTES
        else f"WARM: working set < {_L2_CACHE_SIZE_BYTES / 1e6:.0f}MB L2"
    )
    sched = "persistent" if _AITER_PERSISTENT else "non-persistent"
    print(
        f"[aiter chained CUDA-graph/{sched}] chain={ncopies} "
        f"kv_per_copy~{kv_mb:.1f}MB working_set~{kv_mb * ncopies:.1f}MB ({cold_state}) "
        f"| per-op {per_op_s * 1e6:.2f}us"
    )
    return per_op_s, _compute_decode_bytes(config, batch_size, cache_len, dtype)


def bench_aiter_ps_decode(
    batch_size: int,
    cache_len: int,
    config: Config,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Alias: aiter decode is always the persistent `_ps` kernel now."""
    return bench_aiter_decode(batch_size, cache_len, config, dtype, num_iters)


def _bench_aiter_prefill_bf16(
    batch_size: int,
    qkv_len: int,
    config: Config,
    num_iters: int,
) -> tuple[float, int] | None:
    """BF16 MLA prefill via `aiter.flash_attn_varlen_func` on decompressed
    Q/K/V. This is the vllm + sglang production path:
    `vllm/v1/attention/backends/mla/rocm_aiter_mla.py:443-455`,
    `sglang/srt/layers/attention/aiter_backend.py:2138-2148`.
    """
    aiter_mod = _aiter
    if aiter_mod is None or not hasattr(aiter_mod, "flash_attn_varlen_func"):
        print(
            "aiter.flash_attn_varlen_func not present in this wheel. Skipping."
        )
        return None

    total_tokens = batch_size * qkv_len
    q = torch.randn(
        total_tokens,
        config.num_q_heads,
        config.qk_prefill_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        total_tokens,
        config.num_q_heads,
        config.qk_prefill_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn(
        total_tokens,
        config.num_q_heads,
        config.v_head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    )

    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        qkv_len,
        dtype=torch.int32,
        device="cuda",
    )
    softmax_scale = 1.0 / math.sqrt(config.qk_prefill_head_dim)

    def run_kernel() -> Any:
        return aiter_mod.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            qkv_len,
            qkv_len,
            softmax_scale=softmax_scale,
            causal=True,
        )

    try:
        time_s = _time_cuda_graph(run_kernel)
        print(f"[aiter prefill CUDA-graph/bf16] replay {time_s * 1e6:.2f}us")
    except Exception as e:
        print(f"  (aiter prefill CUDA-graph failed, eager: {e})")
        time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    return time_s, _compute_prefill_flops(config, batch_size, qkv_len, True)


def _bench_aiter_prefill_fp8(
    batch_size: int,
    qkv_len: int,
    config: Config,
    num_iters: int,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[float, int] | None:
    """FP8 MLA prefill via `aiter.mla_prefill_ps_asm_fwd` + `aiter.mla_reduce_v1`.

    Recipe ported from `aiter/op_tests/test_mla_prefill_ps.py` (the upstream
    test harness) and sglang's `aiter_backend.py:586-658`. Persistent
    scheduling with:
      - `block_size=1` (per-token KV layout)
      - `tile_q=256`, `tile_kv=128`
      - `qhead_granularity=gqa_ratio`, `qlen_granularity=tile_q/gqa_ratio`
      - `kvlen_granularity=max(tile_kv, block_size)=128`
    Q/K pre-quantized to FP8; V sliced from K to `v_head_dim`.
    """
    aiter_mod = _aiter
    if aiter_mod is None:
        print("aiter not available, skipping bench_aiter_prefill_fp8")
        return None
    required = [
        "get_ps_metadata_info_v1",
        "get_ps_metadata_v1",
        "mla_prefill_ps_asm_fwd",
        "mla_reduce_v1",
    ]
    missing = [n for n in required if not hasattr(aiter_mod, n)]
    if missing:
        print(
            f"aiter FP8 MLA prefill requires {missing}, not present in this "
            f"wheel (aiter {getattr(aiter_mod, '__version__', '?')}). Skipping."
        )
        return None

    device = "cuda"
    num_head_q = config.num_q_heads
    num_head_kv = config.num_q_heads  # matches aiter op_test: gqa_ratio=1
    gqa_ratio = num_head_q // num_head_kv
    block_size = 1
    tile_q = 256
    tile_kv = 128
    qhead_granularity = gqa_ratio
    qlen_granularity = tile_q // qhead_granularity
    kvlen_granularity = max(tile_kv, block_size)

    softmax_scale = 1.0 / math.sqrt(config.qk_prefill_head_dim)

    # qo_indptr / kv_indptr with uniform batch.
    seq_lens_kv = torch.full(
        (batch_size,),
        qkv_len,
        dtype=torch.int32,
        device=device,
    )
    qo_indptr = torch.arange(
        0,
        (batch_size + 1) * qkv_len,
        qkv_len,
        dtype=torch.int32,
        device=device,
    )
    actual_blocks = (seq_lens_kv + block_size - 1) // block_size
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    num_blocks = int(kv_indptr[-1].item())
    kv_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)

    # Pre-quantize Q/K/V from BF16 to FP8 per-tensor.
    num_tokens = batch_size * qkv_len
    q_bf16 = torch.randn(
        num_tokens,
        num_head_q,
        config.qk_prefill_head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k_bf16 = torch.randn(
        num_blocks,
        num_head_kv,
        config.qk_prefill_head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v_bf16 = k_bf16[:, :, : config.v_head_dim].contiguous()
    q = q_bf16.to(dtype).contiguous()
    k = k_bf16.to(dtype).contiguous()
    v = v_bf16.to(dtype).contiguous()
    one = torch.ones([1], dtype=torch.float32, device=device)

    # Persistent metadata via get_ps_metadata_info_v1 + get_ps_metadata_v1.
    (
        (m_meta_sz, m_meta_dt),
        (m_indptr_sz, m_indptr_dt),
        (m_info_sz, m_info_dt),
        (r_indptr_sz, r_indptr_dt),
        (r_final_sz, r_final_dt),
        (r_partial_sz, r_partial_dt),
    ) = aiter_mod.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_head_kv,
        max_qlen=qkv_len,
        qlen_granularity=qlen_granularity,
    )
    work_meta = torch.empty(m_meta_sz, dtype=m_meta_dt, device=device)
    work_indptr = torch.empty(m_indptr_sz, dtype=m_indptr_dt, device=device)
    work_info = torch.empty(m_info_sz, dtype=m_info_dt, device=device)
    reduce_indptr = torch.empty(r_indptr_sz, dtype=r_indptr_dt, device=device)
    reduce_final_map = torch.empty(r_final_sz, dtype=r_final_dt, device=device)
    reduce_partial_map = torch.empty(
        r_partial_sz, dtype=r_partial_dt, device=device
    )

    # metadata v1 wants CPU inputs.
    aiter_mod.get_ps_metadata_v1(
        qo_indptr.to("cpu"),
        kv_indptr.to("cpu"),
        seq_lens_kv.to("cpu"),
        gqa_ratio,
        num_head_kv,
        work_meta,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qhead_granularity=qhead_granularity,
        qlen_granularity=qlen_granularity,
        kvlen_granularity=kvlen_granularity,
        block_size=block_size,
        is_causal=True,
    )

    output = torch.empty(
        num_tokens,
        num_head_q,
        config.v_head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    logits = torch.empty(
        reduce_partial_map.size(0) * tile_q,
        num_head_q,
        config.v_head_dim,
        dtype=torch.float32,
        device=device,
    )
    attn_lse = torch.empty(
        reduce_partial_map.size(0) * tile_q,
        num_head_q,
        dtype=torch.float32,
        device=device,
    )
    final_lse = torch.empty(
        num_tokens,
        num_head_q,
        dtype=torch.float32,
        device=device,
    )

    def run_kernel() -> Any:
        aiter_mod.mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            work_indptr,
            work_info,
            qkv_len,
            softmax_scale,
            True,
            logits,
            attn_lse,
            output,
            one,
            one,
            one,
        )
        aiter_mod.mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )

    try:
        time_s = _time_cuda_graph(run_kernel)
        print(f"[aiter prefill CUDA-graph/fp8] replay {time_s * 1e6:.2f}us")
    except Exception as e:
        print(f"  (aiter prefill CUDA-graph failed, eager: {e})")
        time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    return time_s, _compute_prefill_flops(config, batch_size, qkv_len, True)


def bench_aiter_prefill(
    batch_size: int,
    qkv_len: int,
    config: Config,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """aiter MLA prefill benchmark.

    - BF16: `aiter.flash_attn_varlen_func` on decompressed Q/K/V
      (vllm + sglang production path).
    - FP8: `aiter.mla_prefill_ps_asm_fwd` + `aiter.mla_reduce_v1` with
      persistent-scheduling metadata from `get_ps_metadata_v1`. Requires
      aiter >= v0.1.10 (sglang production path).
    """
    if _aiter is None:
        print("aiter not available, skipping bench_aiter_prefill")
        return None

    is_fp8 = dtype == torch.float8_e4m3fn
    if is_fp8:
        return _bench_aiter_prefill_fp8(batch_size, qkv_len, config, num_iters)
    return _bench_aiter_prefill_bf16(batch_size, qkv_len, config, num_iters)


# ----------------------------------------------------------------------------
# Dispatch and CLI.
# ----------------------------------------------------------------------------


_ENGINE_MAP: dict[tuple[str, str], Callable[..., tuple[float, int] | None]] = {
    ("modular_max", "prefill"): bench_max_prefill,
    ("modular_max", "decode"): bench_max_decode,
    ("aiter", "prefill"): bench_aiter_prefill,
    ("aiter", "decode"): bench_aiter_decode,
    # aiter_ps = persistent scheduling for all dtypes (vllm post-PR #36574).
    # Only decode has a meaningful persistent variant.
    ("aiter_ps", "decode"): bench_aiter_ps_decode,
}


def bench_mla(
    mode: str,
    engine: str,
    batch_size: int,
    seq_len: int,  # qkv_len for prefill, cache_len for decode
    config: Config,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    print("=" * 80)
    print(
        f"AMD MLA {mode.title()} (batch={batch_size},"
        f" {'qkv_len' if mode == 'prefill' else 'cache_len'}={seq_len},"
        f" heads={config.num_q_heads}, dtype={dtype}, engine={engine})"
    )
    print("=" * 80)

    fn = _ENGINE_MAP.get((engine, mode))
    if fn is None:
        raise ValueError(f"Unknown (engine, mode)=({engine}, {mode})")

    try:
        result = fn(batch_size, seq_len, config, dtype, num_iters)
    except Exception as e:
        print(f"{engine} {mode} benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return None

    if result is not None:
        time_s, units = result
        if mode == "prefill":
            tflops = units / time_s / 1e12
            print(f"  Time: {time_s * 1e3:.3f} ms | {tflops:.2f} TFLOPS")
        else:
            tb_s = units / time_s / 1e12
            print(f"  Time: {time_s * 1e3:.3f} ms | {tb_s:.2f} TB/s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="AMD MLA Benchmark")
    parser.add_argument(
        "--mode",
        choices=["prefill", "decode"],
        default="decode",
    )
    parser.add_argument(
        "--engine",
        choices=["modular_max", "aiter", "aiter_ps"],
        default="modular_max",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float8_e4m3fn"],
        default="bfloat16",
    )
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument(
        "--qkv_len",
        "--qkv-len",
        type=int,
        default=1024,
        help="Prefill Q/K/V sequence length",
    )
    parser.add_argument(
        "--cache_len",
        "--cache-len",
        type=int,
        default=4096,
        help="Decode KV cache length",
    )
    parser.add_argument("--num_q_heads", "--num-q-heads", type=int, default=64)
    parser.add_argument(
        "--qk_nope_head_dim",
        "--qk-nope-head-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--qk_rope_head_dim",
        "--qk-rope-head-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--kv_lora_rank",
        "--kv-lora-rank",
        type=int,
        default=512,
    )
    parser.add_argument("--v_head_dim", "--v-head-dim", type=int, default=128)
    parser.add_argument("--num_iters", "--num-iters", type=int, default=100)
    parser.add_argument("--output", "-o", type=str, default="output.csv")
    parser.add_argument(
        "--ncopies",
        type=int,
        default=0,
        help=(
            "Rotating KV copies for graph-replay decode (0 = auto: "
            "footprint > L2, capped at 16)."
        ),
    )
    parser.add_argument(
        "--aiter-sched",
        "--aiter_sched",
        choices=["persistent", "nonpersistent"],
        default="persistent",
        help=(
            "aiter decode kernel: persistent `_ps` asm (default) or "
            "non-persistent split-K + Triton reduce."
        ),
    )
    args, _ = parser.parse_known_args()

    # Decode benches always use the overhead-free graph-replay path with
    # `--ncopies` rotating KV buffers.
    global _NCOPIES, _AITER_PERSISTENT
    _NCOPIES = args.ncopies
    _AITER_PERSISTENT = args.aiter_sched == "persistent"

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }
    dtype = dtype_map[args.dtype]
    config = Config(
        num_q_heads=args.num_q_heads,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
        v_head_dim=args.v_head_dim,
    )
    seq_len = args.qkv_len if args.mode == "prefill" else args.cache_len

    print(
        (
            f"[bench_amd_mla] mode={args.mode} engine={args.engine} "
            f"dtype={args.dtype} batch_size={args.batch_size} "
            f"{'qkv_len' if args.mode == 'prefill' else 'cache_len'}={seq_len} "
            f"num_q_heads={config.num_q_heads} qk_nope={config.qk_nope_head_dim} "
            f"qk_rope={config.qk_rope_head_dim} kv_lora_rank={config.kv_lora_rank} "
            f"v_head_dim={config.v_head_dim}"
        ),
        file=sys.stderr,
    )

    result = bench_mla(
        args.mode,
        args.engine,
        args.batch_size,
        seq_len,
        config,
        dtype,
        args.num_iters,
    )

    if result is None:
        sys.exit(1)

    time_s, units = result
    metric = (
        ThroughputMeasure(Bench.flops, units)
        if args.mode == "prefill"
        else ThroughputMeasure(Bench.bytes, units)
    )
    name = (
        f"MLA_{args.mode.title()}/batch_size={args.batch_size}/"
        f"{'qkv_len' if args.mode == 'prefill' else 'cache_len'}={seq_len}/"
        f"num_q_heads={config.num_q_heads}/"
        f"qk_nope_head_dim={config.qk_nope_head_dim}/"
        f"qk_rope_head_dim={config.qk_rope_head_dim}/"
        f"kv_lora_rank={config.kv_lora_rank}/"
        f"v_head_dim={config.v_head_dim}/"
        f"dtype={args.dtype}/engine={args.engine}/"
    )
    b = Bench(name, iters=1, met=time_s, metric_list=[metric])
    b.dump_report(output_path=args.output)


if __name__ == "__main__":
    main()
