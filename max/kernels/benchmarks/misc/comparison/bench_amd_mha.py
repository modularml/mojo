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

# AMD MHA prefill benchmark comparing MAX against aiter backends (CK v2/v3)
# and torch SDPA baselines.
# Run via kbench: kbench bench_amd_mha.yaml

from __future__ import annotations

import argparse
import math
import os
import types
from collections.abc import Callable
from functools import partial
from typing import Any

# Configure aiter JIT environment before any imports.
# When run via kbench (file: mode), the Bazel env vars aren't applied,
# so we set them here as fallbacks.
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

import torch
from bencher_utils import Bench, ThroughputMeasure

# MAX imports
from max.driver import Accelerator
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, TensorType
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu

# aiter imports (JIT-compiled on first use via AITER_JIT_DIR).
_aiter: types.ModuleType | None
try:
    import aiter as _aiter
except (ImportError, Exception) as e:
    print(f"Warning: aiter not available: {e}")
    _aiter = None


# MI355X L2 cache size (256 MB) used for cache flushing between iterations.
_L2_CACHE_SIZE_BYTES = int(256e6)


def _bench_cuda_events(
    fn: Callable[[], Any],
    num_warmups: int = 50,
    num_iters: int = 100,
    flush_l2: bool = True,
) -> float:
    """Benchmark using CUDA events. Returns median time in seconds."""
    torch.cuda.synchronize()

    # Warmup
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

    times = []
    for _ in range(num_iters):
        if flush_buffer is not None:
            flush_buffer.zero_()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    # Return median time in seconds
    times.sort()
    median_ms = times[len(times) // 2]
    return median_ms / 1e3


def _compute_flops(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    head_dim: int,
    causal: bool,
) -> int:
    if causal:
        return batch_size * qkv_len * qkv_len * num_q_heads * head_dim * 2
    else:
        return batch_size * qkv_len * qkv_len * num_q_heads * head_dim * 4


def bench_aiter(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Benchmark using aiter.flash_attn_func (auto CK v2/v3 dispatch)."""
    if _aiter is None:
        print("aiter not available, skipping bench_aiter")
        return None

    q = torch.randn(
        batch_size, qkv_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    def run_kernel() -> torch.Tensor:
        out = _aiter.flash_attn_func(q, k, v, causal=causal, deterministic=True)
        return out

    time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    flops = _compute_flops(batch_size, qkv_len, num_q_heads, head_dim, causal)
    return time_s, flops


def bench_aiter_ck_v3(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Benchmark using aiter.fmha_v3_fwd (CK v3 backend directly)."""
    if _aiter is None:
        print("aiter not available, skipping bench_aiter_ck_v3")
        return None

    softmax_scale = head_dim ** (-0.5)

    q = torch.randn(
        batch_size, qkv_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    def run_kernel() -> torch.Tensor:
        out, _, _, _ = _aiter.fmha_v3_fwd(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            return_softmax_lse=False,
            return_dropout_randval=False,
            how_v3_bf16_cvt=1,
        )
        return out

    time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    flops = _compute_flops(batch_size, qkv_len, num_q_heads, head_dim, causal)
    return time_s, flops


def bench_aiter_ck_v2(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Benchmark using aiter.mha_fwd (CK v2 backend directly)."""
    if _aiter is None:
        print("aiter not available, skipping bench_aiter_ck_v2")
        return None

    softmax_scale = head_dim ** (-0.5)

    q = torch.randn(
        batch_size, qkv_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    def run_kernel() -> torch.Tensor:
        out, _, _, _ = _aiter.mha_fwd(
            q,
            k,
            v,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            is_causal=causal,
            window_size_left=-1,
            window_size_right=-1,
            sink_size=0,
            return_softmax_lse=False,
            return_dropout_randval=False,
        )
        return out

    time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    flops = _compute_flops(batch_size, qkv_len, num_q_heads, head_dim, causal)
    return time_s, flops


def bench_torch_sdpa(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Benchmark using torch.nn.functional.scaled_dot_product_attention."""
    # SDPA expects (batch, heads, seq_len, head_dim)
    q = torch.randn(
        batch_size, num_q_heads, qkv_len, head_dim, dtype=dtype, device="cuda"
    )
    # Expand KV heads for GQA
    k = torch.randn(
        batch_size, num_kv_heads, qkv_len, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, num_kv_heads, qkv_len, head_dim, dtype=dtype, device="cuda"
    )

    # SDPA handles GQA via broadcasting if num_q_heads is a multiple of num_kv_heads
    gqa_ratio = num_q_heads // num_kv_heads
    if gqa_ratio > 1:
        k = (
            k.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1, -1)
            .reshape(batch_size, num_q_heads, qkv_len, head_dim)
        )
        v = (
            v.unsqueeze(2)
            .expand(-1, -1, gqa_ratio, -1, -1)
            .reshape(batch_size, num_q_heads, qkv_len, head_dim)
        )

    def run_kernel() -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal
        )

    time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    flops = _compute_flops(batch_size, qkv_len, num_q_heads, head_dim, causal)
    return time_s, flops


def bench_max(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Benchmark MAX flash_attention_gpu kernel."""
    max_dtype = torch_dtype_to_max(dtype)

    q = torch.randn(
        batch_size, qkv_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    q_type = TensorType(
        max_dtype,
        shape=["batch", "seq_len", num_q_heads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        max_dtype,
        shape=["batch", "seq_len", num_kv_heads, head_dim],
        device=DeviceRef.GPU(),
    )

    session = InferenceSession(devices=[Accelerator()])

    mask_variant = (
        MHAMaskVariant.CAUSAL_MASK if causal else MHAMaskVariant.NULL_MASK
    )
    graph = Graph(
        "flash_attn_max",
        forward=partial(
            flash_attention_gpu,
            scale=math.sqrt(1.0 / head_dim),
            mask_variant=mask_variant,
        ),
        input_types=[q_type, kv_type, kv_type],
    )

    model = session.load(graph)

    def run_kernel() -> torch.Tensor:
        output = model.execute(q.detach(), k.detach(), v.detach())[0]
        return output

    time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    flops = _compute_flops(batch_size, qkv_len, num_q_heads, head_dim, causal)
    return time_s, flops


def bench_aiter_triton_v2(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Benchmark using aiter triton flash attention v2."""
    if _aiter is None:
        print("aiter not available, skipping bench_aiter_triton_v2")
        return None
    try:
        from aiter.ops.triton.mha import (  # type: ignore[import-not-found]
            flash_attn_func as triton_fa2,
        )
    except ImportError as e:
        print(f"aiter triton v2 not available: {e}")
        return None

    q = torch.randn(
        batch_size, qkv_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    def run_kernel() -> torch.Tensor:
        out = triton_fa2(q, k, v, causal=causal, deterministic=True)
        return out

    time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    flops = _compute_flops(batch_size, qkv_len, num_q_heads, head_dim, causal)
    return time_s, flops


def bench_aiter_triton_v3(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
) -> tuple[float, int] | None:
    """Benchmark using aiter triton flash attention v3."""
    if _aiter is None:
        print("aiter not available, skipping bench_aiter_triton_v3")
        return None
    try:
        from aiter.ops.triton.mha_v3 import (  # type: ignore[import-not-found]
            flash_attn_func as triton_fa3,
        )
    except ImportError as e:
        print(f"aiter triton v3 not available: {e}")
        return None

    q = torch.randn(
        batch_size, qkv_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, qkv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    def run_kernel() -> torch.Tensor:
        out = triton_fa3(q, k, v, causal=causal, deterministic=True)
        return out

    time_s = _bench_cuda_events(run_kernel, num_iters=num_iters)
    flops = _compute_flops(batch_size, qkv_len, num_q_heads, head_dim, causal)
    return time_s, flops


_ENGINE_MAP: dict[
    str,
    Callable[..., tuple[float, int] | None],
] = {
    "modular_max": bench_max,
    "aiter": bench_aiter,
    "aiter_ck_v3": bench_aiter_ck_v3,
    "aiter_ck_v2": bench_aiter_ck_v2,
    "aiter_triton_v2": bench_aiter_triton_v2,
    "aiter_triton_v3": bench_aiter_triton_v3,
    "torch_sdpa": bench_torch_sdpa,
}


def bench_prefill(
    batch_size: int,
    qkv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    engine: str,
    num_iters: int,
) -> tuple[float, int] | None:
    """Run MHA prefill benchmark for specified engine.

    Args:
        batch_size: Batch size.
        qkv_len: Sequence length for Q, K, V.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Dimension of each head.
        causal: Whether to use causal masking.
        dtype: Torch dtype for inputs.
        engine: Backend engine name.
        num_iters: Number of benchmark iterations.
    """
    print("=" * 80)
    print(
        f"AMD MHA Prefill Benchmark (batch={batch_size}, seq_len={qkv_len},"
        f" q_heads={num_q_heads}, kv_heads={num_kv_heads},"
        f" head_dim={head_dim}, causal={causal}, engine={engine})"
    )
    print("=" * 80)

    bench_fn = _ENGINE_MAP.get(engine)
    if bench_fn is None:
        raise ValueError(
            f"Unknown engine '{engine}'. Available: {list(_ENGINE_MAP.keys())}"
        )

    try:
        result = bench_fn(
            batch_size,
            qkv_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            causal,
            dtype,
            num_iters,
        )
    except Exception as e:
        print(f"{engine} benchmark failed: {e}")
        result = None

    if result is not None:
        time_s, flops = result
        tflops = flops / time_s / 1e12
        print(f"  Time: {time_s * 1e3:.3f} ms | {tflops:.2f} TFLOPS")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMD MHA Prefill Benchmark")
    parser.add_argument(
        "--batch_size", "--batch-size", type=int, default=16, help="Batch size"
    )
    parser.add_argument(
        "--qkv_len",
        "--qkv-len",
        type=int,
        default=4096,
        help="QKV length",
    )
    parser.add_argument(
        "--num_q_heads",
        "--num-q-heads",
        type=int,
        default=64,
        help="Number of query heads",
    )
    parser.add_argument(
        "--num_kv_heads",
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads (defaults to num_q_heads if not specified)",
    )
    parser.add_argument(
        "--head_dim", "--head-dim", type=int, default=128, help="Head dimension"
    )
    parser.add_argument(
        "--causal",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="Causal masking",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Data type",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="modular_max",
        choices=list(_ENGINE_MAP.keys()),
        help="Engine",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output path",
    )
    parser.add_argument(
        "--num_iters",
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    args, _ = parser.parse_known_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    num_kv_heads = (
        args.num_kv_heads if args.num_kv_heads is not None else args.num_q_heads
    )

    result = bench_prefill(
        batch_size=args.batch_size,
        qkv_len=args.qkv_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=args.head_dim,
        causal=args.causal,
        dtype=dtype_map[args.dtype],
        engine=args.engine,
        num_iters=args.num_iters,
    )

    if result is not None:
        time_s, flops = result
        flops_per_sec = ThroughputMeasure(Bench.flops, flops)
        name = (
            f"MHA_Prefill/batch_size={args.batch_size}/qkv_len={args.qkv_len}/"
            f"num_q_heads={args.num_q_heads}/num_kv_heads={num_kv_heads}/"
            f"head_dim={args.head_dim}/"
            f"causal={args.causal}/dtype={dtype_map[args.dtype]}/"
            f"engine={args.engine}/"
        )
        b = Bench(
            name,
            iters=1,
            met=time_s,
            metric_list=[flops_per_sec],
        )
        b.dump_report(output_path=args.output)
