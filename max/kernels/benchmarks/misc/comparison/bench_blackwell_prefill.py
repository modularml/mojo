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

# Blackwell prefill benchmark comparing MAX against FlashInfer and flash-attention baselines.
# Run via kbench: kbench bench_prefill.yaml

from __future__ import annotations

import argparse
import math
import types
from collections.abc import Callable
from functools import partial
from typing import Any

import torch

# Import bench utilities from Bazel dependency (bench_utils target)
from bench import bench_kineto_with_cupti_warmup, setup_ninja_path
from bencher_utils import Bench, ThroughputMeasure

# MAX imports
from max.driver import Accelerator
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, TensorType
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu

# Try importing external libraries (installed via Bazel pycross_wheel_library)
_flashinfer: types.ModuleType | None
try:
    setup_ninja_path()  # Required for FlashInfer JIT compilation
    import flashinfer as _flashinfer
except ImportError as e:
    print(f"Error: flashinfer not available: {e}")
    _flashinfer = None

_flash_attn_varlen_func: Callable[..., Any] | None
try:
    # The pure Python flash-attention wheel's __init__.py tries to import
    # flash_attn_2_cuda (CUDA extension not in pure wheel).
    # Bypass this by creating a stub flash_attn module with valid __path__
    # but no imports.
    import importlib.util
    import sys

    flash_attn_spec = importlib.util.find_spec("flash_attn")
    if (
        flash_attn_spec is None
        or flash_attn_spec.submodule_search_locations is None
    ):
        raise ImportError("flash_attn package not found")

    # Create stub module with valid __path__ but no imports
    flash_attn_stub = types.ModuleType("flash_attn")
    flash_attn_stub.__path__ = list(flash_attn_spec.submodule_search_locations)
    flash_attn_stub.__file__ = flash_attn_spec.origin
    sys.modules["flash_attn"] = flash_attn_stub

    # Now import the cute subpackage and interface
    from flash_attn.cute.interface import (
        flash_attn_varlen_func as _flash_attn_varlen_func,
    )
except ImportError as e:
    print(f"Error: flash_attn not available: {e}")
    _flash_attn_varlen_func = None


def _attention_flops(
    batch_size: int,
    q_len: int,
    kv_len: int,
    num_q_heads: int,
    head_dim: int,
    causal: bool,
    window_size: int,
) -> int:
    """Compute the FLOPs for one MHA prefill pass.

    Q has ``q_len`` tokens and K/V have ``kv_len = cache_len + q_len`` tokens, so
    ``cache_len = kv_len - q_len`` is the already-cached prefix. Each valid
    (query, key) pair costs ``4 * head_dim`` FLOPs (``QK^T`` and ``P@V`` each
    contribute ``2 * head_dim``).

    A ``window_size > 0`` selects a causal sliding window, so the i-th new query
    (absolute position ``cache_len + i``) attends to at most ``window_size``
    keys; ``window_size <= 0`` falls back to full causal or full bidirectional
    attention.

    Args:
        batch_size: Batch size.
        q_len: Query sequence length.
        kv_len: Key/value sequence length (``cache_len + q_len``).
        num_q_heads: Number of query heads.
        head_dim: Dimension of each head.
        causal: Whether causal masking is applied.
        window_size: Causal sliding window size (keys per query, itself
            included); ``<= 0`` disables the window.

    Returns:
        The total floating-point operation count.
    """
    if window_size > 0:
        # Causal sliding window with a cache prefix: the i-th new query sits at
        # absolute position cache_len + i and attends to keys
        # [pos - window_size + 1, pos], i.e. min(pos + 1, window_size) keys.
        # Summing min(cache_len + i + 1, window_size) over the q_len new queries
        # reduces exactly to qkv_len * w - w * (w - 1) / 2 when cache_len == 0.
        cache_len = kv_len - q_len
        valid_pairs = sum(
            min(cache_len + i + 1, window_size) for i in range(q_len)
        )
        return batch_size * num_q_heads * head_dim * 4 * valid_pairs
    if causal:
        # Prefill-with-cache: query i attends to ~(cache_len + i + 1) keys, so
        # the average attended length is (cache_len + kv_len) / 2, which equals
        # kv_len - q_len/2. This matches the flash-attention varlen FLOP
        # convention and reduces to 2 * B * S^2 * H * D exactly when
        # kv_len == q_len (the pure-prefill case).
        return (
            2
            * batch_size
            * q_len
            * (2 * kv_len - q_len)
            * num_q_heads
            * head_dim
        )
    return 4 * batch_size * q_len * kv_len * num_q_heads * head_dim


def bench_flashinfer(
    batch_size: int,
    q_len: int,
    cache_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
    window_size: int = -1,
    backend: str = "cutlass",
    no_kineto: bool = False,
) -> tuple[float, int] | None:
    if _flashinfer is None:
        print("flashinfer not available, skipping bench_flashinfer")
        return None

    # The SM100 cutlass FMHA backend used here does not support sliding window
    # attention: BatchPrefillWithRaggedKVCacheWrapper.run() routes cutlass
    # through fmha_varlen(), which has no window_left argument and builds its
    # module with use_sliding_window=False, so window_left is silently dropped.
    # Skip rather than report full-causal numbers mislabeled as windowed.
    if window_size > 0:
        print(
            "FlashInfer SM100 cutlass backend does not support sliding window"
            " attention, skipping bench_flashinfer."
        )
        return None

    kv_len = cache_len + q_len

    # Validate backend option
    available_backends = ["auto", "fa2", "fa3", "trtllm-gen", "cutlass"]
    assert backend in available_backends, (
        f"backend must be one of {available_backends}, got {backend}"
    )

    # Note: trtllm-gen backend doesn't support variable length yet
    if backend == "trtllm-gen":
        print(
            "Warning: trtllm-gen backend doesn't support variable length yet,"
            " skipping..."
        )
        return None

    q = torch.randn(
        batch_size * q_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    qo_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * q_len
    )
    kv_segment_offsets = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * kv_len
    )
    wrapper = _flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(
            128 * 1024 * 1024, dtype=dtype, device="cuda"
        ),  # work space
        kv_layout="NHD",
        backend=backend,
    )
    wrapper.plan(
        qo_segment_offsets,
        kv_segment_offsets,
        num_q_heads,
        num_kv_heads,
        head_dim,
        head_dim_vo=head_dim,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    def run_kernel() -> torch.Tensor:
        return wrapper.run(q, k, v)

    # Select kernel name pattern based on backend
    if backend == "cutlass":
        kernel_name = "Sm100FmhaFwdKernelTmaWarpspecialized"
    elif backend == "fa3":
        kernel_name = "fmha"  # FlashAttention 3 kernel pattern
    elif backend == "fa2":
        kernel_name = "fmha"  # FlashAttention 2 kernel pattern
    else:  # auto
        kernel_name = "fmha"  # Generic pattern

    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return None

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup for CUTLASS
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names=kernel_name,
        num_tests=num_iters,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    return time_s, _attention_flops(
        batch_size, q_len, kv_len, num_q_heads, head_dim, causal, window_size
    )


def bench_max(
    batch_size: int,
    q_len: int,
    cache_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
    window_size: int = -1,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
    """Benchmark MAX flash_attention_gpu kernel.

    Q has ``q_len`` tokens and K/V have ``kv_len = cache_len + q_len`` tokens.
    The MHA kernel derives the cache prefix automatically from the differing
    sequence lengths (``start_pos = kv_len - q_len``) and bottom-right-aligns
    the causal mask, so no explicit mask is required.

    Args:
        batch_size: Batch size
        q_len: Query sequence length
        cache_len: Already-cached prefix length (K/V have cache_len + q_len)
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        causal: Whether to use causal masking
        dtype: torch dtype for inputs (e.g., torch.bfloat16)
        window_size: Sliding window size (number of keys each query attends
            to, itself included). When > 0, a causal sliding window mask is
            used; otherwise the mask is full causal or null per ``causal``.
    """
    kv_len = cache_len + q_len

    # Convert torch dtype to MAX DType
    max_dtype = torch_dtype_to_max(dtype)

    # Create input tensors in (batch, seq_len, num_heads, head_dim) format.
    # Q is q_len long; K/V are kv_len long (cache prefix + new tokens).
    q = torch.randn(
        batch_size, q_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    # Define tensor types for MAX graph. Use distinct symbolic dims for the Q
    # and KV sequence lengths so the compiler does not unify them.
    q_type = TensorType(
        max_dtype,
        shape=["batch", "q_len", num_q_heads, head_dim],
        device=DeviceRef.GPU(),
    )
    kv_type = TensorType(
        max_dtype,
        shape=["batch", "kv_len", num_kv_heads, head_dim],
        device=DeviceRef.GPU(),
    )

    # Create inference session
    session = InferenceSession(devices=[Accelerator()])

    # Construct MAX graph. A positive window_size selects the causal sliding
    # window mask (SlidingWindowCausalMask[window_size], where each query
    # attends to keys [q - window_size + 1, q]); local_window_size feeds that
    # parameter directly. Otherwise fall back to full causal / null masking.
    if window_size > 0:
        mask_variant = MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
    else:
        mask_variant = (
            MHAMaskVariant.CAUSAL_MASK if causal else MHAMaskVariant.NULL_MASK
        )
    graph = Graph(
        "flash_attn_max",
        forward=partial(
            flash_attention_gpu,
            scale=math.sqrt(1.0 / head_dim),
            mask_variant=mask_variant,
            local_window_size=window_size if window_size > 0 else -1,
        ),
        input_types=[q_type, kv_type, kv_type],
    )

    # Compile model
    model = session.load(graph)

    def run_kernel() -> torch.Tensor:
        output = model.execute(q.detach(), k.detach(), v.detach())[0]
        return output

    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return None

    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="mha",
        num_tests=num_iters,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    return time_s, _attention_flops(
        batch_size, q_len, kv_len, num_q_heads, head_dim, causal, window_size
    )


def bench_tridao(
    batch_size: int,
    q_len: int,
    cache_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    num_iters: int,
    window_size: int = -1,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
    if _flash_attn_varlen_func is None:
        print("flash_attn not available, skipping bench_tridao")
        return None

    kv_len = cache_len + q_len

    # Create input tensors in varlen format (similar to test_flash_attn_varlen_output).
    # Q is q_len long; K/V are kv_len long (cache prefix + new tokens).
    q = torch.randn(
        batch_size * q_len, num_q_heads, head_dim, dtype=dtype, device="cuda"
    )
    k = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )
    v = torch.randn(
        batch_size * kv_len, num_kv_heads, head_dim, dtype=dtype, device="cuda"
    )

    # Create cumulative sequence length offsets
    cu_seqlens_q = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * q_len
    )
    cu_seqlens_k = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32)
        * kv_len
    )

    # tridao expresses the window as (window_size_left, window_size_right).
    # MAX's window_size W (keys [q - W + 1, q]) maps to left = W - 1, right = 0
    # (causal). (None, None) preserves the dense causal / non-causal behavior.
    window_arg = (window_size - 1, 0) if window_size > 0 else (None, None)

    def run_kernel() -> torch.Tensor:
        assert _flash_attn_varlen_func is not None
        out, _ = _flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            causal=causal,
            window_size=window_arg,
            pack_gqa=False,
        )
        return out

    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return None

    # Use bench_kineto_with_cupti_warmup to handle CUPTI warmup
    time_s = bench_kineto_with_cupti_warmup(
        run_kernel,
        kernel_names="kernel_cutlass_kernel_flash_attncuteflash_fwd_sm100FlashAttentionForwardSm100",
        num_tests=num_iters,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    return time_s, _attention_flops(
        batch_size, q_len, kv_len, num_q_heads, head_dim, causal, window_size
    )


def bench_prefill(
    batch_size: int,
    q_len: int,
    cache_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    causal: bool,
    dtype: torch.dtype,
    engine: str,
    num_iters: int,
    window_size: int = -1,
    no_kineto: bool = False,
) -> tuple[float, int] | None:
    """Run all MHA prefill benchmarks and display results side-by-side.

    Args:
        batch_size: Batch size
        q_len: Query sequence length
        cache_len: Already-cached prefix length (K/V have cache_len + q_len)
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        causal: Whether to use causal masking
        dtype: torch dtype for inputs (e.g., torch.bfloat16)
        engine: backend to run the benchmark ("flashinfer" or "tridao" or "modular_max")
        num_iters: Number of benchmark iters.
        window_size: Sliding window size (keys per query, itself included).
            When > 0, a causal sliding window mask is used. FlashInfer is
            skipped in this case (its SM100 cutlass backend lacks SWA support).
    """
    print("=" * 80)
    print(
        f"MHA Prefill Benchmark (batch={batch_size}, q_len={q_len},"
        f" cache_len={cache_len}, kv_len={cache_len + q_len},"
        f" q_heads={num_q_heads}, kv_heads={num_kv_heads},"
        f" head_dim={head_dim}, causal={causal}, window_size={window_size})"
    )
    print("=" * 80)

    result: tuple[float, int] | None = None

    if engine == "flashinfer":
        # Run FlashInfer benchmark
        if _flashinfer is not None:
            try:
                result = bench_flashinfer(
                    batch_size,
                    q_len,
                    cache_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    causal,
                    dtype,
                    num_iters,
                    window_size=window_size,
                    backend="cutlass",
                    no_kineto=no_kineto,
                )
            except Exception as e:
                print(f"FlashInfer benchmark failed: {e}")

    elif engine == "tridao":
        # Run Tri Dao benchmark
        if _flash_attn_varlen_func is not None:
            try:
                result = bench_tridao(
                    batch_size,
                    q_len,
                    cache_len,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    causal,
                    dtype,
                    num_iters,
                    window_size=window_size,
                    no_kineto=no_kineto,
                )
            except Exception as e:
                print(f"Tri Dao benchmark failed: {e}")

    # Run MAX benchmark
    elif engine == "modular_max":
        try:
            result = bench_max(
                batch_size,
                q_len,
                cache_len,
                num_q_heads,
                num_kv_heads,
                head_dim,
                causal,
                dtype,
                num_iters,
                window_size=window_size,
                no_kineto=no_kineto,
            )
        except Exception as e:
            print(f"MAX benchmark failed: {e}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHA Prefill Benchmark")
    parser.add_argument(
        "--batch_size", "--batch-size", type=int, default=1, help="Batch size"
    )
    parser.add_argument(
        "--q_len",
        "--q-len",
        type=int,
        default=4096,
        help="Query length",
    )
    parser.add_argument(
        "--cache_len",
        "--cache-len",
        type=int,
        default=0,
        help="Cached prefix length (kv_len = cache_len + q_len)",
    )
    parser.add_argument(
        "--num_q_heads",
        "--num-q-heads",
        type=int,
        default=32,
        help="Number of query heads",
    )
    parser.add_argument(
        "--num_kv_heads",
        "--num-kv-heads",
        type=int,
        default=None,
        help="Number of KV heads (defaults to num_q_heads)",
    )

    parser.add_argument(
        "--head_dim", "--head-dim", type=int, default=128, help="Head dimension"
    )

    parser.add_argument(
        "--causal",
        type=int,
        default=1,
        help="Causal",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="modular_max",
        choices=["flashinfer", "tridao", "modular_max"],
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
    parser.add_argument(
        "--window_size",
        "--window-size",
        type=int,
        default=-1,
        help=(
            "Sliding window size: number of keys each query attends to,"
            " itself included. When > 0, a causal sliding window mask is used"
            " (FlashInfer is skipped; its SM100 cutlass backend lacks SWA"
            " support). <= 0 disables the window (full causal / null mask)."
        ),
    )
    parser.add_argument(
        "--no-kineto",
        action="store_true",
        help="Skip kineto timing (for ncu/nsys).",
    )
    args, _ = parser.parse_known_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if args.engine not in ["flashinfer", "tridao", "modular_max"]:
        raise ValueError(f"engine {args.engine} is not supported!")

    num_kv_heads = (
        args.num_kv_heads if args.num_kv_heads is not None else args.num_q_heads
    )

    result = bench_prefill(
        batch_size=args.batch_size,
        q_len=args.q_len,
        cache_len=args.cache_len,
        num_q_heads=args.num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=args.head_dim,
        causal=args.causal,
        dtype=dtype_map[args.dtype],
        engine=args.engine,
        num_iters=args.num_iters,
        window_size=args.window_size,
        no_kineto=args.no_kineto,
    )

    if args.num_iters > 1 and not args.no_kineto:
        met_sec, flops = result or [0, 0]
        flops_per_sec = ThroughputMeasure(Bench.flops, flops)
        name = (
            f"MHA_Prefill/batch_size={args.batch_size}/q_len={args.q_len}/"
            f"cache_len={args.cache_len}/"
            f"num_q_heads={args.num_q_heads}/num_kv_heads={num_kv_heads}/"
            f"head_dim={args.head_dim}/"
            f"causal={args.causal}/dtype={dtype_map[args.dtype]}/"
            f"window_size={args.window_size}/"
            f"engine={args.engine}/"
        )

        b = Bench(
            name,
            iters=1,
            met=met_sec,
            metric_list=[flops_per_sec],
        )

        b.dump_report(output_path=args.output)
