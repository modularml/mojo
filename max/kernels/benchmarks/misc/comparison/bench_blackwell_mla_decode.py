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

# Benchmark MLA (Multi-head Latent Attention) decode kernels.
# Compares FlashInfer's TRT-LLM MLA implementation against MAX's MLA implementation.
# Model config (num_q_heads, qk_nope_head_dim, etc.) is passed via CLI args from YAML.
# Run via kbench: kbench bench_mla_decode.yaml

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

# Import bench utilities from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# MAX imports
from bench import bench_kineto_with_cupti_warmup, setup_ninja_path
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
    flare_mla_decode_ragged_scaled,
)
from max.nn.kv_cache import (
    MLAKVCacheParams,
    PagedCacheValues,
)

LINE = "=" * 80

# Config presets: maps a config label to (engine, dtype, q_dtype).
# Using a single $config parameter instead of separate $engine/$dtype/$q_dtype
# avoids kbench's auto-pivot splitting on $dtype and keeps all 3 bars grouped.
CONFIG_MAP: dict[str, tuple[str, str, str]] = {
    # BF16 family
    "max_bf16": ("modular_max", "bfloat16", "bf16"),
    "max_qbf16_kvfp8": ("modular_max", "float8_e4m3fn", "bf16"),
    "fi_bf16": ("flashinfer", "bfloat16", "bf16"),
    # FP8 family
    "max_fp8_snap": ("modular_max", "float8_e4m3fn", "fp8_rope_aware"),
    "max_fp8": ("modular_max", "float8_e4m3fn", "fp8"),
    "fi_fp8": ("flashinfer", "float8_e4m3fn", "fp8"),
}

# Model presets: maps a model name to its architecture dimensions.
# When --model matches a key here, the corresponding dimensions are applied
# as defaults (overridden by explicit CLI args like --num_q_heads).
# This prevents the common mistake of passing --model kimi-k25 but forgetting
# --num_q_heads=64, which silently uses the wrong default (128 = DeepSeek).
_DEEPSEEK_DIMS: dict[str, int] = {
    "num_q_heads": 128,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
}
_KIMI_DIMS: dict[str, int] = {
    "num_q_heads": 64,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "kv_lora_rank": 512,
}
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "deepseek-v3": _DEEPSEEK_DIMS,
    "kimi-k2.5": _KIMI_DIMS,
}


def to_float8(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a tensor to FP8 format.

    Args:
        x: Input tensor to quantize
        dtype: Target FP8 dtype (default: float8_e4m3fn)

    Returns:
        Tuple of (quantized_tensor, scale) where scale is the dequantization scale
    """
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


# Try importing external libraries (installed via Bazel pycross_wheel_library)
_flashinfer: types.ModuleType | None
try:
    setup_ninja_path()  # Required for FlashInfer JIT compilation
    import flashinfer as _flashinfer
except ImportError as e:
    print(f"Error: flashinfer not available: {e}")
    _flashinfer = None


@dataclass
class Config:
    num_q_heads: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    kv_lora_rank: int


def calculate_mla_memory_bytes(
    batch_size: int,
    q_len_per_request: int,
    cache_len: int,
    dtype: torch.dtype,
    model_config: Config,
    q_dtype: torch.dtype | None = None,
    per_token_scale_rope_aware: bool = False,
) -> int:
    """Calculate memory throughput for MLA operations.

    Args:
        batch_size: Number of sequences
        q_len_per_request: Query length per request
        cache_len: KV cache length per sequence
        dtype: Data type of KV cache tensors
        model_config: Model configuration
        q_dtype: Data type of Q tensor (defaults to dtype if not specified)
        per_token_scale_rope_aware: When True, accounts for the interleaved
            FP8+BF16 layout (640 bytes/row) instead of uniform dtype,
            plus per-token scale memory (1 float32 per KV token for sigma_KV,
            1 float32 per Q token for sigma_Q).

    Returns:
        Total memory bytes read and written
    """
    if q_dtype is None:
        q_dtype = dtype

    num_q_heads = model_config.num_q_heads
    kv_lora_rank = model_config.kv_lora_rank
    qk_rope_head_dim = model_config.qk_rope_head_dim

    # MLA reads compressed KV cache and query, writes output
    # Read: query (in q_dtype) + kv_cache (in kv dtype)
    # Write: output (always bf16 for FP8 inputs, otherwise same as input dtype)
    def _bytes_per_element(dt: torch.dtype) -> int:
        if dt in [torch.float8_e4m3fn, torch.float8_e5m2]:
            return 1
        elif dt in [torch.bfloat16, torch.float16]:
            return 2
        else:
            return 4

    kv_bytes_per_element = _bytes_per_element(dtype)
    q_bytes_per_element = _bytes_per_element(q_dtype)

    # Output is always bf16 for FP8 inputs, otherwise same as input dtype
    if dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bfloat16,
        torch.float16,
    ]:
        output_bytes_per_element = 2
    else:
        output_bytes_per_element = 4

    if per_token_scale_rope_aware:
        # Per-token-scale rope-aware: each row = FP8 content (kv_lora_rank bytes)
        # + BF16 rope (qk_rope_head_dim * 2 bytes) = 640 bytes/row.
        bytes_per_row = kv_lora_rank + qk_rope_head_dim * 2  # 512 + 128 = 640

        # Query: [batch_size, q_len, num_q_heads, 640] in FP8 (1 byte/elem)
        query_bytes = (
            batch_size * q_len_per_request * num_q_heads * bytes_per_row
        )

        # KV cache: [batch_size, cache_len, 640] in FP8 (1 byte/elem)
        kv_bytes = batch_size * cache_len * bytes_per_row

        # Per-token scales: sigma_KV (1 float32 per KV token) +
        # sigma_Q (1 float32 per Q token, negligible for decode).
        # In MLA absorbed mode, K and V share one scale per token.
        kv_scale_bytes = batch_size * cache_len * 4  # float32
        q_scale_bytes = batch_size * q_len_per_request * 4  # float32
        kv_bytes += kv_scale_bytes + q_scale_bytes

    else:
        # Standard: uniform dtype for all dimensions
        # Input: query [batch_size, q_len, num_q_heads, kv_lora_rank + qk_rope_head_dim]
        query_bytes = (
            batch_size
            * q_len_per_request
            * num_q_heads
            * (kv_lora_rank + qk_rope_head_dim)
            * q_bytes_per_element
        )

        # KV cache: [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]
        kv_bytes = (
            batch_size
            * cache_len
            * (kv_lora_rank + qk_rope_head_dim)
            * kv_bytes_per_element
        )

    # Output: [batch_size, q_len_per_request, num_q_heads, kv_lora_rank]
    output_bytes = (
        batch_size
        * q_len_per_request
        * num_q_heads
        * kv_lora_rank
        * output_bytes_per_element
    )

    total_bytes = query_bytes + kv_bytes + output_bytes
    return total_bytes


def bench_flashinfer_trtllm(
    batch_size: int,
    cache_len: int,
    page_size: int,
    dtype: torch.dtype,
    model_config: Config,
    q_len_per_request: int = 1,
    enable_pdl: bool | None = None,
    no_kineto: bool = False,
    num_iters: int = 100,
) -> tuple[float, int] | None:
    """Benchmark FlashInfer MLA decode with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence (max sequence length)
        page_size: Page/block size for paged KV cache
        dtype: torch dtype for inputs (supports bfloat16, float16, float8_e4m3fn)
        q_len_per_request: Query length per request (1 for decode, >1 for chunked prefill)
        enable_pdl: Enable PDL (Persistent Dynamic Load) optimization
        num_iters: Number of benchmark iterations for noise reduction.
    """
    if _flashinfer is None:
        print("flashinfer not available, skipping bench_flashinfer")
        return None

    device = "cuda"
    is_fp8 = dtype == torch.float8_e4m3fn

    # DeepSeek MLA configuration
    num_q_heads = model_config.num_q_heads
    qk_nope_head_dim = model_config.qk_nope_head_dim
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_lora_rank = model_config.kv_lora_rank

    # For FP8: generate in bf16 first, then quantize for proper scaling
    gen_dtype = torch.bfloat16 if is_fp8 else dtype

    # Query tensor: [batch_size, q_len_per_request, num_q_heads, kv_lora_rank + qk_rope_head_dim]
    query_raw = torch.randn(
        batch_size,
        q_len_per_request,
        num_q_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=gen_dtype,
        device=device,
    )

    # Sequence lengths - all sequences have cache_len for simplicity
    seq_lens_tensor = torch.full(
        (batch_size,), cache_len, dtype=torch.int, device=device
    )

    # Calculate number of blocks per sequence
    max_num_blocks_per_seq = (cache_len + page_size - 1) // page_size

    # Generate block tables using arange (simple sequential block IDs)
    block_tables = torch.arange(
        batch_size * max_num_blocks_per_seq, dtype=torch.int, device=device
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Calculate total number of blocks needed for KV cache allocation
    num_blocks = batch_size * max_num_blocks_per_seq

    # Create MLA KV cache: [num_blocks, page_size, kv_lora_rank + qk_rope_head_dim]
    # This is the compressed format for MLA - stores ckv (compressed kv) and kpe (key positional encoding)
    kv_cache_raw = torch.randn(
        num_blocks,
        page_size,
        kv_lora_rank + qk_rope_head_dim,
        dtype=gen_dtype,
        device=device,
    )

    # For FP8: quantize and compute proper scales
    # bmm1_scale = q_scale * k_scale * sm_scale (where sm_scale = 1/sqrt(head_dim))
    # bmm2_scale = v_scale * o_scale (typically 1.0 for bf16 output)
    if is_fp8:
        query, q_scale = to_float8(query_raw)
        kv_cache, kv_scale = to_float8(kv_cache_raw)
        q_scale_val = q_scale.item()
        kv_scale_val = kv_scale.item()
        sm_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
        bmm1_scale = q_scale_val * kv_scale_val * sm_scale
        bmm2_scale = (
            kv_scale_val  # v_scale * o_scale (o_scale=1 for bf16 output)
        )
    else:
        query = query_raw
        kv_cache = kv_cache_raw
        sm_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
        bmm1_scale = sm_scale
        bmm2_scale = 1.0

    # Workspace buffer (must be zero-initialized for trtllm-gen backend)
    workspace_buffer = torch.zeros(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )

    def run_kernel() -> torch.Tensor:
        return _flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(
                1
            ),  # Add layer dimension: [num_blocks, 1, page_size, ...]
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens_tensor,
            max_seq_len=cache_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            enable_pdl=enable_pdl,
            backend="trtllm-gen",
        )

    # Warmup: run enough iterations with L2 flushes to bring the GPU into
    # a stable power/clock state (see bench_max docstring for rationale).
    flush_l2_size = int(1e9 // 4)
    for _ in range(20):
        torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
        run_kernel()

    # Calculate memory throughput
    total_bytes = calculate_mla_memory_bytes(
        batch_size,
        q_len_per_request,
        cache_len,
        dtype,
        model_config=model_config,
    )

    # If running under external profilers (ncu/nsys), skip kineto to avoid assertion failures.
    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return 1.0, total_bytes

    # Benchmark with CUPTI warmup for CUTLASS kernels.
    # FlashInfer split-K launches TWO kernels: fmhaSm100fKernel_* (decode)
    # and fmhaReductionKernel (combine). We must sum both to match MAX's
    # decode+combine timing. The prefix "fmha" matches both kernel names
    # without matching unrelated kernels.
    try:
        time_s = bench_kineto_with_cupti_warmup(
            run_kernel,
            kernel_names="fmha",  # Matches fmhaSm100fKernel_* + fmhaReductionKernel
            num_tests=num_iters,
            suppress_kineto_output=True,
            flush_l2=True,
            with_multiple_kernels=True,
        )
        assert isinstance(time_s, float)  # Single kernel_name returns float
    except RuntimeError as e:
        # If kineto fails (e.g., running under ncu/nsys), return dummy time
        if "No kernel times found" in str(e):
            print(
                f"Warning: kineto profiling failed (likely running under ncu/nsys). "
                f"Use --no-kineto flag to skip kineto. Error: {e}"
            )
            run_kernel()
            torch.cuda.synchronize()
            return 1.0, total_bytes
        raise

    return time_s, total_bytes


def bench_max(
    batch_size: int,
    cache_len: int,
    page_size: int,
    dtype: torch.dtype,
    model_config: Config,
    q_len_per_request: int = 1,
    no_kineto: bool = False,
    q_dtype_override: torch.dtype | None = None,
    num_iters: int = 100,
    per_token_scale_rope_aware: bool = False,
) -> tuple[float, int]:
    """Benchmark MAX MLA decode with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence
        page_size: Page size for paged KV cache
        dtype: torch dtype for inputs (bf16 or fp8 for KV cache)
        q_len_per_request: Query length per request (1 for decode)
        no_kineto: Skip kineto timing (for ncu/nsys)
        q_dtype_override: Override Q tensor dtype. If None, Q is BF16 (legacy).
            Set to torch.float8_e4m3fn for native QKV FP8 mode.
        num_iters: Number of benchmark iterations for noise reduction.
        per_token_scale_rope_aware: When True, uses the FP8 per-token-scale rope-aware
            path where content (kv_lora_rank=512 dims) is FP8 and rope
            (qk_rope_head_dim=64 dims) stays BF16. Q and KV cache rows are
            640 bytes each (512 FP8 + 128 BF16). The kernel uses real per-token
            FP8 scaling: sigma_KV (random in [0.8, 1.2]) per KV token and
            sigma_Q (random in [0.8, 1.2]) per Q token, passed as explicit
            scale tensors through the scaled graph API.
    """
    is_fp8_kv = dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    kv_dtype = dtype  # KV cache dtype (bf16 or fp8)
    # For native FP8: Q can be FP8 too. Otherwise default to BF16.
    q_dtype = (
        q_dtype_override if q_dtype_override is not None else torch.bfloat16
    )
    is_fp8_q = q_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)

    max_kv_dtype = torch_dtype_to_max(kv_dtype)
    max_q_dtype = torch_dtype_to_max(q_dtype)

    # Create inference session
    session = InferenceSession(devices=[Accelerator()])

    # DeepSeek MLA configuration
    num_q_heads = model_config.num_q_heads
    num_kv_heads = 1  # MLA uses 1 KV head (MQA with compression)
    qk_nope_head_dim = model_config.qk_nope_head_dim
    qk_rope_head_dim = model_config.qk_rope_head_dim
    kv_lora_rank = model_config.kv_lora_rank
    # For per_token_scale_rope_aware: Q and KV rows are interleaved FP8+BF16.
    # FP8 content (kv_lora_rank=512 bytes) + BF16 rope (qk_rope_head_dim*2=128 bytes)
    # = 640 bytes/row. Since Q dtype is FP8 (1 byte/element), the last dim = 640.
    # For standard: qk_head_dim = kv_lora_rank + qk_rope_head_dim = 576.
    if per_token_scale_rope_aware:
        qk_head_dim = kv_lora_rank + qk_rope_head_dim * 2  # 512 + 128 = 640
    else:
        qk_head_dim = kv_lora_rank + qk_rope_head_dim  # 512 + 64 = 576

    # Setup KV cache configuration for MLA
    # MLA stores compressed KV (kv_lora_rank) + rope embeddings (qk_rope_head_dim)
    # For per_token_scale_rope_aware: KV cache head_dim is 640 (FP8 content + BF16 rope)
    kv_cache_head_dim = (
        qk_head_dim
        if per_token_scale_rope_aware
        else (kv_lora_rank + qk_rope_head_dim)
    )
    kv_params = MLAKVCacheParams(
        dtype=max_kv_dtype,
        head_dim=kv_cache_head_dim,
        num_layers=1,  # Benchmarking a single layer
        page_size=page_size,
        devices=[DeviceRef.GPU()],
        num_q_heads=num_q_heads,
    )

    num_blocks_per_seq = (
        cache_len + q_len_per_request + page_size - 1
    ) // page_size

    # For MLA: [num_pages, 1 (K and V compressed together), 1 layer, page_size, 1 kv_head, compressed_dim]
    # MLA stores compressed KV cache, not separate K and V
    # For FP8: generate in bf16 first then cast (torch.randn doesn't support FP8)
    # For per_token_scale_rope_aware: KV cache uses interleaved FP8+BF16 layout (640 bytes/row)
    if per_token_scale_rope_aware:
        # SnapMLA Scale Domain Alignment (Eq. 6):
        # K_rope_aligned[t] = K_rope[t] / sigma_KV[t]
        # Generate content and rope separately, apply scale alignment, then pack.
        num_blocks = batch_size * num_blocks_per_seq

        # Generate per-token KV scales first (needed for alignment).
        # Shape: [num_blocks, 1, 1, page_size, 1, 1] to match 6D paged layout.
        kv_scales_torch = 0.8 + 0.4 * torch.rand(
            num_blocks,
            1,
            1,
            page_size,
            num_kv_heads,
            1,
            dtype=torch.float32,
            device="cuda",
        )

        # Content portion: [num_blocks, 1, 1, page_size, 1, kv_lora_rank] in bf16 -> fp8
        kv_content_bf16 = torch.randn(
            num_blocks,
            1,
            1,
            page_size,
            num_kv_heads,
            kv_lora_rank,
            dtype=torch.bfloat16,
            device="cuda",
        )
        kv_content_fp8 = kv_content_bf16.to(
            torch.float8_e4m3fn
        )  # [.., 512] fp8 = 512 bytes

        # Rope portion: [num_blocks, 1, 1, page_size, 1, qk_rope_head_dim] in bf16
        kv_rope_bf16 = torch.randn(
            num_blocks,
            1,
            1,
            page_size,
            num_kv_heads,
            qk_rope_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )

        # Scale Domain Alignment: divide rope by per-token sigma_KV.
        # kv_scales_torch shape [num_blocks,1,1,page_size,1,1] broadcasts over rope dim.
        kv_rope_aligned = kv_rope_bf16 / kv_scales_torch.to(torch.bfloat16)

        # Pack into interleaved byte layout: FP8 content (512 bytes) + BF16 rope (128 bytes) = 640 bytes.
        # View content as uint8 [.., 512] and rope as uint8 [.., 128], concatenate on last dim.
        content_bytes = kv_content_fp8.view(torch.uint8)  # [.., 512]
        rope_bytes = kv_rope_aligned.view(torch.uint8)  # [.., 128]
        packed_bytes = torch.cat(
            [content_bytes, rope_bytes], dim=-1
        )  # [.., 640]
        # Reinterpret as fp8 (the kernel knows the byte layout).
        paged_blocks_torch = packed_bytes.view(torch.float8_e4m3fn)
    else:
        paged_blocks_torch = torch.randn(
            batch_size * num_blocks_per_seq,
            1,  # MLA uses compressed KV format, not separate K and V
            1,
            page_size,
            num_kv_heads,
            kv_cache_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        if is_fp8_kv:
            paged_blocks_torch = paged_blocks_torch.to(kv_dtype)

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

    # For decode: max_prompt_length=q_len_per_request, max_cache_length=cache_len
    max_prompt_length_torch = torch.tensor(
        [q_len_per_request], dtype=torch.uint32, device="cpu"
    )
    max_cache_length_torch = torch.tensor(
        [cache_len], dtype=torch.uint32, device="cpu"
    )

    # Convert torch tensors to MAX types
    # For FP8: DLPack doesn't support float8 types, so we transfer as uint8
    # and reinterpret the dtype using Buffer.view()
    if is_fp8_kv:
        paged_blocks_max = Buffer.from_dlpack(
            paged_blocks_torch.view(torch.uint8)
        ).view(max_kv_dtype)
    else:
        paged_blocks_max = Buffer.from_dlpack(paged_blocks_torch)
    lut_max = Buffer.from_dlpack(lut_torch)
    cache_lengths_max = Buffer.from_dlpack(cache_lengths_torch)
    max_prompt_length_max = Buffer.from_dlpack(max_prompt_length_torch)
    max_cache_length_max = Buffer.from_dlpack(max_cache_length_torch)

    # Define input types
    # Query for MLA decode: [total_tokens, num_q_heads, qk_head_dim]
    # Q is BF16 by default, or FP8 in native QKV FP8 mode
    q_type = TensorType(
        max_q_dtype,
        shape=["total_tokens", num_q_heads, qk_head_dim],
        device=DeviceRef.GPU(),
    )

    input_row_offsets_type = TensorType(
        DType.uint32,
        shape=["batch_size_plus_1"],
        device=DeviceRef.GPU(),
    )

    blocks_type = BufferType(
        max_kv_dtype,
        shape=[
            "total_num_pages",
            1,
            1,
            page_size,
            num_kv_heads,
            kv_cache_head_dim,
        ],
        device=DeviceRef.GPU(),
    )

    cache_lengths_type = TensorType(
        DType.uint32,
        shape=["batch_size"],
        device=DeviceRef.GPU(),
    )

    lookup_table_type = TensorType(
        DType.uint32,
        shape=["batch_size", "max_num_pages"],
        device=DeviceRef.GPU(),
    )

    max_prompt_length_type = TensorType(
        DType.uint32,
        shape=[1],
        device=DeviceRef.CPU(),
    )
    max_cache_length_type = TensorType(
        DType.uint32,
        shape=[1],
        device=DeviceRef.CPU(),
    )

    # Additional types for per-token scale path
    # kv_scales: [num_blocks, 1, 1, page_size, 1, 1] float32 (one scale per KV token)
    # q_scales: [total_tokens] float32 (one scale per Q token)
    kv_scales_type = BufferType(
        DType.float32,
        shape=["total_num_pages", 1, 1, page_size, num_kv_heads, 1],
        device=DeviceRef.GPU(),
    )
    q_scales_type = TensorType(
        DType.float32,
        shape=["total_tokens"],
        device=DeviceRef.GPU(),
    )

    scalar_args_type = TensorType(
        DType.int64,
        shape=[3],
        device=DeviceRef.GPU(),
    )

    # Build graph with MLA decode
    if per_token_scale_rope_aware:
        # Scaled path: pass explicit kv_scales and q_scales tensors
        with Graph(
            "mla_decode_max_scaled",
            input_types=[
                q_type,
                input_row_offsets_type,
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                max_prompt_length_type,
                max_cache_length_type,
                kv_scales_type,
                q_scales_type,
                scalar_args_type,
            ],
        ) as graph:
            (
                q,
                input_row_offsets,
                blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                kv_scales_graph,
                q_scales_graph,
                scalar_args,
            ) = graph.inputs

            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
            )

            result = flare_mla_decode_ragged_scaled(
                kv_params,
                q.tensor,
                input_row_offsets.tensor,
                kv_collection,
                kv_scales=kv_scales_graph.buffer,
                q_scales=q_scales_graph.tensor,
                layer_idx=layer_idx,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim),
                scalar_args=scalar_args.tensor,
                qk_rope_dim=qk_rope_head_dim,
                per_token_scale_rope_aware=True,
                quantization_granularity=kv_cache_head_dim,
            )

            graph.output(result)
    else:
        with Graph(
            "mla_decode_max",
            input_types=[
                q_type,
                input_row_offsets_type,
                blocks_type,
                cache_lengths_type,
                lookup_table_type,
                max_prompt_length_type,
                max_cache_length_type,
                scalar_args_type,
            ],
        ) as graph:
            (
                q,
                input_row_offsets,
                blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                scalar_args,
            ) = graph.inputs

            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
            )

            result = flare_mla_decode_ragged(
                kv_params,
                q.tensor,
                input_row_offsets.tensor,
                kv_collection,
                layer_idx,
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim),
                scalar_args=scalar_args.tensor,
                qk_rope_dim=qk_rope_head_dim,
            )

            graph.output(result)

    # Compile model
    model = session.load(graph)

    # Prepare inputs
    # Query: [batch_size * q_len_per_request, num_q_heads, qk_head_dim]
    # Q is BF16 by default, or FP8 in native QKV FP8 mode
    total_tokens = batch_size * q_len_per_request

    if per_token_scale_rope_aware:
        # SnapMLA Scale Domain Alignment (Eq. 6):
        # Q_rope_aligned[q] = Q_rope[q] / sigma_Q[q]
        # Generate Q content and rope separately, apply scale alignment, then pack.

        # Generate per-token Q scales first (needed for alignment).
        q_scales_torch = 0.8 + 0.4 * torch.rand(
            total_tokens,
            dtype=torch.float32,
            device="cuda",
        )

        # Content portion: [total_tokens, num_q_heads, kv_lora_rank] in bf16 -> fp8
        q_content_bf16 = torch.randn(
            total_tokens,
            num_q_heads,
            kv_lora_rank,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q_content_fp8 = q_content_bf16.to(torch.float8_e4m3fn)  # [.., 512] fp8

        # Rope portion: [total_tokens, num_q_heads, qk_rope_head_dim] in bf16
        q_rope_bf16 = torch.randn(
            total_tokens,
            num_q_heads,
            qk_rope_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )

        # Scale Domain Alignment: divide rope by per-token sigma_Q.
        # q_scales shape [total_tokens] -> [total_tokens, 1, 1] for broadcasting.
        q_rope_aligned = q_rope_bf16 / q_scales_torch.to(
            torch.bfloat16
        ).unsqueeze(-1).unsqueeze(-1)

        # Pack into interleaved byte layout: FP8 content (512 bytes) + BF16 rope (128 bytes) = 640 bytes.
        q_content_bytes = q_content_fp8.view(torch.uint8)  # [.., 512]
        q_rope_bytes = q_rope_aligned.view(torch.uint8)  # [.., 128]
        q_packed_bytes = torch.cat(
            [q_content_bytes, q_rope_bytes], dim=-1
        )  # [.., 640]
        q_input_torch = q_packed_bytes.view(torch.float8_e4m3fn)

        # Use uint8 view workaround for DLPack (FP8 not supported by DLPack).
        q_input = Buffer.from_dlpack(q_input_torch.view(torch.uint8)).view(
            max_q_dtype
        )

        # kv_scales_torch was already generated during KV cache construction above.
        kv_scales_max = Buffer.from_dlpack(kv_scales_torch)
        q_scales_max = q_scales_torch
    else:
        q_input_torch = torch.randn(
            total_tokens,
            num_q_heads,
            qk_head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        if is_fp8_q:
            q_input_torch = q_input_torch.to(q_dtype)

        # For FP8 Q: DLPack doesn't support float8 types, use uint8 view workaround
        if is_fp8_q:
            q_input = Buffer.from_dlpack(q_input_torch.view(torch.uint8)).view(
                max_q_dtype
            )
        else:
            q_input = q_input_torch

    # Input row offsets for ragged tensor
    input_row_offsets = torch.arange(
        0, total_tokens + 1, q_len_per_request, dtype=torch.int32, device="cuda"
    ).to(torch.uint32)

    # Scalar dispatch args for the MLA decode kernel (shape [3], int64, GPU).
    # Use the canonical Mojo dispatch heuristic via mla_dispatch_args_scalar
    # instead of duplicating the logic in Python.
    # The kernel reads these from device memory, so we must place them on GPU
    # (matching the production path in MLAKVCacheParams.resolve_attn_key).
    device = Accelerator()
    scalar_args_gpu = Buffer.from_numpy(
        np.array(
            mla_dispatch_args_scalar(
                batch_size,
                cache_len,
                q_len_per_request,
                num_q_heads,
                is_fp8_kv,
                device,
            ),
            dtype=np.int64,
        )
    ).to(device)

    def run_kernel() -> Any:
        if per_token_scale_rope_aware:
            output = model.execute(
                q_input if is_fp8_q else q_input_torch.detach(),
                input_row_offsets.detach(),
                paged_blocks_max,
                cache_lengths_max,
                lut_max,
                max_prompt_length_max,
                max_cache_length_max,
                kv_scales_max,
                q_scales_max,
                scalar_args_gpu,
            )[0]
        else:
            output = model.execute(
                q_input if is_fp8_q else q_input_torch.detach(),
                input_row_offsets.detach(),
                paged_blocks_max,
                cache_lengths_max,
                lut_max,
                max_prompt_length_max,
                max_cache_length_max,
                scalar_args_gpu,
            )[0]
        return output

    # Warmup: run enough iterations with L2 flushes to bring the GPU into
    # a stable power/clock state.  Without the flushes the warmup runs at
    # unrealistically high occupancy; the profiling pass then interleaves
    # 1 GB L2 flushes which can cause a GPU power-state dip that inflates
    # the first profiling pass by 2-7x.
    flush_l2_size = int(1e9 // 4)
    for _ in range(20):
        torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
        run_kernel()

    # Calculate memory throughput
    total_bytes = calculate_mla_memory_bytes(
        batch_size,
        q_len_per_request,
        cache_len,
        dtype,
        model_config=model_config,
        q_dtype=q_dtype,
        per_token_scale_rope_aware=per_token_scale_rope_aware,
    )

    # If running under external profilers (ncu/nsys), skip kineto to avoid assertion failures.
    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return 1.0, total_bytes

    # Benchmark with CUPTI warmup.
    # Split-K launches two kernels (decode + combine), so we use
    # with_multiple_kernels=True to sum their times.
    # Kernel names use the @__name decorator pattern, e.g.
    #   sm100_mla_decode_kv_bf16_..._<hash>   (main decode)
    #   sm100_mla_decode_combine_..._<hash>    (reduction)
    # The substring "sm100_mla_decode" matches all MLA decode variants
    # (kv_bf16, kv_fp8, qkv_fp8, qkv_fp8_per_token_scale_rope_aware,
    # and the combine kernel).
    try:
        time_s = bench_kineto_with_cupti_warmup(
            run_kernel,
            kernel_names="sm100_mla_decode",
            num_tests=num_iters,
            suppress_kineto_output=True,
            flush_l2=True,
            with_multiple_kernels=True,
        )
        assert isinstance(time_s, float)  # Single kernel_name returns float
    except RuntimeError as e:
        # If kineto fails (e.g., running under ncu/nsys), return dummy time
        if "No kernel times found" in str(e):
            print(
                f"Warning: kineto profiling failed (likely running under ncu/nsys). "
                f"Use --no-kineto flag to skip kineto. Error: {e}"
            )
            run_kernel()
            torch.cuda.synchronize()
            return 1.0, total_bytes
        raise

    return time_s, total_bytes


def bench_mla_decode(
    batch_size: int,
    cache_len: int,
    dtype: torch.dtype,
    model_config: Config,
    engine: str,  # "modular_max" or "flashinfer"
    q_len_per_request: int = 1,
    backend: str = "trtllm-gen",
    enable_pdl: bool | None = None,
    no_kineto: bool = False,
    q_dtype: str = "bf16",
    num_iters: int = 100,
) -> tuple[float, int] | None:
    """Run all MLA decode benchmarks and return results.

    Args:
        batch_size: Batch size (number of sequences)
        cache_len: KV cache length per sequence (max sequence length)
        dtype: torch dtype for inputs (e.g., torch.bfloat16, torch.float8_e4m3fn)
        model_config: Model configuration
        engine: Engine to benchmark ("modular_max" or "flashinfer")
        q_len_per_request: Query length per request (1 for decode, >1 for chunked prefill)
        backend: Backend for FlashInfer ("trtllm-gen" or "xqa")
        enable_pdl: Enable PDL optimization for FlashInfer
        no_kineto: Skip kineto timing (for ncu/nsys)
        q_dtype: Query dtype ("bf16", "fp8", or "fp8_rope_aware"). When "fp8",
            Q is float8_e4m3fn for native QKV FP8 mode. When "fp8_rope_aware",
            uses FP8 per-token-scale rope-aware path (content FP8 + rope BF16).
            Only affects MAX engine; FlashInfer always matches Q dtype to dtype.
        num_iters: Number of benchmark iterations for noise reduction.

    Returns:
        Tuple of (time_seconds, total_bytes) or None on failure.
    """
    # Use appropriate page sizes for each backend
    flashinfer_page_size = 64  # FlashInfer TRT-LLM tested with 32 and 64
    max_page_size = 128  # MAX only supports 128

    # Resolve Q dtype override and per_token_scale_rope_aware for MAX engine
    q_dtype_override: torch.dtype | None = None
    rope_aware: bool = False
    if q_dtype == "fp8":
        q_dtype_override = torch.float8_e4m3fn
    elif q_dtype == "fp8_rope_aware":
        q_dtype_override = torch.float8_e4m3fn
        rope_aware = True

    result: tuple[float, int] | None = None
    if engine == "flashinfer":
        # Run FlashInfer benchmark with TensorRT-LLM MLA backend
        # FlashInfer always uses the same dtype for Q and KV
        if _flashinfer is not None:
            try:
                result = bench_flashinfer_trtllm(
                    batch_size=batch_size,
                    cache_len=cache_len,
                    page_size=flashinfer_page_size,
                    dtype=dtype,
                    model_config=model_config,
                    q_len_per_request=q_len_per_request,
                    enable_pdl=enable_pdl,
                    no_kineto=no_kineto,
                    num_iters=num_iters,
                )
            except Exception as e:
                print(f"FlashInfer benchmark failed: {e}")
                import traceback

                traceback.print_exc()

    # Run MAX benchmark
    if engine == "modular_max":
        try:
            result = bench_max(
                batch_size=batch_size,
                cache_len=cache_len,
                page_size=max_page_size,
                dtype=dtype,
                model_config=model_config,
                q_len_per_request=q_len_per_request,
                no_kineto=no_kineto,
                q_dtype_override=q_dtype_override,
                num_iters=num_iters,
                per_token_scale_rope_aware=rope_aware,
            )
        except Exception as e:
            print(f"MAX benchmark failed: {e}")
            import traceback

            traceback.print_exc()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLA Decode Benchmark")
    parser.add_argument(
        "--batch_size", "--batch-size", type=int, default=128, help="Batch size"
    )
    parser.add_argument(
        "--cache_len",
        "--cache-len",
        type=int,
        default=1024,
        help="KV cache length",
    )
    parser.add_argument(
        "--q_len_per_request",
        "--q-len-per-request",
        type=int,
        default=1,
        help="Q Length per Request",
    )
    parser.add_argument(
        "--num_q_heads",
        "--num-q-heads",
        type=int,
        default=None,
        help="Number of query heads (default: from --model preset)",
    )
    parser.add_argument(
        "--qk_nope_head_dim",
        "--qk-nope-head-dim",
        type=int,
        default=None,
        help="qk nope head dim (default: from --model preset)",
    )
    parser.add_argument(
        "--qk_rope_head_dim",
        "--qk-rope-head-dim",
        type=int,
        default=None,
        help="qk rope head dim (default: from --model preset)",
    )
    parser.add_argument(
        "--kv_lora_rank",
        "--kv-lora-rank",
        type=int,
        default=None,
        help="kv lora rank (default: from --model preset)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "float8_e4m3fn"],
        help="Data type (float8_e4m3fn for FP8 quantized MLA)",
    )
    parser.add_argument(
        "--q_dtype",
        "--q-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8", "fp8_rope_aware"],
        help=(
            "Query tensor dtype: 'bf16' (default), 'fp8' for native QKV FP8, "
            "or 'fp8_rope_aware' for FP8 per-token-scale rope-aware (content FP8 + "
            "rope BF16). Only affects MAX engine."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(CONFIG_MAP.keys()),
        help=(
            "Config preset encoding engine+dtype+q_dtype. "
            "Overrides --engine/--dtype/--q_dtype when set. "
            "Valid values: " + ", ".join(CONFIG_MAP.keys())
        ),
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="modular_max",
        choices=["modular_max", "flashinfer"],
        help="Engine (ignored when --config is set)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v3",
        help=(
            "Model name. Sets default architecture dimensions from "
            "MODEL_PRESETS when --num_q_heads etc. are not explicitly "
            "provided. Known models: " + ", ".join(MODEL_PRESETS.keys())
        ),
    )
    parser.add_argument(
        "--no-kineto",
        action="store_true",
        help="Skip kineto timing (for ncu/nsys).",
    )
    parser.add_argument(
        "--num_iters",
        "--num-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations for noise reduction",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output path",
    )
    args, _ = parser.parse_known_args()

    # Resolve $config preset into engine/dtype/q_dtype (overrides individual args).
    if args.config is not None:
        if args.config not in CONFIG_MAP:
            raise ValueError(
                f"Unknown config '{args.config}'. "
                f"Valid: {', '.join(CONFIG_MAP.keys())}"
            )
        _engine, _dtype_str, _q_dtype = CONFIG_MAP[args.config]
        args.engine = _engine
        args.dtype = _dtype_str
        args.q_dtype = _q_dtype

    if args.engine not in ["flashinfer", "modular_max"]:
        raise ValueError(f"engine {args.engine} is not supported!")

    # Resolve model preset dimensions. Explicit CLI args take priority;
    # anything left as None falls back to the model preset (or hardcoded
    # DeepSeek defaults if the model name is unknown).
    _preset = MODEL_PRESETS.get(args.model, MODEL_PRESETS["deepseek-v3"])
    if args.num_q_heads is None:
        args.num_q_heads = _preset["num_q_heads"]
    if args.qk_nope_head_dim is None:
        args.qk_nope_head_dim = _preset["qk_nope_head_dim"]
    if args.qk_rope_head_dim is None:
        args.qk_rope_head_dim = _preset["qk_rope_head_dim"]
    if args.kv_lora_rank is None:
        args.kv_lora_rank = _preset["kv_lora_rank"]

    cfg = Config(
        num_q_heads=args.num_q_heads,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        kv_lora_rank=args.kv_lora_rank,
    )

    # Print resolved config to stderr for diagnostics (visible when debugging,
    # captured by kbench but not mixed into the CSV output on stdout).
    print(
        f"[bench_mla_decode] model={args.model} config={args.config} "
        f"engine={args.engine} dtype={args.dtype} q_dtype={args.q_dtype} "
        f"batch_size={args.batch_size} cache_len={args.cache_len} "
        f"num_q_heads={args.num_q_heads} qk_nope_head_dim={args.qk_nope_head_dim} "
        f"qk_rope_head_dim={args.qk_rope_head_dim} kv_lora_rank={args.kv_lora_rank}",
        file=sys.stderr,
    )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float8_e4m3fn": torch.float8_e4m3fn,
    }

    result = bench_mla_decode(
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        dtype=dtype_map[args.dtype],
        engine=args.engine,
        model_config=cfg,
        q_len_per_request=args.q_len_per_request,
        backend="trtllm-gen",
        enable_pdl=True,
        no_kineto=args.no_kineto,
        q_dtype=args.q_dtype,
        num_iters=args.num_iters,
    )

    if result is None:
        print(
            "Benchmark returned no result (kernel error or unsupported config)"
        )
        sys.exit(1)

    met_sec, bytes = result
    if met_sec <= 0:
        print(f"Benchmark returned invalid time: {met_sec}")
        sys.exit(1)

    bytes_per_sec = ThroughputMeasure(Bench.bytes, bytes)

    config_tag = (
        f"config={args.config}"
        if args.config is not None
        else f"dtype={args.dtype}/q_dtype={args.q_dtype}"
    )
    name = (
        f"MLA_Decode/model={args.model}/batch_size={args.batch_size}/"
        f"cache_len={args.cache_len}/"
        f"q_len_per_request={args.q_len_per_request}/num_q_heads={args.num_q_heads}/"
        f"qk_nope_head_dim={args.qk_nope_head_dim}/qk_rope_head_dim={args.qk_rope_head_dim}/"
        f"kv_lora_rank={args.kv_lora_rank}/engine={args.engine}/"
        f"{config_tag}/"
    )

    b = Bench(
        name,
        iters=1,
        met=met_sec,
        metric_list=[bytes_per_sec],
    )

    b.dump_report(output_path=args.output)
