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

# Setup: run setup_bench_env.py (MAX installs by default) then activate the
# produced venv. Example:
#   python $MODULAR_PATH/Kernels/benchmarks/comparison/setup_bench_env.py
#   source $MODULAR_PATH/.venv/bin/activate
# The only SoTA MHA decode kernel on blackwell is from TRTLLM, called via flashinfer.
# Run via kbench: kbench bench_decode.yaml

from __future__ import annotations

import math
import os
import sys
import types
from typing import Any

import torch

# Import bench utilities from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse

from bench import bench_kineto, setup_ninja_path
from bencher_utils import Bench, ThroughputMeasure

# MAX imports
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.torch import torch_dtype_to_max
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues

# Try importing external libraries (installed via Bazel pycross_wheel_library)
_flashinfer: types.ModuleType | None
try:
    setup_ninja_path()  # Required for FlashInfer JIT compilation
    import flashinfer as _flashinfer
except ImportError as e:
    print(f"Error: flashinfer not available: {e}")
    _flashinfer = None


def bench_flashinfer(
    batch_size: int,
    cache_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    use_tensor_cores: bool = True,
    backend: str = "fa",
) -> tuple[float, int] | None:
    """Benchmark FlashInfer decode with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        block_size: Page/block size for paged KV cache
        dtype: torch dtype for inputs
        use_tensor_cores: Whether to use tensor cores
        backend: Backend to use ("fa" for FlashAttention, "trtllm" for TensorRT-LLM, "trtllm-gen" for TensorRT-LLM generation)
    """
    if _flashinfer is None:
        print("flashinfer not available, skipping bench_flashinfer")
        return None

    # Determine layout based on backend
    kv_layout = "HND" if backend in ["trtllm", "trtllm-gen"] else "NHD"

    # Query tensor (one token per sequence for decode)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device="cuda")

    # Create block tables
    max_num_blocks_per_seq = (cache_len + block_size - 1) // block_size
    block_tables = torch.arange(
        batch_size * max_num_blocks_per_seq, dtype=torch.int, device="cuda"
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Build FlashInfer metadata
    kv_indptr = [0]
    kv_indices = []
    kv_last_page_lens = []

    for i in range(batch_size):
        num_blocks = (cache_len + block_size - 1) // block_size
        kv_indices.extend(block_tables[i, :num_blocks])
        kv_indptr.append(kv_indptr[-1] + num_blocks)

        kv_last_page_len = cache_len % block_size
        if kv_last_page_len == 0:
            kv_last_page_len = block_size
        kv_last_page_lens.append(kv_last_page_len)

    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32, device="cuda")
    kv_indices = torch.tensor(kv_indices, dtype=torch.int32, device="cuda")
    kv_last_page_lens = torch.tensor(
        kv_last_page_lens, dtype=torch.int32, device="cuda"
    )

    # Create paged KV cache with layout matching the backend
    if kv_layout == "HND":
        # TensorRT-LLM backend expects [num_blocks, 2, num_kv_heads, block_size, head_dim]
        key_value_cache = torch.randn(
            batch_size * max_num_blocks_per_seq,
            2,
            num_kv_heads,
            block_size,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
    else:
        # FlashAttention backend expects [num_blocks, 2, block_size, num_kv_heads, head_dim]
        key_value_cache = torch.randn(
            batch_size * max_num_blocks_per_seq,
            2,
            block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device="cuda"
    )

    # Create decode wrapper
    wrapper = _flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout,
        use_tensor_cores=use_tensor_cores,
        backend=backend,
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        "NONE",  # position encoding mode
        q_data_type=dtype,
    )

    def run_kernel() -> torch.Tensor:
        return wrapper.forward(q, key_value_cache)

    # Warmup
    for _ in range(10):
        wrapper.forward(q, key_value_cache)

    # Now benchmark with the correct kernel name
    # TensorRT-LLM backend uses BatchPrefillWithPagedKVCacheKernel (even for decode)
    # FlashAttention backend uses BatchDecodeWithPagedKVCacheKernel
    if backend in ["trtllm", "trtllm-gen"]:
        kernel_name = "BatchPrefillWithPagedKVCacheKernel"
    else:
        kernel_name = "BatchDecodeWithPagedKVCacheKernel"

    time_s = bench_kineto(
        run_kernel,
        kernel_names=kernel_name,
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    # Calculate memory throughput (GB/s)
    # Read: q (batch_size * num_heads * head_dim) + k,v (batch_size * cache_len * num_kv_heads * head_dim * 2)
    # Write: output (batch_size * num_heads * head_dim)
    bytes_per_element = (
        2 if dtype == torch.bfloat16 or dtype == torch.float16 else 4
    )
    total_bytes = (
        2
        * batch_size
        * (num_heads * head_dim + cache_len * num_kv_heads * head_dim)
        * bytes_per_element
    )

    return time_s, total_bytes


def bench_max(
    batch_size: int,
    cache_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    dtype: torch.dtype,
) -> tuple[float, int]:
    """Benchmark MAX flash_attention_ragged with paged KV cache.

    Args:
        batch_size: Number of sequences
        cache_len: KV cache length per sequence
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        page_size: Page size for paged KV cache
        dtype: torch dtype for inputs
    """
    # Convert torch dtype to MAX DType
    max_dtype = torch_dtype_to_max(dtype)

    # Create inference session
    session = InferenceSession(devices=[Accelerator()])

    # Setup KV cache configuration
    kv_params = MHAKVCacheParams(
        dtype=max_dtype,
        n_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
    )

    # Calculate required memory:
    # batch_size * num_blocks_per_seq * block_size * num_kv_heads * num_layers * head_dim * dtype_size
    num_blocks_per_seq = (cache_len + page_size - 1) // page_size
    bytes_per_element = (
        2 if max_dtype == DType.bfloat16 or max_dtype == DType.float16 else 4
    )
    # required_memory
    _ = (
        2
        * batch_size
        * num_blocks_per_seq
        * page_size
        * num_kv_heads
        * head_dim
        * bytes_per_element
    )

    # [num_pages, 2 for K and V, 1 layer, ...]
    paged_blocks_torch = torch.randn(
        batch_size * num_blocks_per_seq,
        2,
        1,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device="cuda",
    )
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
    # For decode: max_prompt_length=1 (one token), max_cache_length=cache_len
    max_prompt_length_torch = torch.tensor(
        [1], dtype=torch.uint32, device="cpu"
    )
    max_cache_length_torch = torch.tensor(
        [cache_len], dtype=torch.uint32, device="cpu"
    )

    # Convert torch tensors to MAX types (these will be the actual runtime inputs)
    paged_blocks_max = Buffer.from_dlpack(
        paged_blocks_torch
    )  # Buffer for kv_blocks
    lut_max = Buffer.from_dlpack(lut_torch)  # Tensor for lookup_table
    cache_lengths_max = Buffer.from_dlpack(
        cache_lengths_torch
    )  # Tensor for cache_lengths
    max_prompt_length_max = Buffer.from_dlpack(
        max_prompt_length_torch
    )  # Tensor for max_prompt_length
    max_cache_length_max = Buffer.from_dlpack(
        max_cache_length_torch
    )  # Tensor for max_cache_length
    attention_dispatch_metadata_max = Buffer.from_dlpack(
        torch.tensor([batch_size, 1, 0, cache_len], dtype=torch.int64)
    )

    # Define input types.
    # Avoid using KVCacheManager for its complexity.
    # For decode: query is [total_tokens, num_heads, head_dim] where total_tokens = batch_size
    q_type = TensorType(
        max_dtype,
        shape=["total_tokens", num_heads, head_dim],
        device=DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        shape=["batch_size_plus_1"],
        device=DeviceRef.GPU(),
    )

    blocks_type = BufferType(
        max_dtype,
        shape=["total_num_pages", 2, 1, page_size, num_kv_heads, head_dim],
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
    attention_dispatch_metadata_type = TensorType(
        DType.int64, shape=[4], device=DeviceRef.CPU()
    )

    # Build graph with paged KV cache inputs
    with Graph(
        "flash_attn_decode_max",
        input_types=[
            q_type,
            input_row_offsets_type,
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_prompt_length_type,
            max_cache_length_type,
            attention_dispatch_metadata_type,
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
            attention_dispatch_metadata,
        ) = graph.inputs

        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_prompt_length.tensor,
            max_cache_length.tensor,
            attention_dispatch_metadata=attention_dispatch_metadata.tensor,
        )

        result = flash_attention_ragged(
            kv_params,
            input=q.tensor,
            input_row_offsets=input_row_offsets.tensor,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / math.sqrt(head_dim),
        )

        graph.output(result)

    # Compile model
    model = session.load(graph)

    # Prepare inputs - for decode, each sequence has 1 token
    # Total tokens = batch_size
    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=dtype, device="cuda"
    )

    # Input row offsets: [0, 1, 2, ..., batch_size] (one token per sequence)
    input_row_offsets = torch.arange(
        batch_size + 1, dtype=torch.int32, device="cuda"
    ).to(torch.uint32)

    def run_kernel() -> Any:
        output = model.execute(
            q_input.detach(),
            input_row_offsets.detach(),
            paged_blocks_max,
            cache_lengths_max,
            lut_max,
            max_prompt_length_max,
            max_cache_length_max,
            attention_dispatch_metadata_max,
        )[0]
        return output

    # Warmup
    for _ in range(10):
        run_kernel()

    # Use bench_kineto to profile the kernel.
    # Split-K decode launches up to two kernels:
    #   sm100_mha_1q_depth128_..._<hash>    (FA4 SM100MHA2Q main decode; the
    #                                       num_q=1 variant of the @__name
    #                                       "sm100_mha_{nq}q_depth{d}_...")
    #   sm100_splitk_combine_..._<hash>     (fa4_splitk_combine reduction,
    #                                       when partitions > 1)
    time_s = bench_kineto(
        run_kernel,
        kernel_names="sm100_mha_",
        num_tests=100,
        suppress_kineto_output=True,
        flush_l2=True,
        with_multiple_kernels=True,
    )
    assert isinstance(time_s, float)  # Single kernel_name returns float

    total_bytes = (
        2
        * batch_size
        * (num_heads * head_dim + cache_len * num_kv_heads * head_dim)
        * 2
    )

    return time_s, total_bytes


def bench_decode(
    batch_size: int,
    cache_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    engine: str,  # "modular_max" or "flashinfer"
) -> tuple[float, int] | None:
    """Run all MHA decode benchmarks and display results side-by-side.

    Args:
        batch_size: Batch size (number of sequences)
        cache_len: KV cache length per sequence
        num_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Dimension of each head
        dtype: torch dtype for inputs (e.g., torch.bfloat16)
        engine: backend to run the benchmark ("modular_max" or "flashinfer")
    """
    print("=" * 80)
    print(
        f"MHA Decode Benchmark (batch={batch_size}, cache_len={cache_len}, "
        f"q_heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim})"
    )
    print("=" * 80)

    result: tuple[float, int] | None = None
    if engine == "flashinfer":
        # Run FlashInfer benchmark with TensorRT-LLM backend
        if _flashinfer is not None:
            try:
                result = bench_flashinfer(
                    batch_size,
                    cache_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    64,  # TRTLLM's page table size
                    dtype,
                    use_tensor_cores=True,
                    backend="trtllm",
                )
            except Exception as e:
                print(f"FlashInfer benchmark failed: {e}")
                import traceback

                traceback.print_exc()

    # Run MAX benchmark
    if engine == "modular_max":
        try:
            result = bench_max(
                batch_size,
                cache_len,
                num_heads,
                num_kv_heads,
                head_dim,
                128,  # MAX's page size
                dtype,
            )
        except Exception as e:
            print(f"MAX benchmark failed: {e}")
            import traceback

            traceback.print_exc()
    return result


def main() -> None:
    """Main entry point for the MHA decode benchmark."""

    parser = argparse.ArgumentParser(description="MHA Decode Benchmark")
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
        "--num_heads",
        "--num-heads",
        type=int,
        default=4,
        help="Number of query heads",
    )
    parser.add_argument(
        "--num_kv_heads",
        "--num-kv-heads",
        type=int,
        default=4,
        help="Number of KV heads",
    )
    parser.add_argument(
        "--head_dim", "--head-dim", type=int, default=128, help="Head dimension"
    )
    parser.add_argument(
        "--page_size", "--page-size", type=int, default=128, help="Page size"
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
        choices=["modular_max", "flashinfer"],
        help="Engine",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.csv",
        help="Output path",
    )
    args, _ = parser.parse_known_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if args.engine not in ["flashinfer", "modular_max"]:
        raise ValueError(f"engine {args.engine} is not supported!")

    # Decode benchmark: batch_size, cache_len, num_heads, num_kv_heads, head_dim, page_size
    result = bench_decode(
        batch_size=args.batch_size,
        cache_len=args.cache_len,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype=dtype_map[args.dtype],
        engine=args.engine,
    )

    met_sec, bytes = result or [0, 0]
    bytes_per_sec = ThroughputMeasure(Bench.bytes, bytes)

    name = (
        f"Decode/batch_size={args.batch_size}/cache_len={args.cache_len}/"
        f"num_heads={args.num_heads}/num_kv_heads={args.num_kv_heads}/"
        f"head_dim={args.head_dim}/dtype={dtype_map[args.dtype]}/"
        f"engine={args.engine}/"
    )

    b = Bench(
        name,
        iters=1,
        met=met_sec,
        metric_list=[bytes_per_sec],
    )

    b.dump_report(output_path=args.output)


if __name__ == "__main__":
    main()
