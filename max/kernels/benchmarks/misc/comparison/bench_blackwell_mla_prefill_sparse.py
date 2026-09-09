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

# MAX-only benchmark for the sparse MLA prefill kernel (mo.mla.prefill.sparse.paged).
# Measures the same performance shapes as FlashMLA's test_flash_mla_sparse_prefill.py
# so results can be compared directly against FlashMLA's published numbers.
#
# Run directly (no kbench):
#   python bench_blackwell_mla_prefill_sparse.py
#   python bench_blackwell_mla_prefill_sparse.py --num_iters 200
#   python bench_blackwell_mla_prefill_sparse.py --no-kineto   # for ncu/nsys
#
# Output format matches FlashMLA:
#   Prefill:  NNN us, NNNN.N TFlops, N.NN TBps

from __future__ import annotations

import argparse
import math
from typing import Any

import torch
from bench import bench_kineto_with_cupti_warmup
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import PagedCacheValues

# Must match PAGE_SIZE and SOFTMAX_SCALE_BASE_DIM in the Mojo kernel.
_PAGE_SIZE = 128
_SOFTMAX_SCALE_BASE_DIM = 192
_QK_ROPE_HEAD_DIM = (
    64  # rope portion of qk_depth, constant across all supported shapes
)

# Performance shapes from FlashMLA's test_flash_mla_sparse_prefill.py.
# Tuple layout: (qk_depth, v_depth, h_q, topk, s_kv_list). s_q is fixed at 4096.
_PERF_CASES: list[tuple[int, int, int, int, list[int]]] = [
    # DSv3.2 — only shape currently instantiated in the kernel
    (576, 512, 128, 2048, [8192, 32768, 65536, 98304, 131072]),
]


def bench_max(
    s_q: int,
    s_kv: int,
    topk: int,
    h_q: int,
    qk_depth: int,
    v_depth: int,
    num_iters: int = 100,
    no_kineto: bool = False,
    verbose_kineto: bool = False,
) -> tuple[float, int, int] | None:
    """Benchmark MAX sparse MLA prefill for one (s_q, s_kv, topk, h_q, qk_depth) shape.

    Args:
        s_q: Number of query tokens (total_q, single batch).
        s_kv: Number of cached KV tokens in the paged KV cache.
        topk: Number of sparse KV tokens selected per query.
        h_q: Number of query heads.
        qk_depth: QK head dimension (576 for DSv3.2, includes rope).
        v_depth: V head dimension (qk_depth - _QK_ROPE_HEAD_DIM).
        num_iters: Kineto profiling iterations.
        no_kineto: Skip kineto (use for ncu/nsys profiling).
        verbose_kineto: Print full Kineto profiling table.

    Returns:
        (time_s, flops, mem_bytes) or None on failure.
    """
    scale = 1.0 / math.sqrt(_SOFTMAX_SCALE_BASE_DIM)
    num_pages = math.ceil(s_kv / _PAGE_SIZE)

    q_torch = torch.randn(
        s_q, h_q, qk_depth, dtype=torch.bfloat16, device="cuda"
    )
    kv_blocks_torch = torch.randn(
        num_pages,
        1,
        1,
        _PAGE_SIZE,
        1,
        qk_depth,
        dtype=torch.bfloat16,
        device="cuda",
    )
    # batch_size=1, so one row of sequential page IDs.
    lut_torch = (
        torch.arange(num_pages, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .to(torch.uint32)
    )
    cache_lengths_torch = torch.tensor(
        [s_kv], dtype=torch.uint32, device="cuda"
    )
    max_prompt_length_torch = torch.tensor(
        [s_q],
        dtype=torch.uint32,
        device="cpu",  # must be on CPU
    )
    max_cache_length_torch = torch.tensor(
        [s_kv],
        dtype=torch.uint32,
        device="cpu",  # must be on CPU
    )
    input_row_offsets_torch = torch.tensor(
        [0, s_q], dtype=torch.uint32, device="cuda"
    )
    # Logical token indices — the op remaps to physical page offsets internally.
    sparse_indices_torch = torch.randint(
        0, s_kv, (s_q, topk), dtype=torch.int32, device="cuda"
    )
    topk_lengths_torch = torch.full(
        (s_q,), topk, dtype=torch.int32, device="cuda"
    )
    attn_sink_torch = torch.zeros(s_q, dtype=torch.float32, device="cuda")

    kv_blocks_max = Buffer.from_dlpack(kv_blocks_torch)
    lut_max = Buffer.from_dlpack(lut_torch)
    cache_lengths_max = Buffer.from_dlpack(cache_lengths_torch)
    max_prompt_length_max = Buffer.from_dlpack(max_prompt_length_torch)
    max_cache_length_max = Buffer.from_dlpack(max_cache_length_torch)
    input_row_offsets_max = Buffer.from_dlpack(input_row_offsets_torch)
    sparse_indices_max = Buffer.from_dlpack(sparse_indices_torch)
    topk_lengths_max = Buffer.from_dlpack(topk_lengths_torch)
    attn_sink_max = Buffer.from_dlpack(attn_sink_torch)

    q_type = TensorType(
        DType.bfloat16, shape=[s_q, h_q, qk_depth], device=DeviceRef.GPU()
    )
    blocks_type = BufferType(
        DType.bfloat16,
        shape=[num_pages, 1, 1, _PAGE_SIZE, 1, qk_depth],
        device=DeviceRef.GPU(),
    )
    cache_lengths_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.GPU()
    )
    lookup_table_type = TensorType(
        DType.uint32, shape=[1, num_pages], device=DeviceRef.GPU()
    )
    max_prompt_length_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )
    max_cache_length_type = TensorType(
        DType.uint32, shape=[1], device=DeviceRef.CPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=[2], device=DeviceRef.GPU()
    )
    sparse_indices_type = TensorType(
        DType.int32, shape=[s_q, topk], device=DeviceRef.GPU()
    )
    topk_lengths_type = TensorType(
        DType.int32, shape=[s_q], device=DeviceRef.GPU()
    )
    attn_sink_type = TensorType(
        DType.float32, shape=[s_q], device=DeviceRef.GPU()
    )

    with Graph(
        "mla_prefill_sparse_bench",
        input_types=[
            q_type,
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_prompt_length_type,
            max_cache_length_type,
            input_row_offsets_type,
            sparse_indices_type,
            topk_lengths_type,
            attn_sink_type,
        ],
    ) as graph:
        (
            q_g,
            blocks_g,
            cache_lengths_g,
            lookup_table_g,
            max_prompt_length_g,
            max_cache_length_g,
            input_row_offsets_g,
            sparse_indices_g,
            topk_lengths_g,
            attn_sink_g,
        ) = graph.inputs

        kv_collection = PagedCacheValues(
            blocks_g.buffer,
            cache_lengths_g.tensor,
            lookup_table_g.tensor,
            max_prompt_length_g.tensor,
            max_cache_length_g.tensor,
        )

        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())
        scale_cst = ops.constant(scale, DType.float32, DeviceRef.CPU())

        result = ops.inplace_custom(
            "mo.mla.prefill.sparse.paged",
            device=DeviceRef.GPU(),
            values=[
                q_g.tensor,
                *kv_collection.flatten_without_attention_dispatch_metadata(),
                layer_idx,
                input_row_offsets_g.tensor,
                sparse_indices_g.tensor,
                topk_lengths_g.tensor,
                attn_sink_g.tensor,
                scale_cst,
            ],
            out_types=[
                TensorType(
                    DType.bfloat16,
                    shape=[s_q, h_q, v_depth],
                    device=DeviceRef.GPU(),
                )
            ],
            parameters={"indices_stride": topk},
        )[0].tensor

        graph.output(result)

    session = InferenceSession(devices=[Accelerator()])
    model = session.load(graph)

    def run_kernel() -> Any:
        return model.execute(
            q_torch.detach(),
            kv_blocks_max,
            cache_lengths_max,
            lut_max,
            max_prompt_length_max,
            max_cache_length_max,
            input_row_offsets_max,
            sparse_indices_max,
            topk_lengths_max,
            attn_sink_max,
        )[0]

    # Warmup with L2 flushes to reach stable GPU power/clock state.
    flush_l2_size = int(1e9 // 4)
    for _ in range(20):
        torch.empty(flush_l2_size, dtype=torch.int, device="cuda").zero_()
        run_kernel()

    # Matches FlashMLA's lib.py: 2 * total_topk * h_q * (d_qk + d_v).
    flops = 2 * s_q * topk * h_q * (qk_depth + v_depth)
    mem_bytes = (
        s_q * h_q * qk_depth  # Q read
        + s_q * topk * qk_depth  # sparse KV read
        + s_q * h_q * v_depth  # output write
    ) * 2

    if no_kineto:
        run_kernel()
        torch.cuda.synchronize()
        return 1.0, flops, mem_bytes

    # Kernel name in the Kineto table:
    # mla_prefill_sparse_bfloat16_nqh{H}_nkvh1_{hash}
    # "mla_prefill_sparse" is a stable substring across all h_q variants.
    # Use --verbose-kineto to print the full table if the name ever changes.
    try:
        time_s = bench_kineto_with_cupti_warmup(
            run_kernel,
            kernel_names="mla_prefill_sparse",
            num_tests=num_iters,
            suppress_kineto_output=not verbose_kineto,
            flush_l2=True,
        )
        assert isinstance(time_s, float)
    except RuntimeError as e:
        if "No kernel times found" in str(e):
            print(
                f"  Warning: kineto could not find kernel 'sm100_mla_prefill_sparse'. "
                f"Run with --no-kineto or check kernel name in Kineto table. Error: {e}"
            )
            return None
        raise

    return time_s, flops, mem_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAX sparse MLA prefill benchmark (mo.mla.prefill.sparse.paged)"
    )
    parser.add_argument(
        "--s_q", "--s-q", type=int, default=4096, help="Query sequence length"
    )
    parser.add_argument(
        "--num_iters",
        "--num-iters",
        type=int,
        default=100,
        help="Kineto profiling iterations",
    )
    parser.add_argument(
        "--no-kineto",
        action="store_true",
        help="Skip kineto timing (for ncu/nsys)",
    )
    parser.add_argument(
        "--verbose-kineto",
        action="store_true",
        help="Print full Kineto profiling table (use to find correct kernel name)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    for qk_depth, v_depth, h_q, topk, s_kv_list in _PERF_CASES:
        for s_kv in s_kv_list:
            label = (
                f"s_q={args.s_q} s_kv={s_kv} topk={topk} "
                f"h_q={h_q} qk_depth={qk_depth} v_depth={v_depth}"
            )
            try:
                result = bench_max(
                    s_q=args.s_q,
                    s_kv=s_kv,
                    topk=topk,
                    h_q=h_q,
                    qk_depth=qk_depth,
                    v_depth=v_depth,
                    num_iters=args.num_iters,
                    no_kineto=args.no_kineto,
                    verbose_kineto=args.verbose_kineto,
                )
            except Exception as e:
                print(f"  SKIP ({label}): {e}")
                continue

            if result is None:
                print(
                    f"  SKIP ({label}): kernel name mismatch (see warning above)"
                )
                continue

            time_s, flops, mem_bytes = result
            tflops = flops / time_s / 1e12
            tbps = mem_bytes / time_s / 1e12
            print(
                f"Prefill: {time_s * 1e6:4.0f} us, {tflops:6.1f} TFlops,"
                f" {tbps:4.2f} TBps  [{label}]"
            )


if __name__ == "__main__":
    main()
