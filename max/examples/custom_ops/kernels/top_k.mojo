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

import extensibility

from max.gpu.host import DeviceContext
from std.math import iota
from std.sys import align_of, size_of

from max.algorithm import parallelize_over_rows
from std.bit import log2_floor
from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.primitives import warp
from max.gpu.memory import external_memory
from std.collections import Span

from extensibility import InputTensor, OutputTensor

from std.utils.numerics import min_or_neg_inf


@fieldwise_init
struct TopKElement[T: DType](Comparable, TrivialRegisterPassable):
    """Stores the value with it's index."""

    var idx: Int32
    var val: Scalar[Self.T]

    def __eq__(self, rhs: Self) -> Bool:
        return self.val == rhs.val

    def __lt__(self, rhs: Self) -> Bool:
        return self.val < rhs.val


@extensibility.register("top_k_custom")
struct TopK:
    """Registers the `top_k_custom` op, allowing python to use it from the `max`
    package. This is a simplified version without bottom_k and sorting options,
    or fused sampling. The purpose is to demonstrate concisely how you can
    implement your own custom ops in Mojo that can be called from Python. MAX
    has the "mo.top_k" op which is feature complete.
    """

    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        //,  # Forces the previous two params to be inferred from the args
        K: Int,
        target: StaticString,
    ](
        out_vals: OutputTensor[dtype=dtype, rank=rank, ...],
        out_idxs: OutputTensor[dtype=.int32, rank=rank, ...],
        in_vals: InputTensor[dtype=dtype, rank=rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert rank == 2, "rank must be 2"
        comptime assert not (
            target == "gpu" and K > WARP_SIZE
        ), "K can't be larger than warp size"

        var shape = in_vals.shape()
        var batch_size = shape[0]
        var dev_ctx = ctx

        var out_vals_tensor = out_vals.to_layout_tensor()
        var out_idxs_tensor = out_idxs.to_layout_tensor()
        var in_vals_tensor = in_vals.to_layout_tensor()

        @__parameter
        def top_k_gpu[
            K: Int,
        ](
            out_vals: type_of(out_vals_tensor),
            out_idxs: type_of(out_idxs_tensor),
            in_vals: type_of(in_vals_tensor),
        ):
            var bid = block_idx.x
            var tid = thread_idx.x

            # Get a pointer to shared memory for the indices and values
            var top_k_sram = external_memory[
                TopKElement[dtype],
                address_space=.SHARED,
                alignment=align_of[TopKElement[dtype]](),
            ]()

            # Threads put their corresponding index and value into shared memory
            top_k_sram[unsafe_offset=tid] = TopKElement(
                Int32(tid), in_vals[bid, tid][0]
            )
            # Finish packing the values across threads in this block
            barrier()

            comptime for i in range(K):
                var reduced = top_k_sram[unsafe_offset=tid]
                comptime limit = log2_floor(WARP_SIZE)

                # TODO(KERN-1544): `gpu.shuffle.warp_max` support index/value
                comptime for j in reversed(range(limit)):
                    comptime offset = 1 << j
                    # Parallel reduction using warp shuffle. Each thread gets a
                    # value from a thread 'offset' positions higher, keeping the
                    # larger value.
                    var shuffled = TopKElement(
                        warp.shuffle_down(reduced.idx, UInt32(offset)),
                        warp.shuffle_down(reduced.val, UInt32(offset)),
                    )
                    reduced = max(reduced, shuffled)

                # Wait for all threads to finish reducing their values
                barrier()

                # Thread 0 now has the reduced max value for this index
                if tid == 0:
                    # Store the reduced top_k index and value in global memory
                    out_vals[bid, i] = reduced.val
                    out_idxs[bid, i] = reduced.idx

                    # Remove found maximum from consideration in the next iter
                    var index = reduced.idx % Int32(block_dim.x)
                    top_k_sram[unsafe_offset=index].val = min_or_neg_inf[
                        dtype
                    ]()

        comptime if target == "gpu":
            dev_ctx.enqueue_function[top_k_gpu[K]](
                out_vals_tensor,
                out_idxs_tensor,
                in_vals_tensor,
                grid_dim=batch_size,  # One block per batch
                block_dim=K,  # One thread per K
                shared_mem_bytes=K * size_of[TopKElement[dtype]](),
            )
        else:

            def top_k_cpu(start_idx: Int, end_idx: Int) {imm}:
                for row_idx in range(start_idx, end_idx):
                    var offset = row_idx * K
                    iota(out_idxs.unsafe_ptr().unsafe_offset(offset), K)

                    def val_greater_than(
                        lhs: Int32, rhs: Int32
                    ) {in_vals, row_idx} -> Bool:
                        return (
                            in_vals[row_idx, Int(lhs)]
                            > in_vals[row_idx, Int(rhs)]
                        )

                    sort(
                        Span(
                            unsafe_ptr=out_idxs.unsafe_ptr().unsafe_offset(
                                offset
                            ),
                            length=K,
                        ),
                        val_greater_than,
                    )

                    for i in range(K):
                        var sorted_idx = Int(out_idxs[row_idx, i])
                        out_vals[row_idx, i] = in_vals[row_idx, sorted_idx]

            parallelize_over_rows(top_k_cpu, shape, axis=1, grain_size=1)
