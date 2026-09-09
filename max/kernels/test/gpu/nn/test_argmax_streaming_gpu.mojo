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
"""Differential tests for the streaming GPU argmax/argmin.

The reference is a scalar first-index scan, which is what `numpy.argmax`
returns for tie-free and tied rows alike. NaN is skipped rather than
propagated (a NaN never compares greater), matching `TopK_2.insert`; an
all-NaN or all-sentinel row therefore reports index 0.
"""

from std.random import random_float64, seed
from std.utils.index import IndexList
from std.utils.numerics import inf, max_or_inf, min_or_neg_inf, nan, neg_inf

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.argmaxmin_gpu import argmax_gpu, argmin_gpu


def _reference_index[
    dtype: DType, largest: Bool
](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], num_elements: Int) -> Int:
    """Scalar first-index scan: strict comparison, so ties keep the lowest index.
    """
    var best: Scalar[dtype]
    comptime if largest:
        best = min_or_neg_inf[dtype]()
    else:
        best = max_or_inf[dtype]()
    var best_idx = 0
    for i in range(num_elements):
        var val = row[i]
        comptime if largest:
            if val > best:
                best = val
                best_idx = i
        else:
            if val < best:
                best = val
                best_idx = i
    return best_idx


def _check[
    dtype: DType,
    out_idx_type: DType,
    largest: Bool,
    fill_fn: def[dtype: DType](
        MutPointer[Scalar[dtype], MutUntrackedOrigin], Int, Int
    ) capturing[_] -> None,
](ctx: DeviceContext, batch: Int, num_elements: Int, label: String) raises:
    var in_host = ctx.enqueue_create_host_buffer[dtype](batch * num_elements)
    var out_host = ctx.enqueue_create_host_buffer[out_idx_type](batch)
    ctx.synchronize()

    for b in range(batch):
        fill_fn[dtype](in_host.unsafe_ptr() + b * num_elements, num_elements, b)

    var in_dev = ctx.enqueue_create_buffer[dtype](batch * num_elements)
    var out_dev = ctx.enqueue_create_buffer[out_idx_type](batch)
    ctx.enqueue_copy(in_dev, in_host)

    var in_shape = IndexList[2](batch, num_elements)
    var out_shape = IndexList[2](batch, 1)
    var in_tensor = TileTensor(in_dev, row_major(Coord(in_shape)))
    var out_tensor = TileTensor(out_dev, row_major(Coord(out_shape)))

    comptime if largest:
        argmax_gpu(ctx, in_tensor, out_tensor)
    else:
        argmin_gpu(ctx, in_tensor, out_tensor)

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for b in range(batch):
        var expected = _reference_index[dtype, largest](
            in_host.unsafe_ptr() + b * num_elements, num_elements
        )
        if Int(out_host[b]) != expected:
            raise Error(
                t"{label}: dtype={dtype} largest={largest} batch={batch}"
                t" N={num_elements} row={b} got={out_host[b]}"
                t" expected={expected}"
            )

    _ = in_dev^
    _ = out_dev^


def _run_dtype[dtype: DType, out_idx_type: DType](ctx: DeviceContext) raises:
    @__parameter
    def fill_random[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = random_float64(-1e4, 1e4).cast[dtype]()

    @__parameter
    def fill_all_equal[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = Scalar[dtype](1.5)

    @__parameter
    def fill_peak_first[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = Scalar[dtype](-1.0)
        row[0] = Scalar[dtype](7.0)

    @__parameter
    def fill_peak_last[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = Scalar[dtype](-1.0)
        row[n - 1] = Scalar[dtype](7.0)

    @__parameter
    def fill_infs[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = neg_inf[dtype]() if i % 3 == 0 else Scalar[dtype](
                Float64(i % 17) - 8.0
            )
        if n > 5:
            row[n // 2] = inf[dtype]()
            row[n - 2] = inf[dtype]()

    @__parameter
    def fill_all_neg_inf[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = neg_inf[dtype]()

    @__parameter
    def fill_nans[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = nan[dtype]() if i % 5 == 0 else Scalar[dtype](
                Float64(i % 11) - 5.0
            )

    @__parameter
    def fill_all_nan[
        dtype: DType
    ](row: MutPointer[Scalar[dtype], MutUntrackedOrigin], n: Int, b: Int):
        for i in range(n):
            row[i] = nan[dtype]()

    # Vector width is 16B, so N values not divisible by 4 (f32) or 8 (bf16)
    # exercise the ragged tail; the small N also cover sub-vector rows.
    var edge_ns = [1, 3, 7, 8, 9, 31, 33, 127, 129, 1023, 4095, 4097]
    for i in range(len(edge_ns)):
        _check[dtype, out_idx_type, True, fill_random](
            ctx, 3, edge_ns[i], "edge-random-max"
        )
        _check[dtype, out_idx_type, False, fill_random](
            ctx, 3, edge_ns[i], "edge-random-min"
        )

    var batches = [1, 8, 64]
    var sizes = [1024, 4097, 262144]
    for bi in range(len(batches)):
        for si in range(len(sizes)):
            var b = batches[bi]
            var n = sizes[si]
            _check[dtype, out_idx_type, True, fill_random](
                ctx, b, n, "random-max"
            )
            _check[dtype, out_idx_type, False, fill_random](
                ctx, b, n, "random-min"
            )
            _check[dtype, out_idx_type, True, fill_all_equal](
                ctx, b, n, "all-equal"
            )
            _check[dtype, out_idx_type, True, fill_peak_first](
                ctx, b, n, "peak-first"
            )
            _check[dtype, out_idx_type, True, fill_peak_last](
                ctx, b, n, "peak-last"
            )
            _check[dtype, out_idx_type, True, fill_infs](ctx, b, n, "infs-max")
            _check[dtype, out_idx_type, False, fill_infs](ctx, b, n, "infs-min")
            _check[dtype, out_idx_type, True, fill_all_neg_inf](
                ctx, b, n, "all-neg-inf"
            )
            _check[dtype, out_idx_type, True, fill_nans](ctx, b, n, "nans-max")
            _check[dtype, out_idx_type, False, fill_nans](ctx, b, n, "nans-min")
            _check[dtype, out_idx_type, True, fill_all_nan](
                ctx, b, n, "all-nan"
            )

    # A batch past 65535 does not fit a CUDA grid's y or z extent, so the row
    # count has to ride x. Short rows keep this to one split per row, which is
    # the shape a vocab-sized argmax over a large token batch actually launches.
    _check[dtype, out_idx_type, True, fill_random](
        ctx, 65537, 8, "wide-batch-max"
    )
    _check[dtype, out_idx_type, False, fill_random](
        ctx, 65537, 8, "wide-batch-min"
    )


def main() raises:
    seed(24301)
    with DeviceContext() as ctx:
        _run_dtype[.float32, DType.int64](ctx)
        _run_dtype[.bfloat16, DType.int64](ctx)
        _run_dtype[.bfloat16, DType.int32](ctx)
    print("OK")
