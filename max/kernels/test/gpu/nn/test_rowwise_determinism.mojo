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
"""Run-to-run byte-determinism gate for Row-API rowwise ops.

Launches the same op on the same device input N times and requires all N
outputs to be byte-identical and NaN-free. This catches execution-order
nondeterminism in the block-tier cross-thread combines — the class behind
the Metal layer_norm regression introduced by #90697: the trailing
shared-memory broadcast read in a block reduce is not ordered against the
next combine's first store, so a block that grid-strides several rows
races its own shared-memory strip across row iterations.

The three bodies here cover both cross-thread combine implementations
every Row-routed op shares: layer_norm exercises `BlockReducer.generic`
(Welford has no hardware reduce), rms_norm and softmax exercise
`block.sum` / `block.max` (`_block_reduce_with_padding`). Other Row
bodies (log_softmax, row_mean_of_squares, the fused rms_norm variants,
reduce_min_and_max, arg_max/arg_min) combine through the same two
primitives.

Failure polarity, measured on an Apple M5 before the fix (each probe =
8 launches in one process): layer_norm fp32 4096x3072 diverged 8/8
launches in two independent processes; rms_norm fp32 diverged 6/8 and
bf16 8/8; the bf16 layer_norm divergence is intermittent (7/8 distinct
in one process, stable in another) — the race window narrows with
per-thread tile count, so the fp32 configs are the reliable canaries.
Divergence onset tracks rows per block (the grid caps at
sm_count * 32 = 320 blocks on M5): 320- and 512-row shapes never
diverged in 8-launch samples, 1024 rows diverged 5/8, 2048 and 4096
rows 8/8.

Run locally on any GPU machine, including any Apple-silicon Mac:

    ./bazelw test //max/kernels/test/gpu/nn:test_rowwise_determinism.mojo.test
"""

from std.math import isnan
from std.sys import align_of, size_of

from max.gpu.host import DeviceContext, HostBuffer
from layout import Coord, TileTensor, row_major
from nn.normalization import *
from nn.softmax import softmax
from std.utils.coord import ComptimeInt
from std.utils.index import Index, IndexList

comptime _NUM_RUNS = 8


def _fill_pseudorandom[dtype: DType](buf: HostBuffer[dtype], n: Int):
    """Fills with a deterministic LCG stream in [-1, 1)."""
    var state = UInt64(0x9E3779B97F4A7C15)
    for i in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = (state >> 33) % 65536
        buf[i] = ((Float64(Int(u)) / 32768.0) - 1.0).cast[dtype]()


def _fnv1a64[dtype: DType](buf: HostBuffer[dtype], n: Int) -> UInt64:
    """FNV-1a over the buffer's raw bytes."""
    var ptr = buf.unsafe_ptr().bitcast[UInt8]()
    var h = UInt64(0xCBF29CE484222325)
    for i in range(n * size_of[dtype]()):
        h = (h ^ UInt64(Int(ptr[i]))) * UInt64(0x100000001B3)
    return h


def _count_nans[dtype: DType](buf: HostBuffer[dtype], n: Int) -> Int:
    var count = 0
    for i in range(n):
        if isnan(buf[i]):
            count += 1
    return count


def _verdict(hashes: List[UInt64], nans: Int) -> Bool:
    var all_equal = True
    for r in range(len(hashes)):
        if hashes[r] != hashes[0]:
            all_equal = False
    if not all_equal or nans != 0:
        for r in range(len(hashes)):
            print("  run", r, "hash", hashes[r])
    print("  nans:", nans, "| byte-identical:", all_equal)
    return all_equal and nans == 0


def _probe_layer_norm[
    dtype: DType, rows: Int, cols: Int
](ctx: DeviceContext) raises -> Bool:
    """Runs layer_norm N times on identical input; returns True when all
    N outputs are byte-identical and NaN-free."""
    print("== layer_norm", dtype, rows, "x", cols)

    var in_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var out_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)
    ctx.synchronize()

    _fill_pseudorandom(in_h, rows * cols)
    _fill_pseudorandom(gamma_h, cols)
    _fill_pseudorandom(beta_h, cols)

    var in_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(in_d, in_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    var shape = Index(rows, cols)
    var in_buf = TileTensor(in_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))
    var beta = TileTensor(beta_d, row_major(Coord(Index(cols))))
    var epsilon = Scalar[dtype](1e-5)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
        var idx = in_buf.layout(coords)
        return in_buf.raw_load[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_buf}:
        var idx = out_buf.layout(coords)
        out_buf.raw_store[width=width, alignment=alignment * align_of[dtype]()](
            idx, rebind[SIMD[dtype, width]](val)
        )

    var hashes = List[UInt64]()
    var nans = 0
    for _run in range(_NUM_RUNS):
        # Poison the output between runs so a skipped launch cannot fake
        # byte-identical results.
        ctx.enqueue_memset(out_d, Scalar[dtype](0))
        # Static cols (`ComptimeInt`) matches the graph op's static-shape
        # branch, which is what models hit: it enables the Row register
        # cache (`is_cached` fuse path).
        layer_norm[dtype, 2, target="gpu"](
            input_fn,
            output_fn,
            Coord(shape),
            ComptimeInt[cols](),
            gamma,
            beta,
            epsilon,
            ctx,
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        hashes.append(_fnv1a64(out_h, rows * cols))
        nans += _count_nans(out_h, rows * cols)

    _ = in_buf
    _ = in_d^
    _ = out_d^
    _ = gamma_d^
    _ = beta_d^
    return _verdict(hashes, nans)


def _probe_rms_norm[
    dtype: DType, rows: Int, cols: Int
](ctx: DeviceContext) raises -> Bool:
    """Runs rms_norm N times on identical input; returns True when all
    N outputs are byte-identical and NaN-free."""
    print("== rms_norm", dtype, rows, "x", cols)

    var in_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var out_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    ctx.synchronize()

    _fill_pseudorandom(in_h, rows * cols)
    _fill_pseudorandom(gamma_h, cols)

    var in_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    ctx.enqueue_copy(in_d, in_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    var shape = Index(rows, cols)
    var in_buf = TileTensor(in_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(Index(cols))))
    var epsilon = Scalar[dtype](1e-5)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
        var idx = in_buf.layout(coords)
        return in_buf.raw_load[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_buf}:
        var idx = out_buf.layout(coords)
        out_buf.raw_store[width=width, alignment=alignment * align_of[dtype]()](
            idx, rebind[SIMD[dtype, width]](val)
        )

    var hashes = List[UInt64]()
    var nans = 0
    for _run in range(_NUM_RUNS):
        ctx.enqueue_memset(out_d, Scalar[dtype](0))
        rms_norm[dtype, 2, target="gpu"](
            input_fn,
            output_fn,
            Coord(shape),
            ComptimeInt[cols](),
            gamma,
            epsilon,
            Scalar[dtype](0),
            ctx,
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        hashes.append(_fnv1a64(out_h, rows * cols))
        nans += _count_nans(out_h, rows * cols)

    _ = in_buf
    _ = in_d^
    _ = out_d^
    _ = gamma_d^
    return _verdict(hashes, nans)


def _probe_softmax[
    dtype: DType, rows: Int, cols: Int
](ctx: DeviceContext) raises -> Bool:
    """Runs softmax N times on identical input; returns True when all
    N outputs are byte-identical and NaN-free."""
    print("== softmax", dtype, rows, "x", cols)

    var in_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var out_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    ctx.synchronize()

    _fill_pseudorandom(in_h, rows * cols)

    var in_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    ctx.enqueue_copy(in_d, in_h)

    var shape = Index(rows, cols)
    var in_buf = TileTensor(in_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var in_buf} -> SIMD[dtype, width]:
        var idx = in_buf.layout(coords)
        return in_buf.raw_load[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx)

    var hashes = List[UInt64]()
    var nans = 0
    for _run in range(_NUM_RUNS):
        ctx.enqueue_memset(out_d, Scalar[dtype](0))
        softmax[dtype, 2, target="gpu"](
            input_fn,
            Coord(shape),
            ComptimeInt[cols](),
            out_buf,
            1,
            ctx,
        )
        ctx.enqueue_copy(out_h, out_d)
        ctx.synchronize()
        hashes.append(_fnv1a64(out_h, rows * cols))
        nans += _count_nans(out_h, rows * cols)

    _ = in_buf
    _ = in_d^
    _ = out_d^
    return _verdict(hashes, nans)


def main() raises:
    with DeviceContext() as ctx:
        var ok = True
        # Block tier through BlockReducer.generic (Welford). fp32 is the
        # reliable canary on Metal; bf16's window is intermittent. The
        # 4096-row shapes put ~13 rows on each block at a 320-block grid,
        # which is where the pre-fix race manifested.
        ok &= _probe_layer_norm[.float32, 4096, 3072](ctx)
        ok &= _probe_layer_norm[.bfloat16, 4096, 3072](ctx)
        # Warp tier (register-only combine, no shared memory): guards
        # tier-selection changes.
        ok &= _probe_layer_norm[.bfloat16, 512, 1024](ctx)
        # Block tier through block.sum (_block_reduce_with_padding).
        ok &= _probe_rms_norm[.float32, 4096, 3072](ctx)
        ok &= _probe_rms_norm[.bfloat16, 4096, 3072](ctx)
        # Two dependent combines per row (block.max then block.sum).
        ok &= _probe_softmax[.float32, 4096, 3072](ctx)
        ok &= _probe_softmax[.bfloat16, 4096, 3072](ctx)
        if not ok:
            raise Error(
                "run-to-run nondeterminism (or NaNs) detected; see the"
                " per-run hashes above"
            )
        print("DETERMINISM: PASS")
