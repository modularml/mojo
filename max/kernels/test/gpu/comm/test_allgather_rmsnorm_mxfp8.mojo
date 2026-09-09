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
"""Byte-identity gate for the MXFP8 epilogue folded into all-gather + RMSNorm.

Byte-compares what `_dispatch_ag_norm_quant` produces against a genuine
`allgather_rmsnorm` -> `quantize_mx_amd` pair, on every GPU, over BOTH of the
dispatcher's branches. So the fold must quantize the bf16 that lands in
`normed_out` rather than the wider f32 it came from, must keep each 32-element
MX block inside one lane group, and must carry the quantize on the
above-threshold path.

The compare alone cannot tell WHICH branch ran -- byte-identity is the point of
the fold, so an inverted route still compares equal. Each case therefore poisons
the branch it must not take: below the threshold `two_launch_with_quant` raises,
above it the epilogue stores e4m3fn NaN (0x7F, which the quantizer's clamp to
0x7E cannot produce). A case that passes has proven its own route.

BLIND to any fault inside `quantize_mxfp8_lane_group`, since both arms call it
and the error cancels -- measured, not assumed: swapping its
`abs(val).reduce_max()` for a signed `val.reduce_max()` leaves every case here
at 0 mismatches while the host oracle in
`linalg/test_mxfp8_quant_amd_geometry.mojo` fails on the first block. That file
owns the numerics, this one owns the wiring.
"""

from std.memory import bitcast
from std.sys import has_amd_gpu_accelerator, simd_width_of, size_of
from std.testing import assert_equal, assert_true
from std.utils.index import Index

from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from layout import Coord, TileTensor, row_major

from comm import MAX_GPUS, Signal, group_end, group_start
from comm.allgather_rmsnorm import (
    _dispatch_ag_norm_quant,
    allgather_rmsnorm,
)
from comm.reducescatter import ReduceScatterConfig
from comm.sync import enable_p2p, init_signal_buffer
from linalg.block_scaled_quantization import (
    quantize_mx_amd,
    quantize_mxfp8_lane_group,
)
from linalg.fp4_utils import MXFP8_SF_VECTOR_SIZE

comptime H = 6144
"""The calibrated hidden size; the fuse threshold is only valid here."""


def _gathered_value[dtype: DType](row: Int, col: Int) -> Scalar[dtype]:
    """Deterministic input with a wide per-block dynamic range.

    The ladder matters: at a flat magnitude neighbouring MX blocks share an E8M0
    scale, so a misaligned lane group gets the right scale by accident.
    """
    var block = col // MXFP8_SF_VECTOR_SIZE
    var mag = Float64(1 << (block % 17)) / 256.0
    var base = Float64((row * 7 + col * 3) % 251) / 125.0 - 1.0
    return (base * mag).cast[dtype]()


def _run_case[
    in_dtype: DType, ngpus: Int, num_cols: Int, route_two_launch: Bool = False
](num_rows: Int, list_of_ctx: List[DeviceContext]) raises:
    """Byte-compare the fused epilogue against gather+norm followed by quantize.

    Arm B always enters through `_dispatch_ag_norm_quant`; `route_two_launch`
    moves the runtime `threshold` so both branches run at the SAME row count.
    """
    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()
    comptime assert (
        MXFP8_SF_VECTOR_SIZE % simd_width == 0
    ), "the MX block must be a whole number of per-thread slices"
    comptime scale_cols = num_cols // MXFP8_SF_VECTOR_SIZE

    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](1.0)
    var full_n = num_rows * num_cols

    var shard_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed_fused = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_fused = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var quant_ref = List[DeviceBuffer[.float8_e4m3fn]](capacity=ngpus)
    var quant_fused = List[DeviceBuffer[.float8_e4m3fn]](capacity=ngpus)
    var scales_ref = List[DeviceBuffer[.float8_e8m0fnu]](capacity=ngpus)
    var scales_fused = List[DeviceBuffer[.float8_e8m0fnu]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    var gamma_host = List(
        length=num_cols,
        fill_with=lambda (c: Int) -> Scalar[in_dtype]: (
            Float64(c + num_cols) / Float64(num_cols)
        ).cast[in_dtype](),
    )

    for i in range(ngpus):
        var shard_rows = config.rank_units(i)
        var shard_alloc = max(shard_rows * num_cols, 1)
        shard_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](shard_alloc)
        )
        var start_i = config.rank_unit_start(i)
        var h = List[Scalar[in_dtype]](
            length=shard_alloc, fill=Scalar[in_dtype](0)
        )
        for lr in range(shard_rows):
            for c in range(num_cols):
                h[lr * num_cols + c] = _gathered_value[in_dtype](
                    start_i + lr, c
                )
        list_of_ctx[i].enqueue_copy(shard_dev[i], h)
        _ = h^

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        normed_ref.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n)
        )
        normed_fused.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n)
        )
        sum_ref.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        sum_fused.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](full_n))
        quant_ref.append(
            list_of_ctx[i].enqueue_create_buffer[.float8_e4m3fn](full_n)
        )
        quant_fused.append(
            list_of_ctx[i].enqueue_create_buffer[.float8_e4m3fn](full_n)
        )
        scales_ref.append(
            list_of_ctx[i].enqueue_create_buffer[.float8_e8m0fnu](
                num_rows * scale_cols
            )
        )
        scales_fused.append(
            list_of_ctx[i].enqueue_create_buffer[.float8_e8m0fnu](
                num_rows * scale_cols
            )
        )

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](size_of[Signal]())
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    comptime ShardType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime FullType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime QuantType = TileTensor[
        mut=True,
        .float8_e4m3fn,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime ScaleType = TileTensor[
        mut=True,
        .float8_e8m0fnu,
        type_of(row_major(Coord(Index(0, scale_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]

    var in_shards = Array[_, ngpus](
        fill_with=lambda (i: Int) -> ShardType: ShardType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                shard_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
    )

    # --- Arm A: the shipping two-kernel chain. ---
    group_start()
    for i in range(ngpus):
        allgather_rmsnorm(
            in_shards,
            FullType(
                normed_ref[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(num_rows, num_cols))),
            ),
            FullType(
                sum_ref[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(num_rows, num_cols))),
            ),
            GammaType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    gamma_dev[i].unsafe_ptr()
                ),
                row_major(Coord(Index(num_cols))),
            ),
            epsilon,
            weight_offset,
            rank_sigs,
            list_of_ctx[i],
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    for i in range(ngpus):
        quantize_mx_amd(
            list_of_ctx[i],
            QuantType(
                quant_ref[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(num_rows, num_cols))),
            ),
            ScaleType(
                scales_ref[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(num_rows, scale_cols))),
            ),
            FullType(
                normed_ref[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(num_rows, num_cols))),
            ),
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Arm B: through the dispatcher, on the branch `threshold` selects. ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # `full_bytes <= threshold` fuses. 0 can never be reached, and the full
    # byte count always is, so the two arms straddle the branch at one shape.
    var threshold = 0 if route_two_launch else full_n * size_of[in_dtype]()

    group_start()
    for i in range(ngpus):
        var quant_view = QuantType(
            quant_fused[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var scale_view = ScaleType(
            scales_fused[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, scale_cols))),
        )
        var normed_view = FullType(
            normed_fused[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var sum_view = FullType(
            sum_fused[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        var gamma_view = GammaType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
        var ctx_i = list_of_ctx[i]

        # `@__copy_capture` is mandatory: without it these locals reach the
        # device as host-stack pointers and the stores land out of bounds.
        @__copy_capture(quant_view, scale_view)
        @__parameter
        @always_inline
        def mx_epilogue[
            width: Int
        ](row: Int, col: Int, val: SIMD[in_dtype, width]):
            comptime if route_two_launch:
                # Poison: `threshold = 0` forces the two-launch branch, so the
                # fused kernel must never reach this epilogue. Storing a byte
                # the quantizer cannot emit is what makes the route visible to
                # a compare that is otherwise blind to it.
                if col < num_cols:
                    quant_view.store[width=width](
                        Coord(row, col),
                        bitcast[.float8_e4m3fn](SIMD[.uint8, width](0x7F)),
                    )
            else:
                var quantized: SIMD[.float8_e4m3fn, width]
                var e8m0: Float8_e8m0fnu
                # The block max is a cross-lane reduction: lanes past
                # `num_cols` must participate (carrying 0) but must not store.
                quantized, e8m0 = quantize_mxfp8_lane_group[
                    DType.float8_e4m3fn,
                    DType.float8_e8m0fnu,
                    SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE,
                ](val)
                if col < num_cols:
                    quant_view.store[width=width](Coord(row, col), quantized)
                    if col % MXFP8_SF_VECTOR_SIZE == 0:
                        scale_view.store(
                            Coord(row, col // MXFP8_SF_VECTOR_SIZE), e8m0
                        )

        # This branch owes the SAME outputs as the fused one; dropping the
        # `quantize_mx_amd` leaves them stale and arm A catches it (verified).
        @__parameter
        @always_inline
        def two_launch_with_quant() raises:
            comptime if not route_two_launch:
                # Poison: `threshold = full_bytes` forces the fused branch, so
                # being called at all means the routing inverted -- which the
                # byte compare cannot see on its own.
                raise Error(
                    "_dispatch_ag_norm_quant took two_launch at"
                    " threshold=full_bytes; the fused epilogue was expected"
                )
            else:
                allgather_rmsnorm(
                    in_shards,
                    normed_view,
                    sum_view,
                    gamma_view,
                    epsilon,
                    weight_offset,
                    rank_sigs,
                    ctx_i,
                )
                quantize_mx_amd(ctx_i, quant_view, scale_view, normed_view)

        _dispatch_ag_norm_quant[
            two_launch_with_quant=two_launch_with_quant,
            quant_epilogue=mx_epilogue,
        ](
            in_shards,
            normed_view,
            sum_view,
            gamma_view,
            epsilon,
            weight_offset,
            rank_sigs,
            ctx_i,
            threshold=threshold,
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Compare, on every GPU. ---
    var total_quant_mismatch = 0
    var total_scale_mismatch = 0
    var total_normed_mismatch = 0
    var nonzero_quant = 0
    var distinct_scales = List[UInt8]()

    for i in range(ngpus):
        var qa = list_of_ctx[i].enqueue_create_host_buffer[.float8_e4m3fn](
            full_n
        )
        var qb = list_of_ctx[i].enqueue_create_host_buffer[.float8_e4m3fn](
            full_n
        )
        var sa = list_of_ctx[i].enqueue_create_host_buffer[
            DType.float8_e8m0fnu
        ](num_rows * scale_cols)
        var sb = list_of_ctx[i].enqueue_create_host_buffer[
            DType.float8_e8m0fnu
        ](num_rows * scale_cols)
        var na = list_of_ctx[i].enqueue_create_host_buffer[in_dtype](full_n)
        var nb = list_of_ctx[i].enqueue_create_host_buffer[in_dtype](full_n)
        list_of_ctx[i].enqueue_copy(qa, quant_ref[i])
        list_of_ctx[i].enqueue_copy(qb, quant_fused[i])
        list_of_ctx[i].enqueue_copy(sa, scales_ref[i])
        list_of_ctx[i].enqueue_copy(sb, scales_fused[i])
        list_of_ctx[i].enqueue_copy(na, normed_ref[i])
        list_of_ctx[i].enqueue_copy(nb, normed_fused[i])
        list_of_ctx[i].synchronize()

        for e in range(full_n):
            var a = bitcast[.uint8](qa[e])
            var b = bitcast[.uint8](qb[e])
            if a != b:
                total_quant_mismatch += 1
            if b != UInt8(0):
                nonzero_quant += 1
            if na[e] != nb[e]:
                total_normed_mismatch += 1
        for e in range(num_rows * scale_cols):
            var a = bitcast[.uint8](sa[e])
            var b = bitcast[.uint8](sb[e])
            if a != b:
                total_scale_mismatch += 1
            if i == 0 and b not in distinct_scales:
                distinct_scales.append(b)

    print(
        "  M=",
        num_rows,
        " quant_mismatch=",
        total_quant_mismatch,
        " scale_mismatch=",
        total_scale_mismatch,
        " normed_mismatch=",
        total_normed_mismatch,
        " nonzero=",
        nonzero_quant,
        " distinct_scales=",
        len(distinct_scales),
    )

    assert_equal(
        total_quant_mismatch, 0, "fused MXFP8 data differs from quantize_mx_amd"
    )
    assert_equal(
        total_scale_mismatch,
        0,
        "fused MXFP8 scales differ from quantize_mx_amd",
    )
    assert_equal(
        total_normed_mismatch,
        0,
        "the epilogue perturbed the bf16 normed output",
    )
    # Non-vacuity: two all-zero buffers compare equal. The magnitude ladder is
    # what makes the scale count meaningful -- see `_gathered_value`.
    assert_true(
        nonzero_quant > ngpus * full_n // 2,
        "most fused output bytes are zero; the comparison is near-vacuous",
    )
    assert_true(
        len(distinct_scales) >= min(8, num_rows * scale_cols),
        String("only ")
        + String(len(distinct_scales))
        + " distinct scales; the input does not exercise the block max",
    )


def main() raises:
    comptime dtype = DType.bfloat16
    var num_devices = DeviceContext.number_of_devices()
    if num_devices < 2:
        print("SKIPPED: needs at least 2 GPUs, found", num_devices)
        return
    if not has_amd_gpu_accelerator():
        print("SKIPPED: the MXFP8 quantize path is CDNA4-only")
        return

    var list_of_ctx = List[DeviceContext]()
    for i in range(num_devices):
        list_of_ctx.append(DeviceContext(device_id=i))
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    if num_devices >= 4:
        print("=== TP4, H=", H, "===")
        # M3 decode: a handful of rows per rank, far below the fuse threshold.
        _run_case[dtype, 4, H](4, list_of_ctx)
        _run_case[dtype, 4, H](16, list_of_ctx)
        # Ragged: ranks past the remainder own zero rows.
        _run_case[dtype, 4, H](2, list_of_ctx)
        # At the fuse threshold's row count.
        _run_case[dtype, 4, H](128, list_of_ctx)

    print("=== TP2, H=", H, "===")
    _run_case[dtype, 2, H](2, list_of_ctx)
    _run_case[dtype, 2, H](17, list_of_ctx)
    _run_case[dtype, 2, H](128, list_of_ctx)

    # The dispatcher's other branch. `threshold` is a runtime argument, so these
    # reuse the row counts above rather than needing a prefill-sized shape.
    print("=== two_launch branch (threshold forced), H=", H, "===")
    _run_case[dtype, 2, H, route_two_launch=True](17, list_of_ctx)
    _run_case[dtype, 2, H, route_two_launch=True](128, list_of_ctx)
    if num_devices >= 4:
        _run_case[dtype, 4, H, route_two_launch=True](16, list_of_ctx)

    print("PASS")
