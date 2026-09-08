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

"""Correctness test for the fused reduce-scatter + RMSNorm kernel (bf16).

Runs `reducescatter_rmsnorm` on TP4 shards over several M (incl. the M=6
non-divisible ragged edge) and gates each rank's `[rank_units, H]` outputs
against host references (full-H divisor, multiply-before-cast, bf16 last):

  * TIGHT — vs the bf16-rounded ref (sum rounded to bf16 pre-norm, as the kernel
    does): only mean-square associativity wobbles the norm. Gate: frac(ULP>1) <=
    1% and max_ulp <= 4.
  * LOOSE — vs the f32-sum ref. Gate: max_ulp <= 4. The ~24% 1-ULP mismatch is
    the expected f32-vs-bf16-sum gap (deliberate mid-way rounding); not gated.
  * `sum_out` — bit-identical to a standalone `reducescatter` shard on AMD
    non-multimem (tolerance on NVIDIA), given simd==8, f32 accum, and peer
    rotation `(my_rank+i)%ngpus == circular_add`.
"""

from std.sys import (
    has_amd_gpu_accelerator,
    simd_width_of,
    size_of,
)

from std.math import align_down, rsqrt
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from std.utils.index import Index
from std.utils.numerics import get_accum_type
from std.testing import assert_raises, assert_true

from layout import Coord, TileTensor, row_major

from comm import Signal, MAX_GPUS, group_start, group_end
from comm.reducescatter_rmsnorm import (
    RS_NORM_FUSE_THRESHOLD,
    _dispatch_rs_norm,
    reducescatter_rmsnorm,
)
from comm.reducescatter import reducescatter, ReduceScatterConfig
from max.gpu.primitives.grid_controls import PDLLevel
from nn.normalization import rms_norm_gpu
from comm.sync import (
    circular_add,
    enable_p2p,
    init_signal_buffer,
    is_p2p_enabled,
)


def _run_case[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    use_dispatch: Bool = False,
    pdl_level: PDLLevel = PDLLevel(),
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    dispatch_threshold: Int = RS_NORM_FUSE_THRESHOLD,
) raises:
    """Run the fused kernel (or `_dispatch_rs_norm`) and gate both oracles.

    With `use_dispatch`, drive the auto dispatch with a caller `two_launch`
    closure and overridable `dispatch_threshold` (the collective-split-brain
    guard): at a non-divisible `num_rows` straddling the threshold, a
    rank-variant gate would split ranks across fused/`two_launch` on the same
    `rank_sigs` and DEADLOCK; the rank-invariant gate (`rank_units(0)`) makes all
    ranks agree. A regression hangs this case."""
    comptime simd_width = simd_width_of[in_dtype, target=get_gpu_target()]()

    # Preconditions for bit-identical `sum_out` on AMD non-multimem: peer
    # rotation `(my_rank+i)%ngpus == circular_add` and f32 accum (asserted so a
    # future change fails here, not as silent drift); simd==8 is AMD-only.
    comptime assert (
        get_accum_type[in_dtype]() == .float32
    ), "sum_out bit-identity assumes an f32 accumulator"
    comptime for _r in range(ngpus):
        comptime for _i in range(ngpus):
            comptime assert (
                circular_add[ngpus](_r, _i) == (_r + _i) % ngpus
            ), "kernel peer rotation must equal RS circular_add"
    comptime if has_amd_gpu_accelerator():
        comptime assert (
            simd_width == 8
        ), "sum_out bit-identity assumes simd=8 on AMD"

    var config = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var length = num_rows * num_cols
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](0.0)

    # Per-GPU inputs, per-device gamma, signals, and three output shards.
    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var rs_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # Shared gamma values (replicated per device so each rank reads locally).
    var gamma_host = List(
        length=num_cols,
        fill_with=lambda (c: Int) -> Scalar[in_dtype]: (
            Float64(c + num_cols) / Float64(num_cols)
        ).cast[in_dtype](),
    )

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))

        # Distinct positive per-GPU data: the peer sum reaches ~1010 (bf16
        # granule 4), so the mid-way bf16 store is genuinely lossy (the ~24%
        # loose headroom). Positive keeps bf16-ULP-via-bits clean.
        var h = List[Scalar[in_dtype]](length=length, fill=Scalar[in_dtype](0))
        for j in range(length):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        host_bufs.append(h^)

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        # >= 1 row of storage so 0-row ranks (e.g. ranks 1-3 at M=1) aren't
        # zero-size; the true row count drives the launch and the readback.
        var alloc_i = config.rank_num_elements(i)
        if alloc_i < 1:
            alloc_i = 1
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        sum_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        rs_ref.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))

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

    # Tensor views: full [rows, cols] inputs and [rank_units, cols] output
    # shards (true row count; the storage is >= 1 row).
    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]
    var in_bufs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                in_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )
    )

    # --- Run the fused kernel (or the auto dispatch) on every rank. ---
    group_start()
    for i in range(ngpus):
        var normed_view = OutShardType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        var sum_view = OutShardType(
            sum_shard[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        var gamma_view = GammaType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
        comptime if use_dispatch:
            # Production two-launch fallback (== the op's closure): standalone
            # reduce-scatter into `sum_view`, then `rms_norm_gpu` into
            # `normed_view`. Writing both outputs lets it hit the same oracles.
            @__parameter
            @always_inline
            def two_launch() raises:
                reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
                    in_bufs, sum_view, rank_sigs, list_of_ctx[i]
                )
                _rms_norm_shard[in_dtype, num_cols](
                    config.rank_units(i),
                    sum_shard[i],
                    normed[i],
                    gamma_dev[i],
                    epsilon,
                    weight_offset,
                    list_of_ctx[i],
                )

            _dispatch_rs_norm[two_launch=two_launch, pdl_level=pdl_level](
                in_bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
                threshold=dispatch_threshold,
            )
        else:
            reducescatter_rmsnorm[pdl_level=pdl_level](
                in_bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                rank_sigs,
                list_of_ctx[i],
            )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Standalone reduce-scatter into rs_ref, for the sum_out compare. ---
    # Re-init the signal buffers so this collective gets clean barrier state.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var rs_view = OutShardType(
            rs_ref[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(i), num_cols))),
        )
        reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
            in_bufs, rs_view, rank_sigs, list_of_ctx[i]
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Host oracle + comparison over all owned rows of all ranks. ---
    var woff = weight_offset.cast[.float32]()
    var total_elems = 0
    var max_ulp_bf = 0  # fused vs bf16 ref (tight — kernel norms the bf16 sum)
    var gt1_ulp_bf = 0
    var max_ulp_f = 0  # fused vs f32 ref (loose — the wider f32-sum path)
    var mismatch_f = 0  # fused-vs-f32 exact-mismatch (headroom, reported)
    var sum_mismatch = 0  # sum_out vs standalone RS shard
    var sum_max_ulp = 0
    var accum = List[Float32](length=num_cols, fill=Float32(0))

    for i in range(ngpus):
        var local_rows = config.rank_units(i)
        if local_rows == 0:
            continue
        var start = config.rank_unit_start(i)
        var n = local_rows * num_cols

        var normed_h = List[Scalar[in_dtype]](
            length=n, fill=Scalar[in_dtype](0)
        )
        var sum_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var rs_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        list_of_ctx[i].enqueue_copy(normed_h, normed[i])
        list_of_ctx[i].enqueue_copy(sum_h, sum_shard[i])
        list_of_ctx[i].enqueue_copy(rs_h, rs_ref[i])
        list_of_ctx[i].synchronize()
        total_elems += n

        for rr in range(local_rows):
            var grow = start + rr
            var base = grow * num_cols

            # Pass 1: f32 peer sum + both mean-square accumulations.
            var m2_bf = Float32(0)
            var m2_f = Float32(0)
            for c in range(num_cols):
                var s = Float32(0)
                for g in range(ngpus):
                    s += host_bufs[g][base + c].cast[.float32]()
                accum[c] = s
                var xb = s.cast[.bfloat16]().cast[.float32]()
                m2_bf += xb * xb
                m2_f += s * s

            var nf_bf = rsqrt(m2_bf / Float32(num_cols) + epsilon)
            var nf_f = rsqrt(m2_f / Float32(num_cols) + epsilon)

            # Pass 2: normalize, fold gamma in f32, cast bf16 last, compare.
            for c in range(num_cols):
                var s = accum[c]
                var xb = s.cast[.bfloat16]().cast[.float32]()
                var g_f = gamma_host[c].cast[.float32]() + woff
                var ref_bf16 = ((xb * nf_bf) * g_f).cast[.bfloat16]()
                var ref_f16 = ((s * nf_f) * g_f).cast[.bfloat16]()
                var gpu_normed = normed_h[rr * num_cols + c].cast[
                    DType.bfloat16
                ]()

                # Outputs positive, so the uint16 bit pattern is
                # magnitude-monotonic and |bits_a - bits_b| is the bf16 ULP.
                var gpu_bits = Int(gpu_normed.to_bits())
                var ulp_f = abs(gpu_bits - Int(ref_f16.to_bits()))
                var ulp_bf = abs(gpu_bits - Int(ref_bf16.to_bits()))
                if ulp_bf > max_ulp_bf:
                    max_ulp_bf = ulp_bf
                if ulp_bf > 1:
                    gt1_ulp_bf += 1
                if ulp_f > max_ulp_f:
                    max_ulp_f = ulp_f
                if gpu_normed.cast[.float32]() != ref_f16.cast[.float32]():
                    mismatch_f += 1

                # sum_out vs standalone RS shard (bit-identical on AMD).
                var sum_bits = Int(
                    sum_h[rr * num_cols + c].cast[.bfloat16]().to_bits()
                )
                var rs_bits = Int(
                    rs_h[rr * num_cols + c].cast[.bfloat16]().to_bits()
                )
                var sum_ulp = abs(sum_bits - rs_bits)
                if sum_ulp != 0:
                    sum_mismatch += 1
                if sum_ulp > sum_max_ulp:
                    sum_max_ulp = sum_ulp

        _ = normed_h^
        _ = sum_h^
        _ = rs_h^

    var frac_gt1_bf = Float32(gt1_ulp_bf) / Float32(total_elems)
    var rate_f = Float32(mismatch_f) / Float32(total_elems)

    comptime mode_tag = "[dispatch straddle] " if use_dispatch else ""
    print(
        String(
            "  ",
            mode_tag,
            "M=",
            num_rows,
            ": TIGHT(fused vs bf16 ref) frac>1ULP=",
            frac_gt1_bf * 100.0,
            "% max_ulp=",
            max_ulp_bf,
            " | LOOSE(vs f32 ref) exact-mismatch=",
            rate_f * 100.0,
            "% (headroom) max_ulp=",
            max_ulp_f,
            " | sum_out mismatches=",
            sum_mismatch,
            " max_ulp=",
            sum_max_ulp,
        )
    )

    # Tight gate: kernel norms the bf16-rounded sum, matching the bf16 ref
    # within the block-reduce-vs-serial wobble (<=1 ULP); a wrong-eps /
    # wrong-divisor / gamma-after-cast bug is >> 4 ULP.
    if frac_gt1_bf > 0.01 or max_ulp_bf > 4:
        raise Error(
            String(
                "tight bf16-ref gate failed at M=",
                num_rows,
                ": frac>1ULP=",
                frac_gt1_bf * 100.0,
                "% max_ulp=",
                max_ulp_bf,
            )
        )
    # Loose gate: bounded bf16-ULP magnitude vs the f32-sum ref (~24% 1-ULP
    # divergence is the expected mid-way rounding).
    if max_ulp_f > 4:
        raise Error(
            String(
                "loose f32-ref gate failed at M=",
                num_rows,
                ": max_ulp=",
                max_ulp_f,
            )
        )
    # sum_out: bit-identical on AMD non-multimem; 1-ULP tolerance on NVIDIA
    # (RS may take the multimem path, undefined reduction order).
    comptime if has_amd_gpu_accelerator():
        if sum_mismatch != 0:
            raise Error(
                String(
                    "sum_out not bit-identical to standalone RS on AMD at M=",
                    num_rows,
                    ": mismatches=",
                    sum_mismatch,
                    " max_ulp=",
                    sum_max_ulp,
                )
            )
    else:
        if sum_max_ulp > 1:
            raise Error(
                String(
                    "sum_out exceeds 1-ULP tolerance vs standalone RS at M=",
                    num_rows,
                    ": max_ulp=",
                    sum_max_ulp,
                )
            )

    _ = in_dev^
    _ = host_bufs^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_shard^
    _ = rs_ref^
    _ = signal_buffers^
    _ = accum^


def _rms_norm_shard[
    in_dtype: DType,
    num_cols: Int,
](
    rows: Int,
    src: DeviceBuffer[in_dtype],
    dst: DeviceBuffer[in_dtype],
    gamma: DeviceBuffer[in_dtype],
    epsilon: Float32,
    weight_offset: Scalar[in_dtype],
    ctx: DeviceContext,
) raises:
    """Standalone `rms_norm_gpu` on one `[rows, num_cols]` shard, M3-config
    (multiply_before_cast=True). This is the RMSNorm half of the production
    two-launch path (`reducescatter` -> `rms_norm_gpu`); reads `src`, writes
    `dst`."""
    comptime ShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    var src_view = ShardType(
        rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](src.unsafe_ptr()),
        row_major(Coord(Index(rows, num_cols))),
    )
    var dst_view = ShardType(
        rebind[MutPointer[Scalar[in_dtype], MutAnyOrigin]](dst.unsafe_ptr()),
        row_major(Coord(Index(rows, num_cols))),
    )
    var gamma_view = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ](
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            gamma.unsafe_ptr()
        ),
        row_major(Coord(Index(num_cols))),
    )

    @always_inline
    @__copy_capture(src_view)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[in_dtype, width]:
        return src_view.raw_load[width=width](src_view.layout(coords))

    @always_inline
    @__copy_capture(dst_view)
    @__parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[in_dtype, width]) -> None:
        dst_view.raw_store[width=width, alignment=alignment](
            dst_view.layout(coords), val
        )

    rms_norm_gpu[2, input_fn, output_fn, multiply_before_cast=True](
        Coord(Index(rows, num_cols)), gamma_view, epsilon, weight_offset, ctx
    )


def _run_prod_oracle_case[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    use_dispatch: Bool = False,
    group_size: Int = ngpus,
    pdl_level: PDLLevel = PDLLevel(),
](num_rows: Int, list_of_ctx: List[DeviceContext]) raises -> Int:
    """Compare the fused `normed_out` to the ACTUAL M3 production norm; return the
    fused-vs-production exact-mismatch count.

    Production is the fused op's two-launch fallback: standalone `reducescatter`
    then the real `rms_norm_gpu` KERNEL at M3 config (weight_offset=1.0, mbc=True).
    Load-bearing because it runs the real kernel, not a host reduction, so a 1-ULP
    block-reduce-geometry gap between the fused `block.sum` and `rms_norm_gpu`
    shows up as a real bf16 mismatch; sweeping M locates the crossover M* below
    which fused == production.

    When `use_dispatch`, drive the auto dispatch (real `RS_NORM_FUSE_THRESHOLD`)
    with the op's `two_launch` closure: fused below M*, two-launch above, so
    `normed_out` must be bit-identical to production at EVERY M (the routing
    invariant, gated). This is the exact dispatch `distributed.mojo` uses.

    Also asserts `sum_out` == the standalone RS shard at every M (the
    residual-stream contract).

    With `group_size < ngpus` the devices split into `ngpus // group_size`
    independent contiguous groups (production TP4xDP2xEP8), so every collective
    here is group-local: `local_rank` is the rank WITHIN the group, the peer
    arrays hold only the group's devices, and the rows are binned across
    `group_size`. A global-rank or full-world-slicing bug lands on the wrong
    peers and fails the bit-identity gates below."""
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide the device count"
    # Mirrors the handler: a full-world collective keeps barrier domain 0; a
    # subgroup gets its own counter bank so both can share `Signal` buffers.
    comptime domain_id = 0 if group_size == ngpus else group_size

    var config = ReduceScatterConfig[in_dtype, group_size](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var length = num_rows * num_cols
    var epsilon = Float32(1e-6)
    # M3 post-attention layernorm is Gemma-style: weight_offset=1.0, mbc=True.
    var weight_offset = Scalar[in_dtype](1.0)

    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var rs_ref = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var prod = List[DeviceBuffer[in_dtype]](capacity=ngpus)
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
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        var h = List[Scalar[in_dtype]](length=length, fill=Scalar[in_dtype](0))
        for j in range(length):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        # Keep the host buffer alive until after the launch (the copy is async).
        host_bufs.append(h^)

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        var alloc_i = config.rank_num_elements(i % group_size)
        if alloc_i < 1:
            alloc_i = 1
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        sum_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        rs_ref.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        prod.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))

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

    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]

    # --- Fused kernel directly, or the op's dispatch (auto-route at the real
    # threshold) with the production `two_launch` fallback. ---
    group_start()
    for i in range(ngpus):
        var local = i % group_size
        var base = (i // group_size) * group_size
        # Group-local peer/signal arrays: ranks 0..group_size-1 are this
        # device's group, exactly what the handler hands the kernel.
        var bufs = Array[_, group_size](
            fill_with=lambda (k: Int) -> InTensorType: InTensorType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    in_dev[base + k].unsafe_ptr()
                ),
                row_major(Coord(Index(num_rows, num_cols))),
            )
        )
        var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for k in range(group_size):
            sigs[k] = rank_sigs[base + k]

        var normed_view = OutShardType(
            normed[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(local), num_cols))),
        )
        var sum_view = OutShardType(
            sum_shard[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(local), num_cols))),
        )
        var gamma_view = GammaType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                gamma_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_cols))),
        )
        comptime if use_dispatch:
            # Mirror the graph op's fallback (distributed.mojo): standalone
            # reduce-scatter into `sum_view`, then `rms_norm_gpu` into
            # `normed_view`.
            @__parameter
            @always_inline
            def two_launch() raises:
                reducescatter[
                    dtype=in_dtype,
                    ngpus=group_size,
                    axis=0,
                    domain_id=domain_id,
                ](bufs, sum_view, sigs, list_of_ctx[i], local_rank=local)
                _rms_norm_shard[in_dtype, num_cols](
                    config.rank_units(local),
                    sum_shard[i],
                    normed[i],
                    gamma_dev[i],
                    epsilon,
                    weight_offset,
                    list_of_ctx[i],
                )

            _dispatch_rs_norm[
                two_launch=two_launch,
                domain_id=domain_id,
                pdl_level=pdl_level,
            ](
                bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                sigs,
                list_of_ctx[i],
                local_rank=local,
            )
        else:
            reducescatter_rmsnorm[domain_id=domain_id, pdl_level=pdl_level](
                bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                sigs,
                list_of_ctx[i],
                local_rank=local,
            )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Production two-launch: standalone reduce-scatter -> rms_norm_gpu. ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var local = i % group_size
        var base = (i // group_size) * group_size
        var bufs = Array[_, group_size](
            fill_with=lambda (k: Int) -> InTensorType: InTensorType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    in_dev[base + k].unsafe_ptr()
                ),
                row_major(Coord(Index(num_rows, num_cols))),
            )
        )
        var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for k in range(group_size):
            sigs[k] = rank_sigs[base + k]

        var rs_view = OutShardType(
            rs_ref[i].unsafe_ptr().as_unsafe_any_origin(),
            row_major(Coord(Index(config.rank_units(local), num_cols))),
        )
        reducescatter[
            dtype=in_dtype, ngpus=group_size, axis=0, domain_id=domain_id
        ](bufs, rs_view, sigs, list_of_ctx[i], local_rank=local)
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    for i in range(ngpus):
        var local_rows = config.rank_units(i % group_size)
        if local_rows == 0:
            continue
        _rms_norm_shard[in_dtype, num_cols](
            local_rows,
            rs_ref[i],
            prod[i],
            gamma_dev[i],
            epsilon,
            weight_offset,
            list_of_ctx[i],
        )
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Compare fused normed vs production normed (bit-for-bit). ---
    var total_elems = 0
    var normed_mismatch = 0  # fused vs production rms_norm_gpu output
    var normed_max_ulp = 0
    var sum_mismatch = 0  # fused sum_out vs standalone RS shard
    var sum_max_ulp = 0

    for i in range(ngpus):
        var local_rows = config.rank_units(i % group_size)
        if local_rows == 0:
            continue
        var n = local_rows * num_cols
        var normed_h = List[Scalar[in_dtype]](
            length=n, fill=Scalar[in_dtype](0)
        )
        var prod_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var sum_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var rs_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        list_of_ctx[i].enqueue_copy(normed_h, normed[i])
        list_of_ctx[i].enqueue_copy(prod_h, prod[i])
        list_of_ctx[i].enqueue_copy(sum_h, sum_shard[i])
        list_of_ctx[i].enqueue_copy(rs_h, rs_ref[i])
        list_of_ctx[i].synchronize()
        total_elems += n

        for e in range(n):
            var f_bits = Int(normed_h[e].cast[.bfloat16]().to_bits())
            var p_bits = Int(prod_h[e].cast[.bfloat16]().to_bits())
            if f_bits != p_bits:
                normed_mismatch += 1
            var ulp = abs(f_bits - p_bits)
            if ulp > normed_max_ulp:
                normed_max_ulp = ulp

            var s_bits = Int(sum_h[e].cast[.bfloat16]().to_bits())
            var r_bits = Int(rs_h[e].cast[.bfloat16]().to_bits())
            if s_bits != r_bits:
                sum_mismatch += 1
            var s_ulp = abs(s_bits - r_bits)
            if s_ulp > sum_max_ulp:
                sum_max_ulp = s_ulp

        _ = normed_h^
        _ = prod_h^
        _ = sum_h^
        _ = rs_h^

    var rate = Float32(normed_mismatch) / Float32(total_elems) * 100.0
    comptime mode_tag = "dispatched-op-vs" if use_dispatch else "fused-vs"
    print(
        String(
            "  M=",
            num_rows,
            " (G=",
            group_size,
            " rank0 units=",
            config.rank_units(0),
            "): ",
            mode_tag,
            "-PRODUCTION mismatch=",
            normed_mismatch,
            "/",
            total_elems,
            " (",
            rate,
            "%) max_ulp=",
            normed_max_ulp,
            " | sum_out mismatch=",
            sum_mismatch,
            (
                "  <-- BIT-IDENTICAL" if normed_mismatch
                == 0 else "  <-- diverges"
            ),
        )
    )

    # sum_out must be bit-identical to the standalone RS shard on AMD at every M
    # (residual stream is plain reduce-scatter on either branch).
    comptime if has_amd_gpu_accelerator():
        if sum_mismatch != 0:
            raise Error(
                String(
                    "sum_out not bit-identical to standalone RS at M=",
                    num_rows,
                    ": mismatches=",
                    sum_mismatch,
                    " max_ulp=",
                    sum_max_ulp,
                )
            )

    # Routing invariant: the dispatched op must be bit-identical to production at
    # EVERY M (fused below the threshold, two-launch above); a wrong sense would
    # fuse a diverging shape and fail here. AMD-scoped (gfx950-calibrated).
    comptime if use_dispatch and has_amd_gpu_accelerator():
        if normed_mismatch != 0:
            raise Error(
                String(
                    "dispatched op NOT bit-identical to production at M=",
                    num_rows,
                    ": mismatches=",
                    normed_mismatch,
                    " max_ulp=",
                    normed_max_ulp,
                    " (dispatch routed a diverging shape to the fused kernel)",
                )
            )

    _ = in_dev^
    _ = host_bufs^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_shard^
    _ = rs_ref^
    _ = prod^
    _ = signal_buffers^

    return normed_mismatch


@always_inline
def _bf16_ulp_key(bits: Int) -> Int:
    """Map bf16's sign-magnitude bits to a monotonic key.

    `|key(a) - key(b)|` is then the ULP distance for SIGNED data, which the raw
    bit difference is not. `_run_case`'s outputs are all positive; these are
    not.
    """
    return (0x8000 - (bits & 0x7FFF)) if bits >= 0x8000 else (bits + 0x8000)


def _run_residual_case[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
    group_size: Int = ngpus,
    wide_magnitudes: Bool = False,
](num_rows: Int, list_of_ctx: List[DeviceContext]) raises:
    """Gate the folded residual against an INDEPENDENT host oracle.

    The fold replaces the caller's leader-only pre-add with a per-rank add of
    the rank's own shard of the replicated `xs`. The two are value-equal only
    because `xs` is bit-identical across the group, which this case reproduces.

    They are NOT bit-identical: the leader rounds `xs + attn_0` before the
    collective, the fold rounds the peer sum first. Neither ordering dominates,
    so the leader-add arm is bounded rather than required to match.

    Gate 1 (load-bearing, shares no code with the kernel): a host oracle
    recomputes `sum_out` in the kernel's peer order and requires BIT-EXACTness;
    f32 addition is not associative, so a reordering fails it.
    Gate 2: the leader-add spelling, bounded to a small ULP shift.
    Gate 3 (non-vacuity): the residual must actually change the result, else a
    dropped residual would satisfy gates 1 and 2 vacuously at xs == 0.
    Gate 4: `normed_out` vs a host RMSNorm of gate 1's sum -- the only gate that
    sees the residual reach the norm rather than just `sum_out`.
    Gate 5: the leader-add arm's `normed_out`, bounded by gate 2's sum gap.
    """
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide the device count"
    comptime domain_id = 0 if group_size == ngpus else group_size

    var config = ReduceScatterConfig[in_dtype, group_size](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var length = num_rows * num_cols
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](1.0)

    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var pre_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var res_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed_b = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_b = List[DeviceBuffer[in_dtype]](capacity=ngpus)
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

    # `wide_magnitudes` spans 2^-6..2^14 to broaden exponent coverage. It does
    # NOT gate the fold's position: `sum_out` is bf16, so an f32 reordering
    # moves bits below the round (a fold-first control stayed at 0 mismatches).
    var res_host = List[Scalar[in_dtype]](
        length=length, fill=Scalar[in_dtype](0)
    )
    for j in range(length):
        var scale = Float64(1 << (j % 21)) / 64.0 if wide_magnitudes else 1.0
        var mag = Scalar[in_dtype]((Float64(j % 97) * 0.5 + 1.0) * scale)
        res_host[j] = -mag if (j % 3 == 0) else mag

    # Both outlive the loop: `enqueue_copy` is an ASYNC H2D off a raw host
    # pointer, so dropping a source before a synchronize drains it is a UAF.
    var attn_hosts = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var pre_hosts = List[List[Scalar[in_dtype]]](capacity=ngpus)
    for i in range(ngpus):
        var h = List[Scalar[in_dtype]](length=length, fill=Scalar[in_dtype](0))
        for j in range(length):
            # Offset the exponent per device so peers straddle each other's
            # mantissa window; a same-magnitude peer set would cancel the point.
            var scale = (
                Float64(1 << ((j + 7 * i) % 21))
                / 64.0 if wide_magnitudes else 1.0
            )
            var v = Scalar[in_dtype](
                (Float64(i + 1) + Float64(j % 251)) * scale
            )
            h[j] = -v if (j % 5 == 0) else v
        var pre = List[Scalar[in_dtype]](
            length=length, fill=Scalar[in_dtype](0)
        )
        # Leader-add arm: only group-local rank 0 pre-adds, exactly as the
        # caller's `i % tp_degree == 0` list comprehension does.
        for j in range(length):
            pre[j] = (h[j] + res_host[j]) if (i % group_size == 0) else h[j]

        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        pre_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        list_of_ctx[i].enqueue_copy(pre_dev[i], pre)
        res_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        list_of_ctx[i].enqueue_copy(res_dev[i], res_host)
        attn_hosts.append(h^)
        pre_hosts.append(pre^)

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        var alloc_i = config.rank_num_elements(i % group_size)
        if alloc_i < 1:
            alloc_i = 1
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        sum_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i)
        )
        normed_b.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))
        sum_b.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](alloc_i))

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

    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]

    # --- Arm A: the fold. inputs = bare attention partials, residual = xs. ---
    group_start()
    for i in range(ngpus):
        var local = i % group_size
        var base = align_down(i, group_size)
        var bufs = Array[_, group_size](
            fill_with=lambda (k: Int) -> InTensorType: InTensorType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    in_dev[base + k].unsafe_ptr()
                ),
                row_major(Coord(Index(num_rows, num_cols))),
            )
        )
        var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for k in range(group_size):
            sigs[k] = rank_sigs[base + k]
        var res_view = InTensorType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                res_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )
        reducescatter_rmsnorm[has_residual=True, domain_id=domain_id](
            bufs,
            OutShardType(
                normed[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(config.rank_units(local), num_cols))),
            ),
            OutShardType(
                sum_shard[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(config.rank_units(local), num_cols))),
            ),
            GammaType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    gamma_dev[i].unsafe_ptr()
                ),
                row_major(Coord(Index(num_cols))),
            ),
            epsilon,
            weight_offset,
            sigs,
            list_of_ctx[i],
            local,
            residual=res_view,
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Arm B: the leader-add spelling this change replaces. ---
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    group_start()
    for i in range(ngpus):
        var local = i % group_size
        var base = (i // group_size) * group_size
        var bufs = Array[_, group_size](
            fill_with=lambda (k: Int) -> InTensorType: InTensorType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    pre_dev[base + k].unsafe_ptr()
                ),
                row_major(Coord(Index(num_rows, num_cols))),
            )
        )
        var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
            uninitialized=True
        )
        for k in range(group_size):
            sigs[k] = rank_sigs[base + k]
        reducescatter_rmsnorm[domain_id=domain_id](
            bufs,
            OutShardType(
                normed_b[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(config.rank_units(local), num_cols))),
            ),
            OutShardType(
                sum_b[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(config.rank_units(local), num_cols))),
            ),
            GammaType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    gamma_dev[i].unsafe_ptr()
                ),
                row_major(Coord(Index(num_cols))),
            ),
            epsilon,
            weight_offset,
            sigs,
            list_of_ctx[i],
            local,
        )
    group_end()
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Gates. ---
    var oracle_mismatch = 0
    var leader_mismatch = 0
    var leader_max_ulp = 0
    var changed_by_residual = 0
    var nonzero = 0
    var total = 0
    # Gate 4: `normed_out` vs a host RMSNorm of the same oracle sum. Every other
    # gate reads `sum_out` only, leaving the norm unchecked at ULP level.
    var normed_max_ulp = 0
    var normed_gt1_ulp = 0
    # Gate 5 (non-load-bearing): the leader-add arm's norm tracks arm A's, whose
    # only difference is the removed pre-add rounding in the sum.
    var normed_arm_max_ulp = 0
    var woff = weight_offset.cast[.float32]()
    # One row's oracle sum, so the norm reference can take a second pass.
    var want_row = List[Float32](length=num_cols, fill=Float32(0))

    for i in range(ngpus):
        var local = i % group_size
        var base = (i // group_size) * group_size
        var local_rows = config.rank_units(local)
        if local_rows == 0:
            continue
        var n = local_rows * num_cols
        var sum_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var sum_b_h = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
        var normed_h = List[Scalar[in_dtype]](
            length=n, fill=Scalar[in_dtype](0)
        )
        var normed_b_h = List[Scalar[in_dtype]](
            length=n, fill=Scalar[in_dtype](0)
        )
        list_of_ctx[i].enqueue_copy(sum_h, sum_shard[i])
        list_of_ctx[i].enqueue_copy(sum_b_h, sum_b[i])
        list_of_ctx[i].enqueue_copy(normed_h, normed[i])
        list_of_ctx[i].enqueue_copy(normed_b_h, normed_b[i])
        list_of_ctx[i].synchronize()
        total += n

        var my_start = config.rank_unit_start(local)
        for lr in range(local_rows):
            var row = my_start + lr

            # Pass 1: the oracle sum, the gates reading it, and its mean-square.
            var m2 = Float32(0)
            for c in range(num_cols):
                var e = lr * num_cols + c
                var g = row * num_cols + c

                # Kernel's round-robin peer order: peer sum rounded to the
                # output dtype FIRST, then residual in f32, rounded once more.
                var acc = Float32(0)
                for k in range(group_size):
                    var peer = (local + k) % group_size
                    acc += attn_hosts[base + peer][g].cast[.float32]()
                var without_res = acc.cast[in_dtype]()
                var want = (
                    without_res.cast[.float32]() + res_host[g].cast[.float32]()
                ).cast[in_dtype]()
                want_row[c] = want.cast[.float32]()
                m2 += want_row[c] * want_row[c]
                if (
                    sum_h[e].cast[.bfloat16]().to_bits()
                    != want.cast[.bfloat16]().to_bits()
                ):
                    oracle_mismatch += 1
                if (
                    without_res.cast[.bfloat16]().to_bits()
                    != want.cast[.bfloat16]().to_bits()
                ):
                    changed_by_residual += 1
                if want != Scalar[in_dtype](0):
                    nonzero += 1

                var a_bits = Int(sum_h[e].cast[.bfloat16]().to_bits())
                var b_bits = Int(sum_b_h[e].cast[.bfloat16]().to_bits())
                if a_bits != b_bits:
                    leader_mismatch += 1
                var ulp = abs(a_bits - b_bits)
                if ulp > leader_max_ulp:
                    leader_max_ulp = ulp

            # Pass 2: RMSNorm it the way the kernel does -- full-H divisor,
            # epsilon outside the division, gamma folded in f32, bf16 last.
            var norm_factor = rsqrt(m2 / Float32(num_cols) + epsilon)
            for c in range(num_cols):
                var e = lr * num_cols + c
                var g_f = gamma_host[c].cast[.float32]() + woff
                var ref_key = _bf16_ulp_key(
                    Int(
                        ((want_row[c] * norm_factor) * g_f)
                        .cast[.bfloat16]()
                        .to_bits()
                    )
                )
                var got_key = _bf16_ulp_key(
                    Int(normed_h[e].cast[.bfloat16]().to_bits())
                )
                var arm_b_key = _bf16_ulp_key(
                    Int(normed_b_h[e].cast[.bfloat16]().to_bits())
                )
                var n_ulp = abs(got_key - ref_key)
                if n_ulp > normed_max_ulp:
                    normed_max_ulp = n_ulp
                if n_ulp > 1:
                    normed_gt1_ulp += 1
                var arm_ulp = abs(got_key - arm_b_key)
                if arm_ulp > normed_arm_max_ulp:
                    normed_arm_max_ulp = arm_ulp

        _ = sum_h^
        _ = sum_b_h^
        _ = normed_h^
        _ = normed_b_h^

    var normed_frac_gt1 = Float32(normed_gt1_ulp) / Float32(total)
    print(
        String(
            "  M=",
            num_rows,
            " (G=",
            group_size,
            "): residual-fold vs HOST ORACLE mismatch=",
            oracle_mismatch,
            "/",
            total,
            " | vs leader-add mismatch=",
            leader_mismatch,
            " max_ulp=",
            leader_max_ulp,
            " | residual-sensitive elems=",
            changed_by_residual,
            " | normed vs host frac>1ULP=",
            normed_frac_gt1 * 100.0,
            "% max_ulp=",
            normed_max_ulp,
            " (arm-vs-arm max_ulp=",
            normed_arm_max_ulp,
            ")",
        )
    )

    if oracle_mismatch != 0:
        raise Error(
            String(
                "residual fold does not match the host oracle at M=",
                num_rows,
                ": mismatches=",
                oracle_mismatch,
                "/",
                total,
            )
        )
    # Non-vacuity: a dropped residual reads 0 here, so a floor is needed. Only
    # 10% because a large peer sum can absorb it below the bf16 round (43%).
    if changed_by_residual * 10 < total:
        raise Error(
            String(
                "residual barely changes the result (",
                changed_by_residual,
                "/",
                total,
                "); the oracle gate would be near-vacuous",
            )
        )
    if nonzero * 2 < total:
        raise Error(String("output is mostly zero (", nonzero, "/", total, ")"))
    # One bf16 rounding apart, so a bigger gap means the fold landed somewhere
    # other than the leader's contribution. Ungated under `wide_magnitudes`:
    # cancellation across 2^20 makes that rounding ~120 ULP of a near-zero sum.
    comptime if not wide_magnitudes:
        if leader_max_ulp > 2:
            raise Error(
                String(
                    "residual fold diverges from the leader-add spelling by ",
                    leader_max_ulp,
                    " ULP at M=",
                    num_rows,
                    " (expected <= 2 from the removed pre-add rounding)",
                )
            )

    # The kernel norms the value gate 1 just proved bit-exact, so only
    # mean-square associativity separates them: `_run_case`'s tight bound holds.
    if normed_frac_gt1 > 0.01 or normed_max_ulp > 4:
        raise Error(
            String(
                "normed_out gate failed at M=",
                num_rows,
                " (G=",
                group_size,
                "): frac>1ULP=",
                normed_frac_gt1 * 100.0,
                "% max_ulp=",
                normed_max_ulp,
            )
        )
    # Bound arm B's norm against the sum gap that causes it, not a constant --
    # that holds under `wide_magnitudes` too, so it needs no exemption there.
    if normed_arm_max_ulp > leader_max_ulp + 4:
        raise Error(
            String(
                "normed_out diverges from the leader-add arm by ",
                normed_arm_max_ulp,
                " ULP at M=",
                num_rows,
                ", more than the ",
                leader_max_ulp,
                " ULP its sum differs by",
            )
        )

    _ = in_dev^
    _ = pre_dev^
    _ = res_dev^
    _ = attn_hosts^
    _ = pre_hosts^
    _ = res_host^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_shard^
    _ = normed_b^
    _ = sum_b^
    _ = signal_buffers^


def _run_interleaved_barrier_case[
    in_dtype: DType,
    ngpus: Int,
    group_size: Int,
    num_cols: Int,
](num_rows: Int, rounds: Int, list_of_ctx: List[DeviceContext]) raises:
    """Deadlock gate: a GROUPED fused RS+norm interleaved with a FULL-WORLD
    reduce-scatter on the SAME `Signal` buffers, looped `rounds` times with a
    single up-front barrier init.

    This is the production hazard under TP4xDP2xEP8: a TP-group collective and an
    EP/full-world collective share each device's `Signal`. `_multi_gpu_barrier`
    keys its counter slots by in-block thread index, not global rank, so with both
    in barrier domain 0 the two histories alias the same slots -- the generation
    counters desync and a later full-world barrier spins forever, or returns
    early on a stale flag and reduces unready peer data (`sync.mojo`
    NUM_BARRIER_DOMAINS). Both failure modes are gated: this case must TERMINATE
    and both collectives' sums must still be exact afterwards.
    """
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide the device count"
    comptime assert (
        group_size < ngpus
    ), "interleaving is only meaningful for a subgroup collective"
    comptime domain_id = group_size

    var grp_cfg = ReduceScatterConfig[in_dtype, group_size](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var world_cfg = ReduceScatterConfig[in_dtype, ngpus](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var length = num_rows * num_cols
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](1.0)

    var in_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var host_bufs = List[List[Scalar[in_dtype]]](capacity=ngpus)
    var gamma_dev = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var normed = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var sum_shard = List[DeviceBuffer[in_dtype]](capacity=ngpus)
    var world_out = List[DeviceBuffer[in_dtype]](capacity=ngpus)
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
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        var h = List[Scalar[in_dtype]](length=length, fill=Scalar[in_dtype](0))
        for j in range(length):
            h[j] = Scalar[in_dtype](i + 1) + Scalar[in_dtype](j % 251)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)
        host_bufs.append(h^)

        gamma_dev.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](num_cols)
        )
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

        var g_alloc = grp_cfg.rank_num_elements(i % group_size)
        if g_alloc < 1:
            g_alloc = 1
        normed.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](g_alloc))
        sum_shard.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](g_alloc)
        )

        var w_alloc = world_cfg.rank_num_elements(i)
        if w_alloc < 1:
            w_alloc = 1
        world_out.append(
            list_of_ctx[i].enqueue_create_buffer[in_dtype](w_alloc)
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

    # ONE init for the whole run: continuously advancing shared counters are
    # exactly what desyncs when both collectives sit in domain 0.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]

    var world_bufs = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                in_dev[i].unsafe_ptr()
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )
    )

    for _round in range(rounds):
        group_start()
        for i in range(ngpus):
            var local = i % group_size
            var base = (i // group_size) * group_size
            var bufs = Array[_, group_size](
                fill_with=lambda (k: Int) -> InTensorType: InTensorType(
                    rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                        in_dev[base + k].unsafe_ptr()
                    ),
                    row_major(Coord(Index(num_rows, num_cols))),
                )
            )
            var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
                uninitialized=True
            )
            for k in range(group_size):
                sigs[k] = rank_sigs[base + k]

            var normed_view = OutShardType(
                normed[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(grp_cfg.rank_units(local), num_cols))),
            )
            var sum_view = OutShardType(
                sum_shard[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(grp_cfg.rank_units(local), num_cols))),
            )
            var gamma_view = GammaType(
                rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                    gamma_dev[i].unsafe_ptr()
                ),
                row_major(Coord(Index(num_cols))),
            )
            reducescatter_rmsnorm[domain_id=domain_id](
                bufs,
                normed_view,
                sum_view,
                gamma_view,
                epsilon,
                weight_offset,
                sigs,
                list_of_ctx[i],
                local,
            )
        group_end()

        # Full-world collective on the SAME buffers, barrier domain 0.
        group_start()
        for i in range(ngpus):
            var w_view = OutShardType(
                world_out[i].unsafe_ptr().as_unsafe_any_origin(),
                row_major(Coord(Index(world_cfg.rank_units(i), num_cols))),
            )
            reducescatter[dtype=in_dtype, ngpus=ngpus, axis=0](
                world_bufs, w_view, rank_sigs, list_of_ctx[i]
            )
        group_end()

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Both collectives must still be exact: a barrier that returned early
    # reduces peer data that was not ready yet. Integer inputs sum exactly in
    # f32, so the reference is a plain accumulate-then-round.
    var grp_bad = 0
    var world_bad = 0
    for i in range(ngpus):
        var local = i % group_size
        var base = (i // group_size) * group_size

        var g_rows = grp_cfg.rank_units(local)
        if g_rows > 0:
            var n = g_rows * num_cols
            var got = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
            list_of_ctx[i].enqueue_copy(got, sum_shard[i])
            list_of_ctx[i].synchronize()
            var start = grp_cfg.rank_unit_start(local)
            for rr in range(g_rows):
                for c in range(num_cols):
                    var s = Float32(0)
                    for k in range(group_size):
                        s += host_bufs[base + k][
                            (start + rr) * num_cols + c
                        ].cast[.float32]()
                    if got[rr * num_cols + c] != s.cast[in_dtype]():
                        grp_bad += 1
            _ = got^

        var w_rows = world_cfg.rank_units(i)
        if w_rows > 0:
            var n = w_rows * num_cols
            var got = List[Scalar[in_dtype]](length=n, fill=Scalar[in_dtype](0))
            list_of_ctx[i].enqueue_copy(got, world_out[i])
            list_of_ctx[i].synchronize()
            var start = world_cfg.rank_unit_start(i)
            for rr in range(w_rows):
                for c in range(num_cols):
                    var s = Float32(0)
                    for k in range(ngpus):
                        s += host_bufs[k][(start + rr) * num_cols + c].cast[
                            DType.float32
                        ]()
                    if got[rr * num_cols + c] != s.cast[in_dtype]():
                        world_bad += 1
            _ = got^

    print(
        String(
            "  interleaved barrier (",
            ngpus,
            " GPUs, ",
            ngpus // group_size,
            "x TP",
            group_size,
            ", ",
            rounds,
            " rounds, M=",
            num_rows,
            "): completed; grouped-sum errors=",
            grp_bad,
            " full-world-sum errors=",
            world_bad,
        )
    )

    if grp_bad != 0 or world_bad != 0:
        raise Error(
            String(
                (
                    "interleaved grouped/full-world collectives corrupted"
                    " results (grouped errors="
                ),
                grp_bad,
                ", full-world errors=",
                world_bad,
                "): the two barrier domains are aliasing.",
            )
        )

    _ = in_dev^
    _ = host_bufs^
    _ = gamma_dev^
    _ = gamma_host^
    _ = normed^
    _ = sum_shard^
    _ = world_out^
    _ = signal_buffers^


def _run_rank_validation_case[
    in_dtype: DType,
    group_size: Int,
    num_cols: Int,
](list_of_ctx: List[DeviceContext]) raises:
    """Gate the public API's rank / shard-shape validation.

    A wrong-but-in-range rank is a SELF-CONSISTENT out-of-bounds write -- the
    values are right, they just land past the shard -- so every numeric oracle
    in this file reports zero mismatches. It has to be caught at the host
    boundary instead.

    Both mistakes are reachable now that the rank is caller-supplied: a GLOBAL
    device id (out of range for the group-local peer/signal arrays), and a rank
    that disagrees with the shard the caller allocated. `num_rows` is
    deliberately ragged (`group_size + 1`), so rank 0 owns one row more than the
    others and the second mistake overruns by exactly one row of `num_cols`."""
    var num_rows = group_size + 1
    var config = ReduceScatterConfig[in_dtype, group_size](
        axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
    )
    var epsilon = Float32(1e-6)
    var weight_offset = Scalar[in_dtype](1.0)
    var length = num_rows * num_cols
    # Allocate for the largest shard so a deliberately mis-sized VIEW below
    # still points at real memory (nothing launches -- the raises fire first).
    var max_shard = config.rank_num_elements(0)

    var in_dev = List[DeviceBuffer[in_dtype]](capacity=group_size)
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=group_size)
    var sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    for i in range(group_size):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[in_dtype](length))
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](size_of[Signal]())
        )
        sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
    var gamma_dev = list_of_ctx[0].enqueue_create_buffer[in_dtype](num_cols)
    var normed = list_of_ctx[0].enqueue_create_buffer[in_dtype](max_shard)
    var sum_shard = list_of_ctx[0].enqueue_create_buffer[in_dtype](max_shard)
    list_of_ctx[0].synchronize()

    comptime InTensorType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0, num_cols)))), ImmutAnyOrigin
    ]
    comptime OutShardType = TileTensor[
        mut=True,
        in_dtype,
        type_of(row_major(Coord(Index(0, num_cols)))),
        MutAnyOrigin,
    ]
    comptime GammaType = TileTensor[
        in_dtype, type_of(row_major(Coord(Index(0)))), ImmutAnyOrigin
    ]

    var bufs = Array[_, group_size](
        fill_with=lambda (k: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
                in_dev[k].unsafe_ptr()
            ),
            row_major(Coord(Index(num_rows, num_cols))),
        )
    )
    var gamma_view = GammaType(
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            gamma_dev.unsafe_ptr()
        ),
        row_major(Coord(Index(num_cols))),
    )

    comptime domain_id = group_size
    var rank0_rows = config.rank_units(0)
    var rank1_rows = config.rank_units(1)
    assert_true(
        rank0_rows != rank1_rows,
        (
            "ragged M must give rank 0 a different shard height than rank 1,"
            " else the shape case below cannot discriminate"
        ),
    )

    var normed_ok = OutShardType(
        normed.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Index(rank0_rows, num_cols))),
    )
    var sum_ok = OutShardType(
        sum_shard.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Index(rank0_rows, num_cols))),
    )
    # Sized for rank 1, passed with rank 0: one row short.
    var normed_short = OutShardType(
        normed.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Index(rank1_rows, num_cols))),
    )
    var sum_short = OutShardType(
        sum_shard.unsafe_ptr().as_unsafe_any_origin(),
        row_major(Coord(Index(rank1_rows, num_cols))),
    )

    # 1. A global device id on the trailing group: out of range for arrays that
    #    only hold `group_size` entries.
    with assert_raises(contains="local_rank"):
        reducescatter_rmsnorm[domain_id=domain_id](
            bufs,
            normed_ok,
            sum_ok,
            gamma_view,
            epsilon,
            weight_offset,
            sigs,
            list_of_ctx[0],
            local_rank=group_size,
        )

    # 2. In-range rank, shard allocated for a different rank: the one-row
    #    self-consistent overrun.
    with assert_raises(contains="normed_out"):
        reducescatter_rmsnorm[domain_id=domain_id](
            bufs,
            normed_short,
            sum_ok,
            gamma_view,
            epsilon,
            weight_offset,
            sigs,
            list_of_ctx[0],
            local_rank=0,
        )

    with assert_raises(contains="sum_out"):
        reducescatter_rmsnorm[domain_id=domain_id](
            bufs,
            normed_ok,
            sum_short,
            gamma_view,
            epsilon,
            weight_offset,
            sigs,
            list_of_ctx[0],
            local_rank=0,
        )

    # 3. A shard-shaped residual on BOTH arms: indexed by global row either
    #    way, so rejection must precede the arm choice. The marker proves it.
    var res_shard = InTensorType(
        rebind[ImmPointer[Scalar[in_dtype], ImmutAnyOrigin]](
            in_dev[0].unsafe_ptr()
        ),
        row_major(Coord(Index(rank0_rows, num_cols))),
    )

    @__parameter
    @always_inline
    def two_launch_marker() raises:
        raise Error("two_launch ran with an unvalidated residual")

    # threshold=0 forces the two-launch arm; the default fuses at this M.
    for threshold in [0, RS_NORM_FUSE_THRESHOLD]:
        with assert_raises(contains="residual holds"):
            _dispatch_rs_norm[
                two_launch=two_launch_marker,
                has_residual=True,
                domain_id=domain_id,
            ](
                bufs,
                normed_ok,
                sum_ok,
                gamma_view,
                epsilon,
                weight_offset,
                sigs,
                list_of_ctx[0],
                threshold=threshold,
                local_rank=0,
                residual=res_shard,
            )

    print("rank / shard-shape validation passed.")


def _run_grouped_suite[
    in_dtype: DType,
    ngpus: Int,
    group_size: Int,
    num_cols: Int,
]() raises:
    """Grouped-TP suite: `ngpus // group_size` independent groups of
    `group_size`, i.e. what TP4xDP2 hands the op on 8 GPUs.

    The full-world cases cannot fail on any of the grouping logic (local rank ==
    device id, one group, whole-world bins), so these are the only cases that
    cover it."""
    var list_of_ctx = List[DeviceContext]()
    for i in range(ngpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print(
        "\n=== grouped: ",
        ngpus,
        " GPUs as ",
        ngpus // group_size,
        "x TP",
        group_size,
        " (H=",
        num_cols,
        ") ===",
    )

    # Every case routes through the dispatch, so `normed_out` bit-identity vs
    # the grouped two-launch path is GATED at each M, not merely printed: fused
    # below the threshold, two-launch above, plus a ragged M.
    for num_rows in [group_size, 8, group_size + group_size // 2, 512, 1024]:
        _ = _run_prod_oracle_case[
            in_dtype, ngpus, num_cols, use_dispatch=True, group_size=group_size
        ](num_rows, list_of_ctx)

    # One direct call into `reducescatter_rmsnorm` (bypassing the dispatcher) so
    # the fused entry point itself is covered; gates `sum_out` bit-identity.
    _ = _run_prod_oracle_case[in_dtype, ngpus, num_cols, group_size=group_size](
        8, list_of_ctx
    )

    # Narrower H than the suite on purpose: the deadlock gate needs barrier
    # traffic, not the H=6144 the fuse-threshold oracle is calibrated at, and a
    # narrow row keeps the per-round allocation small.
    comptime BARRIER_GATE_COLS = 1024
    # 2 rounds suffice: in domain 0 the desync is off by one in both the counter
    # value and its parity, so round 1's first full-world barrier already hangs.
    _run_interleaved_barrier_case[
        in_dtype, ngpus, group_size, BARRIER_GATE_COLS
    ](32, 2, list_of_ctx)

    _run_rank_validation_case[in_dtype, group_size, num_cols](list_of_ctx)

    # Residual fold at the grouped topology, where the leader-add spelling it
    # replaces is per-GROUP (`i % group_size == 0`), not per-world.
    print("  residual fold:")
    for num_rows in [group_size, 8, group_size + group_size // 2, 512]:
        _run_residual_case[in_dtype, ngpus, num_cols, group_size=group_size](
            num_rows, list_of_ctx
        )
    # Wide exponent range: the fold must hold across magnitudes even where the
    # f32 peer sum drops low bits.
    _run_residual_case[
        in_dtype,
        ngpus,
        num_cols,
        group_size=group_size,
        wide_magnitudes=True,
    ](8, list_of_ctx)

    print("grouped ", ngpus // group_size, "x TP", group_size, " passed.")
    _ = list_of_ctx^


def _run_suite[
    in_dtype: DType,
    ngpus: Int,
    num_cols: Int,
]() raises:
    """Run the full correctness suite on `ngpus` GPUs (caller ensures >= ngpus
    are present and P2P is on). Driven at TP2 and TP4 from `main` so a 2-GPU CI
    lane exercises the kernel instead of skipping to a false green."""
    var list_of_ctx = List[DeviceContext]()
    for i in range(ngpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print("fused reduce-scatter + RMSNorm: TP", ngpus, "H=", num_cols)
    # Decode-shape M and a prefill tile (2048), plus the non-divisible ragged
    # edge `ngpus + ngpus//2` (rank 0 takes the remainder row: 2/2/1/1 at TP4,
    # 2/1 at TP2), exercising the extra-row / 0-row shard bookkeeping.
    for num_rows in [1, 8, 16, 32, 2048, ngpus + ngpus // 2]:
        _run_case[in_dtype, ngpus, num_cols](num_rows, list_of_ctx)

    # Residual fold FULL-WORLD (so barrier domain 0). The grouped suite needs
    # >= 4 GPUs; without this the fold is unrun on the common 2-GPU lane.
    print("\nresidual fold (full-world, TP", ngpus, "):")
    for num_rows in [ngpus, 8, ngpus + ngpus // 2, 512]:
        _run_residual_case[in_dtype, ngpus, num_cols](num_rows, list_of_ctx)
    _run_residual_case[in_dtype, ngpus, num_cols, wide_magnitudes=True](
        8, list_of_ctx
    )

    # --- Production bit-identity sweep (fused vs standalone RS + rms_norm_gpu,
    # M3 config wo=1.0 mbc=True): locates the crossover M* that calibrates
    # RS_NORM_FUSE_THRESHOLD (fuse only where bit-identical to production). ---
    print(
        "\nfused-vs-production bit-identity sweep (wo=1.0, mbc=True, TP",
        ngpus,
        "H=",
        num_cols,
        "):",
    )
    var largest_bit_identical_m = 0
    var smallest_diverging_m = 0
    for num_rows in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        var mismatch = _run_prod_oracle_case[in_dtype, ngpus, num_cols](
            num_rows, list_of_ctx
        )
        if mismatch == 0:
            if num_rows > largest_bit_identical_m:
                largest_bit_identical_m = num_rows
        elif smallest_diverging_m == 0:
            smallest_diverging_m = num_rows

        # Calibration gate: any M the threshold would fuse MUST be bit-identical
        # to production. AMD-scoped (gfx950-calibrated); NVIDIA's `rms_norm_gpu`
        # has a different reduction geometry and may cross over elsewhere
        # (reported, not gated).
        comptime if has_amd_gpu_accelerator():
            var cfg = ReduceScatterConfig[in_dtype, ngpus](
                axis_size=num_rows, unit_numel=num_cols, threads_per_gpu=0
            )
            var per_rank_bytes = (
                cfg.rank_units(0) * num_cols * size_of[in_dtype]()
            )
            if per_rank_bytes <= RS_NORM_FUSE_THRESHOLD and mismatch != 0:
                raise Error(
                    String(
                        "calibration FAILED: M=",
                        num_rows,
                        " (",
                        per_rank_bytes,
                        " B/rank) would fuse (threshold=",
                        RS_NORM_FUSE_THRESHOLD,
                        ") but is NOT bit-identical to production (mismatch=",
                        mismatch,
                        "). Lower RS_NORM_FUSE_THRESHOLD below this shape.",
                    )
                )

    print(
        "\ncrossover: largest bit-identical M =",
        largest_bit_identical_m,
        "; smallest diverging M =",
        smallest_diverging_m,
    )

    # Dispatch routing invariant: the auto dispatch (with the op's `two_launch`
    # fallback) must be bit-identical to production at EVERY M — fused below M*,
    # two-launch above. M spans the M=512 crossover.
    print("\ndispatched-op-vs-production (auto-route at real threshold):")
    for num_rows in [8, 512, 1024, 4096]:
        _ = _run_prod_oracle_case[in_dtype, ngpus, num_cols, use_dispatch=True](
            num_rows, list_of_ctx
        )

    # Collective-split-brain guard: a non-divisible M (`ngpus + ngpus//2`) with
    # a threshold (18432 B = 1.5 rows) straddling the 1-row and 2-row shards. A
    # rank-variant gate would split ranks across fused/two_launch on the same
    # rank_sigs and DEADLOCK; the rank-invariant gate (rank_units(0)=2 rows)
    # makes every rank pick two_launch. A regression hangs here.
    _run_case[in_dtype, ngpus, num_cols, use_dispatch=True](
        ngpus + ngpus // 2, list_of_ctx, dispatch_threshold=18432
    )

    # PDL changes only launch overlap, never the math, so every gate above must
    # still hold with it enabled -- this is what `distributed.mojo` launches. A
    # trigger misplaced before the end barrier would show up here as a `sum_out`
    # mismatch.
    print("\nPDL-on (attribute set) vs production, fused + dispatched:")
    # M=1 and the ragged edge use `_run_case`'s oracles because the production
    # oracle sweep starts at M=8, and M=1 leaves ranks 1..n-1 with 0-row shards.
    for num_rows in [1, 8, ngpus + ngpus // 2]:
        _run_case[in_dtype, ngpus, num_cols, pdl_level=PDLLevel.ON](
            num_rows, list_of_ctx
        )
    for num_rows in [8, 512]:
        _ = _run_prod_oracle_case[
            in_dtype, ngpus, num_cols, pdl_level=PDLLevel.ON
        ](num_rows, list_of_ctx)
    for num_rows in [8, 1024]:
        _ = _run_prod_oracle_case[
            in_dtype,
            ngpus,
            num_cols,
            use_dispatch=True,
            pdl_level=PDLLevel.ON,
        ](num_rows, list_of_ctx)

    print("TP", ngpus, "suite passed.")
    _ = list_of_ctx^


def main() raises:
    comptime in_dtype = DType.bfloat16
    comptime num_cols = 6144

    var num_devices = DeviceContext.number_of_devices()
    if num_devices < 2:
        print(
            "Need at least 2 GPUs but only found",
            num_devices,
            "- skipping.",
        )
        return

    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")
    if not is_p2p_enabled():
        print("P2P not enabled, skipping test.")
        return

    # TP2 is the common multi-GPU CI lane, so the suite runs below 4 GPUs; TP4
    # adds M3's real topology and H=6144 threshold calibration. The crossover is
    # per-rank-row driven, so the calibration gate holds at both.
    _run_suite[in_dtype, 2, num_cols]()
    if num_devices >= 4:
        _run_suite[in_dtype, 4, num_cols]()

    # 4 GPUs as 2xTP2 is the smallest topology exercising the rank remap,
    # group-local slicing and per-group ragged binning; 8 as 2xTP4 is M3's
    # production shape. Announce the skip loudly -- a 2-GPU lane covers none of
    # the grouping logic and a green run there would read as coverage.
    if num_devices >= 4:
        _run_grouped_suite[in_dtype, 4, 2, num_cols]()
    else:
        print(
            "\nSKIPPED grouped-TP coverage (needs >= 4 GPUs, found",
            num_devices,
            (
                "): local-rank remap, group-local slicing, per-group binning"
                " and the barrier-domain deadlock gate are NOT exercised in"
                " this run."
            ),
        )
    if num_devices >= 8:
        _run_grouped_suite[in_dtype, 8, 4, num_cols]()
    else:
        print(
            (
                "\nSKIPPED grouped 2xTP4 (M3's production TP4xDP2 topology,"
                " needs 8 GPUs, found"
            ),
            num_devices,
            ").",
        )

    print("All fused reduce-scatter + RMSNorm tests passed!")
