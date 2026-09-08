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
"""Multi-GPU correctness test for the barrier-free Lamport allreduce (slice 3).

Forces the Lamport path (the production dispatch crossover is "never" for now)
and verifies it is BITWISE-EQUAL to the existing 1-stage allreduce on the same
inputs, sweeping ngpus {2,4,8}, dtypes {bfloat16, float32}, and sizes through
the small regime (one pack, a few packs, ~450 KB, edge offsets).

The 1-stage kernel is the trusted reference: the Lamport kernel changes the
*transport* (push-then-poll instead of barrier-bracketed peer reads) but
accumulates in the same `get_accum_type[dtype]` precision in the same peer
order (`circular_add`), so the reduced result must match bit-for-bit, not just
within tolerance.

Two correctness properties beyond a single reduce:

- generation rotation: each test runs many iterations with an advancing `flag`,
  so the three-buffer rotation and the per-call sentinel clear are exercised;
  every iteration must reproduce the identical reference result.
- producer-sanitize: a dedicated case seeds inputs that actually contain `-0.0`
  lanes. Because `-0.0 == +0.0` and `x + (-0.0) == x`, the reduced result is
  unchanged, but the kernel must `remove_neg_zero` the input so real data is
  never mistaken for the readiness sentinel.

Run multi-GPU only: `bt-b200` then `bt-mi355`, and under `--runs_per_test=20`
for race detection (the slice-2 AMD fenceless caveat is discharged here).
"""

from std.sys import size_of, simd_width_of
from std.itertools import product

from layout import Coord, Idx, TileTensor, coord_to_index_list, row_major
from comm import Signal, MAX_GPUS
from comm.sync import enable_p2p, init_signal_buffer
from comm.lamport import Lamport
from comm.allreduce import (
    AllReduceAlgorithm,
    AllReduceTuningConfig,
    _allreduce_lamport_p2p,
    allreduce,
    elementwise_epilogue_type,
)
from comm.device_query import get_sm_version
from internal_utils import human_readable_size
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.primitives.grid_controls import PDLLevel
from std.memory import bitcast
from std.testing import assert_equal, assert_true
from std.collections import Optional
from std.utils import IndexList, StaticTuple

# Sizes through the small (Lamport) regime. All must be whole 128-bit packs
# (simd_width multiples on every target) and within
# LAMPORT_MAX_SMALL_MESSAGE_BYTES (512 KiB). bf16 simd_width=8, fp32=4 -> use
# multiples of 8 to be a whole pack for both.
comptime test_lengths = (
    8,  # one pack (bf16) / two packs (fp32)
    64,  # a few packs
    8 * 1024,  # 16 KiB bf16 / 32 KiB fp32
    32 * 1024,  # 64 KiB bf16 / 128 KiB fp32
    112 * 1024,  # 224 KiB bf16 / 448 KiB fp32 (~450 KB target case)
)

comptime test_dtypes = (DType.bfloat16, DType.float32)
comptime test_gpu_counts = (2, 4, 8)

# Iterations per case: advances `flag` across many generations to exercise the
# three-buffer rotation and the per-call clear. > 3 so the rotation wraps.
comptime NUM_ITERS = 12

# Back-to-back calls for the unsynced-skew test. Large so inter-rank drift has
# many opportunities to accumulate across the three-generation rotation.
comptime NUM_UNSYNCED_ITERS = 128


def lamport_allreduce_test[
    dtype: DType,
    ngpus: Int,
    *,
    seed_neg_zero: Bool = False,
](list_of_ctx: List[DeviceContext], length: Int) raises:
    """Runs the Lamport path for one (dtype, ngpus, length) and asserts the
    result is bitwise-equal to an independent host-computed reference across
    NUM_ITERS generations.
    """
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"
    comptime simd_width = simd_width_of[dtype, get_gpu_target()]()

    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    # Signal buffers: Signal header + scratch for the Lamport region
    # (3 generations * ngpus slots * max-small-message).
    var scratch_bytes = 3 * ngpus * Lamport.MAX_SMALL_MESSAGE_BYTES
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # The negative-zero bit pattern for this dtype, for the producer-sanitize
    # edge case.
    comptime uint = DType.uint16 if size_of[dtype]() == 2 else DType.uint32
    comptime sign_bit = Scalar[uint](1) << Scalar[uint](
        size_of[dtype]() * 8 - 1
    )
    var neg_zero = bitcast[dtype, 1](SIMD[uint, 1](sign_bit))[0]

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        out_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))

        var host_buffer = alloc[Scalar[dtype]](length)
        host_buffers.append(host_buffer)

        # Rank i seeds value (i + 1). When seed_neg_zero, every other lane is
        # -0.0 -- which IEEE adds as +0.0, so the sum is unchanged, but the
        # kernel must sanitize it or deadlock/misreduce.
        for j in range(length):
            if seed_neg_zero and (j % 2 == 0):
                host_buffer[j] = neg_zero
            else:
                host_buffer[j] = Scalar[dtype](i + 1)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + scratch_bytes
            )
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
        list_of_ctx[i].enqueue_copy(in_dev[i], host_buffers[i])

    comptime InTensorType = TileTensor[
        dtype, type_of(row_major(length)), ImmutAnyOrigin
    ]
    var in_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: TileTensor(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                in_dev[i].unsafe_ptr()
            ),
            row_major(length),
        )
    )

    comptime OutTensorType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]
    var out_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) {
            mut out_dev, imm
        } -> OutTensorType: TileTensor(
            out_dev[i], row_major(length)
        ).as_unsafe_any_origin()
    )

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # --- Reference: independent host calculation. Every rank contributes
    # (i + 1), so the reduced value is the exact integer sum below (bit-exact in
    # bf16/fp32). Seeded -0.0 lanes sanitize to +0.0, so their expected is +0.0.
    comptime expected_sum = ngpus * (ngpus + 1) // 2
    var full_sum = Scalar[dtype](expected_sum)

    # --- Lamport path: force it directly, advancing flag each iteration. ---
    # A minimal forced config: block count from the arch default; the use_*
    # flags are unused by _allreduce_lamport_p2p (it always runs Lamport).
    var lamport_cfg = AllReduceTuningConfig(
        ngpus=ngpus,
        num_bytes=length * size_of[dtype](),
        sm_version=get_sm_version(),
        num_blocks=512,
        algorithm=AllReduceAlgorithm.LAMPORT,
    )

    # One-time init of the signal buffers (zero the counters/state, then set the
    # Lamport region to the -0.0 sentinel). Synchronize so every rank's buffer is
    # initialized before any rank's first push.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Epilogue: store the reduced value to this rank's output buffer.
    var out_capture = StaticTuple[OutTensorType, ngpus]()
    for i in range(ngpus):
        out_capture[i] = TileTensor(out_dev[i], row_major(length))

    for it in range(NUM_ITERS):

        @always_inline
        @__parameter
        @__copy_capture(out_capture)
        def lamport_epilogue[
            input_index: Int,
            _dtype: DType,
            _width: SIMDLength,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            out_capture[input_index].store_linear[
                width=_width, alignment=_alignment
            ](
                rebind[IndexList[1]](coord_to_index_list(coords)),
                rebind[SIMD[dtype, _width]](val),
            )

        comptime for i in range(ngpus):
            _allreduce_lamport_p2p[
                ngpus=ngpus,
                output_lambda=lamport_epilogue[input_index=i, ...],
                pdl_level=PDLLevel(),
            ](
                in_tensors,
                out_tensors[i],
                rank_sigs,
                lamport_cfg,
                list_of_ctx[i],
                i,
            )
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        # Verify bitwise equality vs the host reference for this iteration.
        for i in range(ngpus):
            list_of_ctx[i].enqueue_copy(host_buffers[i], out_dev[i])
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        for i in range(ngpus):
            for j in range(length):
                # Expected: +0.0 on sanitized (-0.0-seeded) lanes, else the
                # integer sum. Compare raw bits so any 1-ulp drift is caught;
                # first normalize a -0.0 output to +0.0 (the only benign
                # divergence, since -0.0 == +0.0 numerically).
                var expected = full_sum
                if seed_neg_zero and j % 2 == 0:
                    expected = Scalar[dtype](0)

                var lam_bits = bitcast[uint, 1](host_buffers[i][j])
                if lam_bits == Scalar[uint](sign_bit):
                    lam_bits = Scalar[uint](0)
                var exp_bits = bitcast[uint, 1](expected)

                try:
                    assert_equal(lam_bits, exp_bits)
                except e:
                    print(
                        "Mismatch at iter",
                        it,
                        "GPU",
                        i,
                        "index",
                        j,
                        "lamport=",
                        host_buffers[i][j],
                        "expected=",
                        expected,
                    )
                    raise e^

    for i in range(ngpus):
        host_buffers[i].free()


def lamport_mixed_size_test[
    dtype: DType,
    ngpus: Int,
](list_of_ctx: List[DeviceContext], max_length: Int) raises:
    """Runs a sequence of DIFFERENT-sized Lamport allreduces sharing ONE signal
    buffer and ONE monotonically-advancing flag, to expose the per-generation
    clear-extent bug.

    The kernel's clear step (e) resets generation `(flag+2)%3` using the
    CURRENT call's `num_packs`. When a generation is written LARGE, then cleared
    by an intervening SMALL call, then re-read LARGE, the tail
    `[small_extent, large_extent)` is never reset to the sentinel and still
    holds the earlier (wrong-generation) data. A reader that polls such a slot
    before the peer's fresh push for this call reads that stale data and
    produces a wrong sum.

    The `[L, S, M]` size pattern reuses generation 0 at LARGE every third call
    after a SMALL clear, so almost the entire LARGE region is stale. Inputs
    vary per call (value depends on `flag`) so a stale read of an earlier call's
    data is a detectable wrong sum; the expected output is the freshly-computed
    per-call sum.

    NOTE: detection is timing-dependent (the reader must poll a stale tail slot
    before the peer's push), so the sequence is repeated and big-L/tiny-S
    maximizes the vulnerable-slot count. Run under `--runs_per_test`. This is
    expected to FAIL on the current clear-this-call's-size kernel and to pass
    once the clear covers each generation's actual written extent.
    """
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    var scratch_bytes = 3 * ngpus * Lamport.MAX_SMALL_MESSAGE_BYTES
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](max_length))
        out_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](max_length))
        host.append(alloc[Scalar[dtype]](max_length))
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + scratch_bytes
            )
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    # Max-shaped output views + a single epilogue. Per-call `result` tensors are
    # length-shaped, so the kernel only generates coords < length, and the
    # epilogue only writes the first `length` slots of the max-shaped view.
    comptime MaxOutType = TileTensor[
        dtype, type_of(row_major(max_length)), MutAnyOrigin
    ]
    var out_capture = StaticTuple[MaxOutType, ngpus]()
    for i in range(ngpus):
        out_capture[i] = TileTensor(out_dev[i], row_major(max_length))

    @always_inline
    @__parameter
    @__copy_capture(out_capture)
    def mixed_epilogue[
        input_index: Int,
        _dtype: DType,
        _width: SIMDLength,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        out_capture[input_index].store_linear[
            width=_width, alignment=_alignment
        ](
            rebind[IndexList[1]](coord_to_index_list(coords)),
            rebind[SIMD[dtype, _width]](val),
        )

    # One-time init of the signal buffers (zero counters/state, sentinel region).
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # [L, S, M]: gen 0 is written LARGE, cleared SMALL one call later, re-read
    # LARGE three calls later -> stale tail. Repeated so gen 0 reuse recurs.
    var sizes = [112 * 1024, 8, 64]
    comptime REPEATS = 4

    var call_idx = 0
    for _rep in range(REPEATS):
        for k in range(len(sizes)):
            var length = sizes[k]

            # Per-call value, distinct across calls (so a stale read of an
            # earlier generation's data is a detectable wrong sum) and small
            # enough that the integer sum is exact in bf16 (< 256).
            for i in range(ngpus):
                var v = Scalar[dtype](
                    Scalar[dtype](i + 1) + Scalar[dtype](call_idx % 5)
                )
                for j in range(max_length):
                    host[i][j] = v
                list_of_ctx[i].enqueue_copy(in_dev[i], host[i])

            comptime InType = TileTensor[
                dtype, type_of(row_major(length)), ImmutAnyOrigin
            ]
            comptime OutType = TileTensor[
                dtype, type_of(row_major(length)), MutAnyOrigin
            ]
            var in_tensors = Array[_, ngpus](
                fill_with=lambda (i: Int) -> InType: TileTensor(
                    rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                        in_dev[i].unsafe_ptr()
                    ),
                    row_major(length),
                )
            )
            var out_tensors = Array[_, ngpus](
                fill_with=lambda (i: Int) -> OutType: TileTensor(
                    rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                        out_dev[i].unsafe_ptr()
                    ),
                    row_major(length),
                )
            )

            var cfg = AllReduceTuningConfig(
                ngpus=ngpus,
                num_bytes=length * size_of[dtype](),
                sm_version=get_sm_version(),
                num_blocks=512,
                algorithm=AllReduceAlgorithm.LAMPORT,
            )

            comptime for i in range(ngpus):
                _allreduce_lamport_p2p[
                    ngpus=ngpus,
                    output_lambda=mixed_epilogue[input_index=i, ...],
                    pdl_level=PDLLevel(),
                ](
                    in_tensors,
                    out_tensors[i],
                    rank_sigs,
                    cfg,
                    list_of_ctx[i],
                    i,
                )
            for i in range(ngpus):
                list_of_ctx[i].synchronize()

            # Expected per-call sum, accumulated in fp32 (the kernel's accum
            # type) and cast to dtype, exactly as the kernel does.
            var expected_accum = Float32(0)
            comptime for i in range(ngpus):
                expected_accum += Float32(i + 1) + Float32(call_idx % 5)
            var expected = Scalar[dtype](expected_accum)

            for i in range(ngpus):
                list_of_ctx[i].enqueue_copy(host[i], out_dev[i])
            for i in range(ngpus):
                list_of_ctx[i].synchronize()

            for i in range(ngpus):
                for j in range(length):
                    try:
                        assert_equal(host[i][j], expected)
                    except e:
                        print(
                            "Mixed-size mismatch at flag",
                            call_idx,
                            "length",
                            length,
                            "GPU",
                            i,
                            "index",
                            j,
                            "got",
                            host[i][j],
                            "expected",
                            expected,
                        )
                        raise e^
            call_idx += 1

    for i in range(ngpus):
        host[i].free()


def lamport_coexist_test[
    dtype: DType,
    ngpus: Int,
](list_of_ctx: List[DeviceContext], small_len: Int, large_len: Int) raises:
    """Interleaves a large barrier-based 2-stage `allreduce` (which uses the
    trailing scratch) between barrier-free Lamport calls on the SAME signal
    buffer, and verifies the Lamport results are unaffected.

    This is the coexistence regression for embedding the Lamport region in
    `Signal`: the region trails the barrier counters and the 2-stage / broadcast
    scratch lives in the bytes after the struct, so the two are disjoint -- a
    2-stage call cannot clobber the Lamport generations or the one-time sentinel
    init. (Before this layout the Lamport region aliased the 2-stage scratch, so
    an interleaved 2-stage corrupted it.) The Lamport flag advances only on
    Lamport calls; gen 0 is re-read at round 3 with two intervening 2-stage
    calls, so a clobber surfaces as a wrong Lamport sum.
    """
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var sin_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var sout_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var lin_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var lout_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var shost = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    # Buffer = sizeof(Signal) (counters + embedded Lamport region) + the 2-stage
    # scratch (ngpus * large message), which trails the struct.
    var scratch_bytes = ngpus * large_len * size_of[dtype]()
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    var lfill = alloc[Scalar[dtype]](large_len)
    for j in range(large_len):
        lfill[j] = Scalar[dtype](1)

    for i in range(ngpus):
        sin_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](small_len))
        sout_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](small_len))
        lin_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](large_len))
        lout_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](large_len))
        shost.append(alloc[Scalar[dtype]](small_len))
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + scratch_bytes
            )
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
        list_of_ctx[i].enqueue_copy(lin_dev[i], lfill)

    comptime LType = TileTensor[
        dtype, type_of(row_major(large_len)), ImmutAnyOrigin
    ]
    comptime LOutType = TileTensor[
        dtype, type_of(row_major(large_len)), MutAnyOrigin
    ]
    var lin_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> LType: LType(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                lin_dev[i].unsafe_ptr()
            ),
            row_major(large_len),
        )
    )
    var lout_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) {mut lout_dev, imm} -> LOutType: TileTensor(
            lout_dev[i], row_major(large_len)
        ).as_unsafe_any_origin()
    )

    comptime SInType = TileTensor[
        dtype, type_of(row_major(small_len)), ImmutAnyOrigin
    ]
    comptime SOutType = TileTensor[
        dtype, type_of(row_major(small_len)), MutAnyOrigin
    ]
    var sin_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> SInType: SInType(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                sin_dev[i].unsafe_ptr()
            ),
            row_major(small_len),
        )
    )
    var sout_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) {mut sout_dev, imm} -> SOutType: TileTensor(
            sout_dev[i], row_major(small_len)
        ).as_unsafe_any_origin()
    )

    var out_capture = StaticTuple[SOutType, ngpus]()
    for i in range(ngpus):
        out_capture[i] = TileTensor(sout_dev[i], row_major(small_len))

    @always_inline
    @__parameter
    @__copy_capture(out_capture)
    def coexist_epilogue[
        input_index: Int,
        _dtype: DType,
        _width: SIMDLength,
        *,
        _alignment: Int,
    ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
        out_capture[input_index].store_linear[
            width=_width, alignment=_alignment
        ](
            rebind[IndexList[1]](coord_to_index_list(coords)),
            rebind[SIMD[dtype, _width]](val),
        )

    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    var call_idx = 0
    comptime ROUNDS = 6
    for _r in range(ROUNDS):
        # (1) Interleaved large 2-stage allreduce -- exercises the trailing
        # scratch on the shared signal buffer.
        comptime for i in range(ngpus):
            allreduce[ngpus=ngpus](
                lin_tensors, lout_tensors[i], rank_sigs, list_of_ctx[i]
            )

        # (2) Small Lamport allreduce on the same buffer; value varies per call.
        for i in range(ngpus):
            var v = Scalar[dtype](
                Scalar[dtype](i + 1) + Scalar[dtype](call_idx % 5)
            )
            for j in range(small_len):
                shost[i][j] = v
            list_of_ctx[i].enqueue_copy(sin_dev[i], shost[i])

        var cfg = AllReduceTuningConfig(
            ngpus=ngpus,
            num_bytes=small_len * size_of[dtype](),
            sm_version=get_sm_version(),
            num_blocks=512,
            algorithm=AllReduceAlgorithm.LAMPORT,
        )
        comptime for i in range(ngpus):
            _allreduce_lamport_p2p[
                ngpus=ngpus,
                output_lambda=coexist_epilogue[input_index=i, ...],
                pdl_level=PDLLevel(),
            ](
                sin_tensors,
                sout_tensors[i],
                rank_sigs,
                cfg,
                list_of_ctx[i],
                i,
            )
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        var expected_accum = Float32(0)
        comptime for i in range(ngpus):
            expected_accum += Float32(i + 1) + Float32(call_idx % 5)
        var expected = Scalar[dtype](expected_accum)

        for i in range(ngpus):
            list_of_ctx[i].enqueue_copy(shost[i], sout_dev[i])
        for i in range(ngpus):
            list_of_ctx[i].synchronize()

        for i in range(ngpus):
            for j in range(small_len):
                try:
                    assert_equal(shost[i][j], expected)
                except e:
                    print(
                        "Coexist mismatch at flag",
                        call_idx,
                        "GPU",
                        i,
                        "index",
                        j,
                        "got",
                        shost[i][j],
                        "expected",
                        expected,
                        "(2-stage interleaving corrupted the Lamport region)",
                    )
                    raise e^
        call_idx += 1

    for i in range(ngpus):
        shost[i].free()
    lfill.free()


def lamport_unsynced_skew_test[
    dtype: DType,
    ngpus: Int,
](list_of_ctx: List[DeviceContext], length: Int) raises:
    """Runs many Lamport allreduces back-to-back with NO cross-rank sync between
    calls, then verifies every call's result in one final pass.

    The other tests `synchronize()` all ranks after each call, which pins the
    inter-rank generation skew to zero -- so the three-generation rotation's
    one-generation slack is never actually exercised. A real model (and the
    benchmark's `iter_custom`) launches collectives back-to-back per rank with
    no cross-rank barrier, letting ranks drift in `flag`. This test reproduces
    that regime: a fast rank's per-call clear (generation `(flag+2)%3`) can race
    a slow rank's read of the generation being reused. If one generation of
    slack is ever insufficient under real drift, a call returns a wrong sum here
    (or the run hangs); a synced test cannot surface it.

    Each call uses a DISTINCT input/output slice of one big per-rank buffer, so
    every result survives to a single final verification pass -- no per-call
    copy-back, hence no per-call sync. Inputs vary per call (value depends on
    the call index), so a stale cross-generation read is a detectable wrong sum.

    Detection is timing-dependent -- run under `--runs_per_test`.
    """
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var total = NUM_UNSYNCED_ITERS * length
    var in_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    var scratch_bytes = 3 * ngpus * Lamport.MAX_SMALL_MESSAGE_BYTES
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for i in range(ngpus):
        in_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](total))
        out_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](total))

        # Fill all iterations' input slices once. Per-call value for rank i is
        # (i+1) + (it % 5): varies across calls so a stale read of another
        # generation's slice is a detectable wrong sum, and < 256 so the integer
        # sum is exact in bf16.
        var h = alloc[Scalar[dtype]](total)
        for it in range(NUM_UNSYNCED_ITERS):
            var v = Scalar[dtype](Scalar[dtype](i + 1) + Scalar[dtype](it % 5))
            for j in range(length):
                h[it * length + j] = v
        host.append(h)
        list_of_ctx[i].enqueue_copy(in_dev[i], h)

        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + scratch_bytes
            )
        )
        rank_sigs[i] = (
            signal_buffers[i]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    # One-time signal-buffer init; sync so every region is ready before any push.
    for i in range(ngpus):
        init_signal_buffer(signal_buffers[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Enqueue every call with NO cross-rank synchronize between them -- this is
    # the point of the test. Each call's tensors point at its own slice.
    for it in range(NUM_UNSYNCED_ITERS):
        var base = it * length
        comptime InType = TileTensor[
            dtype, type_of(row_major(length)), ImmutAnyOrigin
        ]
        comptime OutType = TileTensor[
            dtype, type_of(row_major(length)), MutAnyOrigin
        ]
        var in_tensors = Array[_, ngpus](
            fill_with=lambda (i: Int) -> InType: InType(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    in_dev[i].unsafe_ptr() + base
                ),
                row_major(length),
            )
        )
        var out_capture = StaticTuple[OutType, ngpus]()
        for i in range(ngpus):
            out_capture[i] = OutType(
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    out_dev[i].unsafe_ptr() + base
                ),
                row_major(length),
            )

        @always_inline
        @__parameter
        @__copy_capture(out_capture)
        def skew_epilogue[
            input_index: Int,
            _dtype: DType,
            _width: SIMDLength,
            *,
            _alignment: Int,
        ](coords: Coord, val: SIMD[_dtype, _width]) -> None:
            out_capture[input_index].store_linear[
                width=_width, alignment=_alignment
            ](
                rebind[IndexList[1]](coord_to_index_list(coords)),
                rebind[SIMD[dtype, _width]](val),
            )

        var cfg = AllReduceTuningConfig(
            ngpus=ngpus,
            num_bytes=length * size_of[dtype](),
            sm_version=get_sm_version(),
            num_blocks=512,
            algorithm=AllReduceAlgorithm.LAMPORT,
        )
        comptime for i in range(ngpus):
            _allreduce_lamport_p2p[
                ngpus=ngpus,
                output_lambda=skew_epilogue[input_index=i, ...],
                pdl_level=PDLLevel(),
            ](
                in_tensors,
                out_capture[i],
                rank_sigs,
                cfg,
                list_of_ctx[i],
                i,
            )
        # NO synchronize() here -- ranks are free to drift in `flag`.

    # Single barrier after ALL calls are enqueued, then verify every slice.
    for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host[i], out_dev[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    for it in range(NUM_UNSYNCED_ITERS):
        var expected_accum = Float32(0)
        comptime for i in range(ngpus):
            expected_accum += Float32(i + 1) + Float32(it % 5)
        var expected = Scalar[dtype](expected_accum)

        for i in range(ngpus):
            for j in range(length):
                var got = host[i][it * length + j]
                try:
                    assert_equal(got, expected)
                except e:
                    print(
                        "Unsynced-skew mismatch at iter",
                        it,
                        "GPU",
                        i,
                        "index",
                        j,
                        "got",
                        got,
                        "expected",
                        expected,
                        "(generation rotation skew exceeded its slack)",
                    )
                    raise e^

    for i in range(ngpus):
        host[i].free()


def _case_str[
    dtype: DType, seed_neg_zero: Bool
](ngpus: Int, length: Int) -> String:
    var tag = "-negzero" if seed_neg_zero else ""
    return String(
        "====lamport-allreduce-",
        dtype,
        "-",
        ngpus,
        tag,
        "-",
        human_readable_size(size_of[dtype]() * length),
    )


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    # Emit the observed device count so the log records exactly which ngpus
    # cases run vs silently skip (the shared executor exposes 2, sometimes 4,
    # GPUs; ngpus=8 cases skip via the `< num_gpus` guard below).
    print(
        "====lamport-allreduce device_count=",
        DeviceContext.number_of_devices(),
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    comptime for gpu_idx, dtype_idx, length_idx in product(
        range(len(test_gpu_counts)),
        range(len(test_dtypes)),
        range(len(test_lengths)),
    ):
        comptime num_gpus = rebind[Int](test_gpu_counts[gpu_idx])
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        comptime length = rebind[Int](test_lengths[length_idx])

        if DeviceContext.number_of_devices() < num_gpus:
            continue

        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        print(_case_str[dtype, False](num_gpus, length))
        lamport_allreduce_test[dtype=dtype, ngpus=num_gpus](ctx, length)

    # Producer-sanitize edge case: inputs containing -0.0 lanes, one mid-size
    # case per dtype on 2 GPUs (the property is rank-count independent).
    comptime for dtype_idx in range(len(test_dtypes)):
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        comptime length = 8 * 1024
        if DeviceContext.number_of_devices() < 2:
            continue
        var ctx = List[DeviceContext]()
        for i in range(2):
            ctx.append(DeviceContext(device_id=i))
        print(_case_str[dtype, True](2, length))
        lamport_allreduce_test[dtype=dtype, ngpus=2, seed_neg_zero=True](
            ctx, length
        )

    # Mixed-size sequence sharing one signal buffer + one advancing flag, to
    # expose the per-generation clear-extent bug. 2 GPUs is sufficient (the bug
    # is rank-count independent). Race-dependent -- run under `--runs_per_test`.
    comptime for dtype_idx in range(len(test_dtypes)):
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        if DeviceContext.number_of_devices() < 2:
            continue
        var ctx = List[DeviceContext]()
        for i in range(2):
            ctx.append(DeviceContext(device_id=i))
        print("====lamport-allreduce-mixed-", dtype, "-2")
        lamport_mixed_size_test[dtype=dtype, ngpus=2](ctx, 112 * 1024)

    # Coexistence: a large 2-stage allreduce interleaved between Lamport calls on
    # the same signal buffer must not corrupt the Lamport results -- proves the
    # embedded Lamport region is disjoint from the trailing 2-stage scratch.
    comptime for dtype_idx in range(len(test_dtypes)):
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        if DeviceContext.number_of_devices() < 2:
            continue
        var ctx = List[DeviceContext]()
        for i in range(2):
            ctx.append(DeviceContext(device_id=i))
        print("====lamport-allreduce-coexist-", dtype, "-2")
        lamport_coexist_test[dtype=dtype, ngpus=2](
            ctx, 8 * 1024, 4 * 1024 * 1024
        )

    # Unsynced back-to-back skew stress: many Lamport calls with NO cross-rank
    # sync between them, exercising the generation-rotation slack under real
    # inter-rank drift (the other tests pin skew to zero). Race-dependent -- run
    # under `--runs_per_test`. Swept over available GPU counts (more ranks ->
    # more drift sources).
    comptime for gpu_idx, dtype_idx in product(
        range(len(test_gpu_counts)), range(len(test_dtypes))
    ):
        comptime num_gpus = rebind[Int](test_gpu_counts[gpu_idx])
        comptime dtype = rebind[DType](test_dtypes[dtype_idx])
        if DeviceContext.number_of_devices() < num_gpus:
            continue
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))
        print("====lamport-allreduce-unsynced-skew-", dtype, "-", num_gpus)
        lamport_unsynced_skew_test[dtype=dtype, ngpus=num_gpus](ctx, 8 * 1024)
