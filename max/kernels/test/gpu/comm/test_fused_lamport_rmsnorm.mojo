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
"""Correctness test for the fused high-perf Lamport allreduce + RMSNorm.

2 GPUs, bf16. Sweeps several (M, K) shapes and checks the fused kernel
(`lamport_allreduce_rmsnorm`) against the **unfused** composition the
fused kernel is replacing: a generic `allreduce` (which routes to the
standalone Lamport AR for small bf16 messages with P2P) writes a bf16 AR
sum, then RMSNorm is applied to that bf16 intermediate. The reference
captures the bf16-storage precision of the unfused chain rather than a
host-fp32 idealization, so a passing test means "fusing does not change
the answer beyond the bf16 round-trip noise of the unfused path".

The K range covers the fused kernel's shape constraints
(`cols % atomic_width == 0` and `cols/atomic_width <= BLOCK_SIZE`; for
bf16 that's `cols % 8 == 0` and `cols <= 8192`).
"""

from std.sys import size_of
from std.random import random_float64
from std.memory import bitcast

from max.gpu.host import DeviceBuffer, DeviceContext

from layout import Coord, TileTensor, row_major

from comm import Signal, MAX_GPUS, group_start, group_end
from comm.sync import init_signal_buffer
from comm.sync import enable_p2p
from comm.allreduce import allreduce
from comm.allreduce_lamport_rmsnorm import lamport_allreduce_rmsnorm
from nn.normalization import rms_norm_gpu

from internal_utils import assert_almost_equal
from std.testing import assert_true
from std.utils.index import Index

comptime dtype = DType.bfloat16
comptime EPS = Float32(1e-6)

# Iterations per case: > 3 so the three-generation Lamport rotation wraps at
# least once (the bare-Lamport AR test uses 12; we match it).
comptime NUM_ITERS = 12

# Back-to-back calls for the unsynced-skew test. Large so inter-rank drift
# has many opportunities to accumulate across the three-generation rotation.
comptime NUM_UNSYNCED_ITERS = 32


def _case_str(M: Int, K: Int, *, tag: String = "") -> String:
    var suffix = ("-" + tag) if tag != "" else ""
    return String("====fused-lamport-ar-rmsnorm-bf16-", M, "x", K, suffix)


def _neg_zero_bits() -> Scalar[dtype]:
    """Returns the bf16 bit pattern for -0.0 (sign bit set, mantissa+exp=0).

    Lamport uses fp32 `-0.0` (0x80000000) as the readiness sentinel; bf16
    `-0.0` is the top 16 bits of that (0x8000). The fused kernel calls
    `remove_neg_zero` on the local input pack to sanitize real-data `-0.0`
    lanes before pushing -- if that path is broken, real data is mistaken
    for the sentinel and the kernel deadlocks or returns wrong sums.
    """
    return bitcast[dtype, 1](UInt16(0x8000))[0]


def _run_rms_norm_unfused(
    in_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    out_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    gamma_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    M: Int,
    K: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
    """Per-device RMSNorm reading [M, K] from `in_ptr` and writing [M, K] to
    `out_ptr`. Wraps `rms_norm_gpu` with the minimal input/output lambdas it
    requires. Takes raw pointers (not DeviceBuffer) so callers can pass slice
    offsets into a larger buffer."""
    var shape = Index(M, K)
    var in_view = TileTensor(in_ptr, row_major(Coord(shape)))
    var out_view = TileTensor(out_ptr, row_major(Coord(shape)))
    var gamma_view = TileTensor(gamma_ptr, row_major(Coord(Index(K))))

    @always_inline
    @__copy_capture(in_view)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = in_view.layout(coords)
        return in_view.raw_load[width=width](idx)

    @always_inline
    @__copy_capture(out_view)
    @__parameter
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        var idx = out_view.layout(coords)
        out_view.raw_store[width=width, alignment=alignment](idx, val)

    rms_norm_gpu[
        2,  # rank
        input_fn,
        output_fn,
        multiply_before_cast=True,
    ](Coord(shape), gamma_view, epsilon, Scalar[dtype](0), ctx)


def rmsnorm_test[
    ngpus: Int, *, seed_neg_zero: Bool = False
](
    list_of_ctx: List[DeviceContext],
    M: Int,
    K: Int,
    iters: Int = NUM_ITERS,
) raises:
    """Compare `lamport_allreduce_rmsnorm` (fused) against an on-device
    unfused chain (`allreduce` -> bf16 sum -> `rms_norm_gpu`) across many
    calls. The unfused reference is computed once from a single AR pass via
    on-device `allreduce` + `rms_norm_gpu`; the fused kernel is iterated
    `iters` times so the three-generation rotation wraps multiple times.

    `seed_neg_zero=True` seeds every other input lane with `-0.0` (the bf16
    bit pattern matching Lamport's fp32 sentinel under truncation). Since
    `x + (-0.0) == x`, the sum is unchanged, but the kernel must
    `remove_neg_zero` the input pack before pushing -- if not, real data
    is mistaken for the sentinel and the kernel deadlocks or misreduces."""
    print(
        "==== fused Lamport AR+RMSNorm  ngpus=",
        ngpus,
        " M=",
        M,
        " K=",
        K,
        " iters=",
        iters,
    )

    var act_size = M * K

    # ---- Per-device buffers. ----
    var act_in = List[DeviceBuffer[dtype]](capacity=ngpus)
    var gamma = List[DeviceBuffer[dtype]](capacity=ngpus)
    var ar_out = List[DeviceBuffer[dtype]](capacity=ngpus)
    var unfused_out = List[DeviceBuffer[dtype]](capacity=ngpus)
    var fused_out = List[DeviceBuffer[dtype]](capacity=ngpus)

    # Separate signal buffers for the unfused AR vs the fused AR+RMSNorm
    # kernel; both use the same `Signal` layout (lamport_region +
    # lamport_state) but each rotates its own generation flag independently.
    var sigs_ar_devbufs = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var sigs_fused_devbufs = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs_ar = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var rank_sigs_fused = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    var gamma_host = alloc[Scalar[dtype]](K)
    for i in range(K):
        gamma_host[i] = random_float64(min=0.5, max=1.5).cast[dtype]()
    var act_host = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    var neg_zero = _neg_zero_bits()
    for g in range(ngpus):
        var a = alloc[Scalar[dtype]](act_size)
        for i in range(act_size):
            if seed_neg_zero and (i % 2 == 0):
                a[i] = neg_zero
            else:
                a[i] = random_float64(min=-0.5, max=0.5).cast[dtype]()
        act_host.append(a)

        act_in.append(list_of_ctx[g].enqueue_create_buffer[dtype](act_size))
        gamma.append(list_of_ctx[g].enqueue_create_buffer[dtype](K))
        ar_out.append(list_of_ctx[g].enqueue_create_buffer[dtype](act_size))
        unfused_out.append(
            list_of_ctx[g].enqueue_create_buffer[dtype](act_size)
        )
        fused_out.append(list_of_ctx[g].enqueue_create_buffer[dtype](act_size))
        list_of_ctx[g].enqueue_copy(act_in[g], act_host[g])
        list_of_ctx[g].enqueue_copy(gamma[g], gamma_host)

        # Both kernels need sentinel-initialized Signal buffers. The fused
        # kernel embeds the Lamport region in `Signal.lamport_region`; the
        # generic allreduce also routes to Lamport for these small bf16
        # messages and uses the same region.
        sigs_ar_devbufs.append(
            list_of_ctx[g].create_buffer_sync[.uint8](size_of[Signal]())
        )
        sigs_fused_devbufs.append(
            list_of_ctx[g].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(sigs_ar_devbufs[g], list_of_ctx[g])
        init_signal_buffer(sigs_fused_devbufs[g], list_of_ctx[g])
        rank_sigs_ar[g] = (
            sigs_ar_devbufs[g]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
        rank_sigs_fused[g] = (
            sigs_fused_devbufs[g]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    # ---- TileTensor views over the input/output buffers for `allreduce`. ----
    var act_layout = row_major(act_size)
    comptime InTensorType = TileTensor[
        dtype, type_of(row_major(0)), ImmutAnyOrigin
    ]
    var in_tensors = Array[_, ngpus](
        fill_with=lambda (g: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                act_in[g].unsafe_ptr()
            ),
            act_layout,
        )
    )
    comptime OutTensorType = TileTensor[
        dtype, type_of(row_major(0)), MutAnyOrigin
    ]
    var ar_out_tensors = Array[_, ngpus](
        fill_with=lambda (g: Int) -> OutTensorType: OutTensorType(
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                ar_out[g].unsafe_ptr()
            ),
            act_layout,
        )
    )

    # ---- Unfused reference: on-device AR -> on-device rms_norm_gpu. ----
    # `allreduce` routes to the standalone Lamport AR for this shape (small,
    # bf16, P2P), so the bf16 intermediate is exactly what the unfused
    # production path would produce. Then `rms_norm_gpu` reads that bf16 AR
    # sum and writes the normed result to `unfused_out[g]`.
    group_start()
    comptime for g in range(ngpus):
        allreduce[ngpus=ngpus](
            in_tensors, ar_out_tensors[g], rank_sigs_ar, list_of_ctx[g]
        )
    group_end()
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    for g in range(ngpus):
        _run_rms_norm_unfused(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                ar_out[g].unsafe_ptr()
            ),
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                unfused_out[g].unsafe_ptr()
            ),
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                gamma[g].unsafe_ptr()
            ),
            M,
            K,
            EPS,
            list_of_ctx[g],
        )
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    # Copy rank 0's unfused output back to host as the comparison reference
    # (all ranks compute identical AR+RMSNorm outputs).
    var expected_host = alloc[Scalar[dtype]](act_size)
    list_of_ctx[0].enqueue_copy(expected_host, unfused_out[0])
    list_of_ctx[0].synchronize()
    var expected = alloc[Float32](act_size)
    for i in range(act_size):
        expected[i] = expected_host[i].cast[.float32]()

    # ---- Fused kernel: `lamport_allreduce_rmsnorm` on each device. ----
    var host = alloc[Scalar[dtype]](act_size)
    var got = alloc[Float32](act_size)

    for it in range(iters):
        group_start()
        comptime for g in range(ngpus):
            lamport_allreduce_rmsnorm[dtype, ngpus, pdl=False](
                g,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    act_in[g].unsafe_ptr()
                ),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    fused_out[g].unsafe_ptr()
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma[g].unsafe_ptr()
                ),
                rank_sigs_fused,
                M,
                K,
                EPS.cast[dtype](),
                list_of_ctx[g],
            )
        group_end()
        for g in range(ngpus):
            list_of_ctx[g].synchronize()

        for g in range(ngpus):
            list_of_ctx[g].enqueue_copy(host, fused_out[g])
            list_of_ctx[g].synchronize()
            for i in range(act_size):
                got[i] = host[i].cast[.float32]()
            assert_almost_equal(
                got, expected, num_elements=act_size, atol=1e-2, rtol=1e-2
            )
        print("  [ok] iter", it, "matches unfused reference")

    for g in range(ngpus):
        act_host[g].free()
    gamma_host.free()
    expected_host.free()
    expected.free()
    host.free()
    got.free()
    print("PASSED\n")


def unsynced_skew_test[
    ngpus: Int
](list_of_ctx: List[DeviceContext], M: Int, K: Int) raises:
    """Enqueue NUM_UNSYNCED_ITERS fused calls back-to-back with NO cross-rank
    sync between calls, then verify every call's slice in a single final pass.

    `rmsnorm_test` `synchronize()`s every rank after each call, which pins
    the inter-rank generation skew to zero -- the three-generation slack is
    never exercised. A real model launches collectives back-to-back per rank
    with no cross-rank barrier, letting ranks drift in `flag`. A fast rank's
    per-call clear (generation `(flag+2)%3`) can race a slow rank's read of
    the generation being reused; if one generation of slack is ever
    insufficient, a call returns a wrong sum here.

    Each call uses a DISTINCT input/output slice of one big per-rank buffer,
    so every result survives to a single final verification pass -- no
    per-call sync between fused launches. Per-call inputs vary by call index
    (the constant rank-and-iter pattern from `test_lamport_allreduce.mojo`)
    so a stale cross-generation read is a detectable wrong sum. The unfused
    reference is computed slice-by-slice WITH per-call sync BEFORE the
    fused skew loop, so the reference is deterministic.

    Detection is timing-dependent -- run under `--runs_per_test`.
    """
    print(_case_str(M, K, tag="unsynced-skew"), "iters=", NUM_UNSYNCED_ITERS)
    var slice_size = M * K
    var total_size = NUM_UNSYNCED_ITERS * slice_size

    # Per-rank big buffers spanning all iter slices.
    var act_big = List[DeviceBuffer[dtype]](capacity=ngpus)
    var ar_big = List[DeviceBuffer[dtype]](capacity=ngpus)
    var unfused_big = List[DeviceBuffer[dtype]](capacity=ngpus)
    var fused_big = List[DeviceBuffer[dtype]](capacity=ngpus)
    var gamma = List[DeviceBuffer[dtype]](capacity=ngpus)
    var act_host = List[MutPointer[Scalar[dtype], MutUntrackedOrigin]](
        capacity=ngpus
    )

    # Separate signal buffers per kernel path -- as in `rmsnorm_test`.
    var sigs_ar_devbufs = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var sigs_fused_devbufs = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs_ar = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var rank_sigs_fused = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    var gamma_host = alloc[Scalar[dtype]](K)
    for i in range(K):
        gamma_host[i] = random_float64(min=0.5, max=1.5).cast[dtype]()

    for g in range(ngpus):
        # Per-call value for rank g, call it: (g+1) + (it % 5). Constant
        # across the slice, varies across calls, < 256 so the integer
        # ngpus-sum is exact in bf16.
        var h = alloc[Scalar[dtype]](total_size)
        for it in range(NUM_UNSYNCED_ITERS):
            var v = Scalar[dtype](Scalar[dtype](g + 1) + Scalar[dtype](it % 5))
            for j in range(slice_size):
                h[it * slice_size + j] = v
        act_host.append(h)

        act_big.append(list_of_ctx[g].enqueue_create_buffer[dtype](total_size))
        ar_big.append(list_of_ctx[g].enqueue_create_buffer[dtype](total_size))
        unfused_big.append(
            list_of_ctx[g].enqueue_create_buffer[dtype](total_size)
        )
        fused_big.append(
            list_of_ctx[g].enqueue_create_buffer[dtype](total_size)
        )
        gamma.append(list_of_ctx[g].enqueue_create_buffer[dtype](K))
        list_of_ctx[g].enqueue_copy(act_big[g], h)
        list_of_ctx[g].enqueue_copy(gamma[g], gamma_host)

        sigs_ar_devbufs.append(
            list_of_ctx[g].create_buffer_sync[.uint8](size_of[Signal]())
        )
        sigs_fused_devbufs.append(
            list_of_ctx[g].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(sigs_ar_devbufs[g], list_of_ctx[g])
        init_signal_buffer(sigs_fused_devbufs[g], list_of_ctx[g])
        rank_sigs_ar[g] = (
            sigs_ar_devbufs[g]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
        rank_sigs_fused[g] = (
            sigs_fused_devbufs[g]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    # ---- Unfused reference, slice-by-slice WITH per-call sync. ----
    comptime InType = TileTensor[dtype, type_of(row_major(0)), ImmutAnyOrigin]
    comptime OutType = TileTensor[dtype, type_of(row_major(0)), MutAnyOrigin]
    var act_layout = row_major(slice_size)

    for it in range(NUM_UNSYNCED_ITERS):
        var base = it * slice_size
        var in_tensors = Array[_, ngpus](
            fill_with=lambda (g: Int) -> InType: InType(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    act_big[g].unsafe_ptr() + base
                ),
                act_layout,
            )
        )
        var ar_out_tensors = Array[_, ngpus](
            fill_with=lambda (g: Int) -> OutType: OutType(
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    ar_big[g].unsafe_ptr() + base
                ),
                act_layout,
            )
        )

        group_start()
        comptime for g in range(ngpus):
            allreduce[ngpus=ngpus](
                in_tensors, ar_out_tensors[g], rank_sigs_ar, list_of_ctx[g]
            )
        group_end()
        for g in range(ngpus):
            list_of_ctx[g].synchronize()

        # rms_norm_gpu on this slice's AR output per device. Pointer-offset
        # into the big buffer (no DeviceBuffer slice API needed).
        for g in range(ngpus):
            _run_rms_norm_unfused(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    ar_big[g].unsafe_ptr() + base
                ),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    unfused_big[g].unsafe_ptr() + base
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma[g].unsafe_ptr()
                ),
                M,
                K,
                EPS,
                list_of_ctx[g],
            )
        for g in range(ngpus):
            list_of_ctx[g].synchronize()

    # ---- Fused: enqueue all calls back-to-back, NO cross-rank sync. ----
    for it in range(NUM_UNSYNCED_ITERS):
        var base = it * slice_size
        comptime for g in range(ngpus):
            lamport_allreduce_rmsnorm[dtype, ngpus, pdl=False](
                g,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    act_big[g].unsafe_ptr() + base
                ),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    fused_big[g].unsafe_ptr() + base
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma[g].unsafe_ptr()
                ),
                rank_sigs_fused,
                M,
                K,
                EPS.cast[dtype](),
                list_of_ctx[g],
            )
        # NO cross-rank synchronize() here -- the whole point of the test.

    # Single barrier after all calls are enqueued, then verify every slice.
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    var unfused_host = alloc[Scalar[dtype]](total_size)
    var fused_host = alloc[Scalar[dtype]](total_size)
    var expected_f32 = alloc[Float32](slice_size)
    var got_f32 = alloc[Float32](slice_size)

    # Verify rank 0's slices against rank 0's unfused reference.
    list_of_ctx[0].enqueue_copy(unfused_host, unfused_big[0])
    list_of_ctx[0].enqueue_copy(fused_host, fused_big[0])
    list_of_ctx[0].synchronize()

    for it in range(NUM_UNSYNCED_ITERS):
        var base = it * slice_size
        for i in range(slice_size):
            expected_f32[i] = unfused_host[base + i].cast[.float32]()
            got_f32[i] = fused_host[base + i].cast[.float32]()
        assert_almost_equal(
            got_f32,
            expected_f32,
            num_elements=slice_size,
            atol=1e-2,
            rtol=1e-2,
        )

    for g in range(ngpus):
        act_host[g].free()
    gamma_host.free()
    unfused_host.free()
    fused_host.free()
    expected_f32.free()
    got_f32.free()
    print("  [ok] all", NUM_UNSYNCED_ITERS, "unsynced slices match")
    print("PASSED\n")


def main() raises:
    assert_true(
        DeviceContext.number_of_devices() > 1, "must have multiple GPUs"
    )
    assert_true(enable_p2p(), "failed to enable P2P access between GPUs")

    comptime ngpus = 2
    var ctx = List[DeviceContext]()
    for i in range(ngpus):
        ctx.append(DeviceContext(device_id=i))

    # Shape sweep. K must be a multiple of `atomic_width` (= 16B / sizeof(dtype)
    # = 8 for bf16) and at most BLOCK_SIZE * atomic_width (= 1024 * 8 = 8192 on
    # Hopper/Blackwell). M is unconstrained at the kernel level.
    #
    # The shapes below cover: small-M decode (Kimi/Llama hidden sizes), wider
    # K, multi-row decode, and the cols == BLOCK_SIZE * atomic_width edge.
    var shapes = List[Tuple[Int, Int]]()
    shapes.append((1, 2048))
    shapes.append((1, 7168))
    shapes.append((8, 4096))
    shapes.append((8, 7168))
    shapes.append((16, 7168))
    shapes.append((32, 1024))
    shapes.append((4, 8192))  # cols == BLOCK_SIZE * atomic_width edge

    # Shape sweep with random inputs (12 iters wraps the Lamport rotation 4x).
    for shape in shapes:
        print(_case_str(shape[0], shape[1]))
        rmsnorm_test[ngpus](ctx, shape[0], shape[1])

    # Producer-sanitize edge case: every other input lane is `-0.0`. Sums are
    # unchanged (x + (-0.0) == x), but the kernel must `remove_neg_zero` the
    # input or real data is mistaken for the readiness sentinel. Two shapes
    # are enough -- the property is shape-independent.
    print(_case_str(8, 7168, tag="negzero"))
    rmsnorm_test[ngpus, seed_neg_zero=True](ctx, 8, 7168)
    print(_case_str(1, 7168, tag="negzero"))
    rmsnorm_test[ngpus, seed_neg_zero=True](ctx, 1, 7168)

    # Unsynced back-to-back stress: many calls without cross-rank sync,
    # exercising the generation-rotation slack under real inter-rank drift.
    # Race-dependent -- run under `--runs_per_test` for race detection.
    unsynced_skew_test[ngpus](ctx, 8, 7168)

    print("All fused Lamport AR+RMSNorm checks passed.")
