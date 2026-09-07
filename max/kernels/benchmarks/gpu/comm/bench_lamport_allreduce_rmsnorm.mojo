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

"""Benchmark for the fused Lamport allreduce + RMSNorm pipeline.

Measures two variants on the same `[num_rows, num_cols]` bf16 input:

1. **fused** — `lamport_allreduce_rmsnorm` (single kernel: one-shot Lamport AR
   with the RMSNorm epilogue inline; PDL off so the timing is the kernel
   alone, not its consumer-overlap window).
2. **unfused** — `allreduce` (which routes to the standalone Lamport AR for
   these small bf16 messages with P2P) followed by `rms_norm_gpu` (the
   warp-tiling GPU RMSNorm). This is the composition the fused kernel is
   meant to beat.

The fused kernel constrains `cols % atomic_width == 0` (atomic_width = 16B /
size_of[dtype], so 8 for bf16) and `cols / atomic_width <= BLOCK_SIZE`
(1024 on Hopper/Blackwell, capping bf16 hidden at 8192). The YAML sweeps
shapes within those bounds.

Verification compares the fused and unfused outputs element-wise; they
differ only by the bf16 round-trip on the unfused AR intermediate (the
fused kernel keeps the AR sum in fp32 registers all the way through), so
the tolerance is generous.
"""

from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    size_of,
    simd_width_of,
)

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.benchmark import (
    bench_multicontext,
    bencher_iter_custom,
)
from comm import Signal, MAX_GPUS, group_start, group_end
from comm.allreduce import allreduce
from comm.allreduce_lamport_rmsnorm import lamport_allreduce_rmsnorm
from comm.sync import enable_p2p, init_signal_buffer, is_p2p_enabled
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import CacheBustingBuffer, arg_parse, assert_almost_equal

from layout import Coord, TileTensor, row_major
from nn.normalization import rms_norm_gpu

from std.utils import IndexList
from std.utils.index import Index


# ---------------------------------------------------------------------------
# Helper: per-device RMSNorm reading from `in_ptr` and writing to `out_ptr`.
# Wraps `rms_norm_gpu` with the input/output lambdas it requires so the
# benchmark closures stay readable.
# ---------------------------------------------------------------------------
def _run_rms_norm[
    dtype: DType
](
    in_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    out_ptr: MutPointer[Scalar[dtype], MutAnyOrigin],
    gamma_ptr: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    M: Int,
    K: Int,
    epsilon: Float32,
    ctx: DeviceContext,
) raises:
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


# ---------------------------------------------------------------------------
# Verification: compare fused vs unfused outputs on GPU 0 with bf16 tolerance.
# ---------------------------------------------------------------------------
def _verify_results[
    dtype: DType,
    ngpus: Int,
    num_cols: Int,
](
    num_rows: Int,
    list_of_ctx: List[DeviceContext],
    sigs_ar: List[DeviceBuffer[.uint8]],
    sigs_fused: List[DeviceBuffer[.uint8]],
    cb_inputs: List[CacheBustingBuffer[dtype]],
    rank_sigs_ar: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    rank_sigs_fused: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    gamma_dev: DeviceBuffer[dtype],
    epsilon: Scalar[dtype],
) raises:
    var length = num_rows * num_cols
    var ctx0 = list_of_ctx[0]

    var v_ar_out_dev = ctx0.enqueue_create_buffer[dtype](length)
    var v_unfused_dev = ctx0.enqueue_create_buffer[dtype](length)
    var v_fused_dev = ctx0.enqueue_create_buffer[dtype](length)

    # Re-init both signal buffers so the generation flag starts from 0 for
    # this comparison (matches the bench's first call).
    for i in range(ngpus):
        init_signal_buffer(sigs_ar[i], list_of_ctx[i])
        init_signal_buffer(sigs_fused[i], list_of_ctx[i])
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # ---- Unfused path on GPU 0: allreduce -> rms_norm_gpu. ----
    comptime InTensorType = TileTensor[
        dtype, type_of(row_major(0)), ImmutAnyOrigin
    ]
    comptime OutTensorType = TileTensor[
        dtype, type_of(row_major(0)), MutAnyOrigin
    ]
    var v_layout = row_major(length)
    var v_in_tensors = Array[_, ngpus](
        fill_with=lambda (g: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                cb_inputs[g].offset_ptr(0)
            ),
            v_layout,
        )
    )
    var v_ar_out = OutTensorType(
        rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
            v_ar_out_dev.unsafe_ptr()
        ),
        v_layout,
    )
    allreduce[ngpus=ngpus](v_in_tensors, v_ar_out, rank_sigs_ar, ctx0)
    ctx0.synchronize()
    _run_rms_norm(
        rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
            v_ar_out_dev.unsafe_ptr()
        ),
        rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
            v_unfused_dev.unsafe_ptr()
        ),
        rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
            gamma_dev.unsafe_ptr()
        ),
        num_rows,
        num_cols,
        epsilon.cast[.float32](),
        ctx0,
    )
    ctx0.synchronize()

    # ---- Fused path on GPU 0 (PDL off for clean comparison). ----
    # Both ranks must call so the Lamport pushes complete; otherwise GPU 0's
    # poll blocks waiting for peers.
    group_start()
    comptime for g in range(ngpus):
        var out_ptr = v_fused_dev.unsafe_ptr() if g == 0 else cb_inputs[
            g
        ].offset_ptr(0)
        lamport_allreduce_rmsnorm[dtype, ngpus, pdl=False](
            g,
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                cb_inputs[g].offset_ptr(0)
            ),
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](out_ptr),
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                gamma_dev.unsafe_ptr()
            ),
            rank_sigs_fused,
            num_rows,
            num_cols,
            epsilon,
            list_of_ctx[g],
        )
    group_end()
    for g in range(ngpus):
        list_of_ctx[g].synchronize()

    # ---- Element-wise compare on GPU 0. ----
    var unfused_h = List(length=length, fill=Scalar[dtype](0))
    var fused_h = List(length=length, fill=Scalar[dtype](0))
    ctx0.enqueue_copy(unfused_h, v_unfused_dev)
    ctx0.enqueue_copy(fused_h, v_fused_dev)
    ctx0.synchronize()

    # The unfused chain stores the AR sum in bf16 before RMSNorm; the fused
    # kernel keeps the sum in fp32 registers. The two paths therefore differ
    # only by the bf16 round-trip on the AR intermediate. Use the same
    # tolerance the unit test uses (`atol=rtol=1e-2`).
    var unfused_f32 = List[Float32](length=length, fill=Float32(0))
    var fused_f32 = List[Float32](length=length, fill=Float32(0))
    for i in range(length):
        unfused_f32[i] = unfused_h[i].cast[.float32]()
        fused_f32[i] = fused_h[i].cast[.float32]()
    assert_almost_equal(
        fused_f32.unsafe_ptr(),
        unfused_f32.unsafe_ptr(),
        length,
        atol=1e-2,
        rtol=1e-2,
    )

    _ = v_ar_out_dev^
    _ = v_unfused_dev^
    _ = v_fused_dev^
    _ = unfused_h^
    _ = fused_h^
    _ = unfused_f32^
    _ = fused_f32^

    print("Verification PASSED")


def bench_fused_lamport_allreduce_rmsnorm[
    dtype: DType,
    ngpus: Int,
    num_cols: Int,
    cache_busting: Bool = True,
](num_rows: Int, mut b: Bench, list_of_ctx: List[DeviceContext]) raises:
    var length = num_rows * num_cols

    # ---- Shared per-GPU input + signal buffer setup. ----
    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

    var cb_inputs = List[CacheBustingBuffer[dtype]]()
    var ar_out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var unfused_out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var fused_out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_bufs = List[List[Scalar[dtype]]](capacity=ngpus)

    # Two independent signal buffers: one for the unfused `allreduce` path
    # (Lamport-routed under the hood) and one for the fused kernel. They
    # share the Signal layout but rotate their own generation flag, so
    # interleaving the two paths on the same buffer would confuse the state.
    var sigs_ar = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var sigs_fused = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs_ar = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    var rank_sigs_fused = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # The 2-stage path may use trailing scratch; reserve it for the unfused
    # signal so `allreduce` works for any size the YAML sweeps to.
    var temp_bytes = ngpus * size_of[dtype]() * length

    for i in range(ngpus):
        cb_inputs.append(
            CacheBustingBuffer[dtype](
                length, simd_size, list_of_ctx[i], cache_busting
            )
        )
        ar_out_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](length))
        unfused_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[dtype](length)
        )
        fused_out_dev.append(
            list_of_ctx[i].enqueue_create_buffer[dtype](length)
        )

        var h = List[Scalar[dtype]](
            unsafe_uninit_length=cb_inputs[0].alloc_size()
        )
        # `(i+1) + (j % 251)` — same per-rank seeding the bare-Lamport test
        # uses; 251 is the largest prime < 256 to avoid power-of-two aliasing.
        for j in range(cb_inputs[0].alloc_size()):
            h[j] = Scalar[dtype](i + 1) + Scalar[dtype](j % 251)
        list_of_ctx[i].enqueue_copy(cb_inputs[i].device_buffer(), h)
        host_bufs.append(h^)

        sigs_ar.append(
            list_of_ctx[i].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_bytes
            )
        )
        sigs_fused.append(
            list_of_ctx[i].create_buffer_sync[.uint8](size_of[Signal]())
        )
        init_signal_buffer(sigs_ar[i], list_of_ctx[i])
        init_signal_buffer(sigs_fused[i], list_of_ctx[i])
        rank_sigs_ar[i] = (
            sigs_ar[i].unsafe_ptr().bitcast[Signal]().as_unsafe_any_origin()
        )
        rank_sigs_fused[i] = (
            sigs_fused[i].unsafe_ptr().bitcast[Signal]().as_unsafe_any_origin()
        )

    # Gamma (replicated across devices; benches read from each rank's copy).
    var gamma_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var gamma_host = List(length=num_cols, fill=Scalar[dtype](0))
    for i in range(num_cols):
        gamma_host[i] = (Float64(i + num_cols) / Float64(num_cols)).cast[
            dtype
        ]()
    for i in range(ngpus):
        gamma_dev.append(list_of_ctx[i].enqueue_create_buffer[dtype](num_cols))
        list_of_ctx[i].enqueue_copy(gamma_dev[i], gamma_host)

    var epsilon = Scalar[dtype](0.001)

    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # TileTensor views for the unfused `allreduce` path.
    comptime InTensorType = TileTensor[
        dtype, type_of(row_major(0)), ImmutAnyOrigin
    ]
    comptime OutTensorType = TileTensor[
        dtype, type_of(row_major(0)), MutAnyOrigin
    ]
    var in_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> InTensorType: InTensorType(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                cb_inputs[i].unsafe_ptr()
            ),
            row_major(length),
        )
    )
    var ar_out_tensors = Array[_, ngpus](
        fill_with=lambda (i: Int) -> OutTensorType: OutTensorType(
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                ar_out_dev[i].unsafe_ptr()
            ),
            row_major(length),
        )
    )

    # Pre-capture pointers for the per-iter closures (CacheBustingBuffer uses
    # `offset_ptr(cache_iter)` to rotate through allocations).
    comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    var ar_out_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> PtrType: ar_out_dev[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var unfused_out_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> PtrType: unfused_out_dev[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    var fused_out_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> PtrType: fused_out_dev[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )
    # Gamma pointers stored as Mut for convenience; rebind to Immut at the
    # call site (only the kernel arg slot needs Immut).
    var gamma_ptrs = Array[_, ngpus](
        fill_with=lambda (i: Int) {ref} -> PtrType: gamma_dev[i]
        .unsafe_ptr()
        .as_unsafe_any_origin()
    )

    # ---- Benchmark naming + throughput accounting. ----
    var total_bytes = ngpus * length * size_of[dtype]()
    var bench_name_prefix = String(
        "lamport_allreduce_rmsnorm/",
        dtype,
        "/",
        ngpus,
        "gpu/",
        num_rows,
        "x",
        num_cols,
    )

    # ===== Benchmark 1: fused `lamport_allreduce_rmsnorm` (1 kernel) =====
    @always_inline
    def bench_fused_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {imm}:
        @always_inline
        def call_fn(ctx_inner: DeviceContext, cache_iter: Int) raises {imm}:
            lamport_allreduce_rmsnorm[dtype, ngpus, pdl=False](
                ctx_idx,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    cb_inputs[ctx_idx].offset_ptr(cache_iter)
                ),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    fused_out_ptrs[ctx_idx]
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma_ptrs[ctx_idx]
                ),
                rank_sigs_fused,
                num_rows,
                num_cols,
                epsilon,
                ctx_inner,
            )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_fused_iter,
        list_of_ctx,
        BenchId("fused_lamport_allreduce_rmsnorm", input_id=bench_name_prefix),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    # ===== Benchmark 2: unfused `allreduce` + `rms_norm_gpu` (2 kernels) =====
    @always_inline
    def bench_unfused_iter(
        mut bench: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_tensors, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_tensors, imm}:
            comptime for _j in range(ngpus):
                in_tensors[_j] = InTensorType(
                    rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                        cb_inputs[_j].offset_ptr(cache_iter)
                    ),
                    row_major(length),
                )

            allreduce[ngpus=ngpus](
                in_tensors,
                ar_out_tensors[ctx_idx],
                rank_sigs_ar,
                ctx_inner,
            )
            _run_rms_norm(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    ar_out_ptrs[ctx_idx]
                ),
                rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                    unfused_out_ptrs[ctx_idx]
                ),
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    gamma_ptrs[ctx_idx]
                ),
                num_rows,
                num_cols,
                epsilon.cast[.float32](),
                ctx_inner,
            )

        bencher_iter_custom(bench, call_fn, ctx)

    bench_multicontext(
        b,
        bench_unfused_iter,
        list_of_ctx,
        BenchId(
            "unfused_allreduce_then_rms_norm",
            input_id=bench_name_prefix,
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    b.dump_report()

    # Correctness for the fused vs unfused composition is covered by the unit
    # test at `max/kernels/test/gpu/comm/test_fused_lamport_rmsnorm.mojo`,
    # which runs the same comparison across the same shape sweep with a
    # per-iteration assert. This bench is timing-only.

    # Cleanup.
    _ = host_bufs^
    _ = sigs_ar^
    _ = sigs_fused^
    _ = cb_inputs^
    _ = ar_out_dev^
    _ = unfused_out_dev^
    _ = fused_out_dev^
    _ = gamma_dev^
    _ = gamma_host^


def main() raises:
    comptime dtype = get_defined_dtype["in_dtype", .bfloat16]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()
    var num_rows = Int(arg_parse("num_rows", 8))
    comptime num_cols = get_defined_int["num_cols", 7168]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()

    var num_devices = DeviceContext.number_of_devices()
    if num_devices < num_gpus:
        print(
            "Need",
            num_gpus,
            "GPUs but only found",
            num_devices,
            "- skipping.",
        )
        return

    _ = enable_p2p()
    if not is_p2p_enabled():
        print("P2P not enabled, skipping benchmark.")
        return

    # The fused kernel requires `cols / atomic_width <= BLOCK_SIZE` (= 1024
    # on Hopper/Blackwell), i.e. `cols <= 8192` for bf16. Larger shapes
    # would deadlock the kernel; gate them out at the bench boundary.
    comptime atomic_width = 16 // size_of[dtype]()
    comptime if num_cols % atomic_width != 0:
        print(
            "num_cols=",
            num_cols,
            " not a multiple of atomic_width=",
            atomic_width,
            " (bf16 needs cols % 8 == 0); skipping.",
        )
        return
    comptime BLOCK_SIZE_MAX = 1024
    comptime if num_cols // atomic_width > BLOCK_SIZE_MAX:
        print(
            "num_cols=",
            num_cols,
            " exceeds BLOCK_SIZE*atomic_width=",
            BLOCK_SIZE_MAX * atomic_width,
            " (kernel cap); skipping.",
        )
        return

    var list_of_ctx = List[DeviceContext]()
    for i in range(num_gpus):
        list_of_ctx.append(DeviceContext(device_id=i))

    print(
        "Benchmarking fused vs unfused Lamport allreduce + RMSNorm:",
        num_gpus,
        "GPUs,",
        dtype,
        ",",
        num_rows,
        "x",
        num_cols,
    )

    var m = Bench(BenchConfig(num_repetitions=1))
    bench_fused_lamport_allreduce_rmsnorm[
        dtype, num_gpus, num_cols, cache_busting
    ](num_rows, m, list_of_ctx)
