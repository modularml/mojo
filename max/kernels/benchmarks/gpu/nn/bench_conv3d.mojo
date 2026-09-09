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
"""Benchmark the native Mojo 3D conv paths against cuDNN.

Driven by `kbench` via a YAML shape sweep; one CSV row per `(impl, shape)`.
`impl` selects among `naive`, `im2col`, `1x1x1`, `qslice`, and `cudnn`.
Dispatchers that decline a shape raise an `Error` so kbench records the
(impl, shape) as failed rather than timing a no-op that misleadingly
looks fastest.

Usage (standalone):
    bazel run --config=remote-b200 \\
        //max/kernels/benchmarks/gpu/nn:bench_conv3d

Usage (kbench):
    ./bazelw run //max/kernels/benchmarks/autotune:kbench -- \\
        max/kernels/benchmarks/gpu/nn/bench_conv3d.yaml \\
        --target-accelerator cuda:b200 -c
"""

from std.math import ceildiv
from std.random import rand
from std.sys import get_defined_dtype, get_defined_int
from std.sys.info import has_amd_gpu_accelerator

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import arg_parse
from layout import (
    UNKNOWN_VALUE,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from nn.conv.conv import conv3d_gpu_naive_ndhwc_qrscf, conv3d_cudnn, conv_miopen
from nn.conv.gpu.amd.dispatch_3d import dispatch_amd_4wave_conv3d
from nn.conv.gpu.im2col_matmul_3d import dispatch_im2col_matmul_conv3d
from nn.conv.gpu.matmul_1x1x1_conv3d import dispatch_1x1x1_matmul_conv3d
from nn.conv.gpu.nvidia.sm100.qslice_conv3d import (
    dispatch_qslice_conv3d_sm100,
)

from std.utils.index import IndexList


def compute_conv3d_flops(
    batch: Int,
    d_out: Int,
    h_out: Int,
    w_out: Int,
    c_out: Int,
    c_in: Int,
    q: Int,
    r: Int,
    s: Int,
) -> Int:
    return 2 * batch * d_out * h_out * w_out * c_out * c_in * q * r * s


def bench_conv3d[
    dtype: DType,
    batch: Int,
    in_depth: Int,
    in_height: Int,
    in_width: Int,
    in_channels: Int,
    out_channels: Int,
    filter_q: Int,
    filter_r: Int,
    filter_s: Int,
](
    ctx: DeviceContext,
    mut b: Bench,
    label: String,
    impl: String,
    verify: Bool,
    pad_d: Int,
    pad_h: Int,
    pad_w: Int,
    stride_d: Int,
    stride_h: Int,
    stride_w: Int,
) raises:
    var d_out = (in_depth + 2 * pad_d - filter_q) // stride_d + 1
    var h_out = (in_height + 2 * pad_h - filter_r) // stride_h + 1
    var w_out = (in_width + 2 * pad_w - filter_s) // stride_w + 1

    comptime input_layout = Layout.row_major(
        batch, in_depth, in_height, in_width, in_channels
    )
    comptime filter_qrscf_layout = Layout.row_major(
        filter_q, filter_r, filter_s, in_channels, out_channels
    )
    comptime filter_fcqrs_layout = Layout.row_major(
        out_channels, in_channels, filter_q, filter_r, filter_s
    )
    # Output spatial dims depend on runtime pad / stride, so leave them as
    # UNKNOWN_VALUE in the static layout and supply concrete sizes via a
    # RuntimeLayout below.
    comptime output_layout = Layout.row_major(
        batch, UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE, out_channels
    )

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_qrscf_layout.size())
    var output_size = batch * d_out * h_out * w_out * out_channels

    var flops = compute_conv3d_flops(
        batch,
        d_out,
        h_out,
        w_out,
        out_channels,
        in_channels,
        filter_q,
        filter_r,
        filter_s,
    )

    var bench_input_id = String(
        label,
        "/impl=",
        impl,
        "/dtype=",
        dtype,
        "/NDHWC=(",
        batch,
        "x",
        in_depth,
        "x",
        in_height,
        "x",
        in_width,
        "x",
        in_channels,
        ")/filter=(",
        filter_q,
        "x",
        filter_r,
        "x",
        filter_s,
        ")/Cout=",
        out_channels,
        "/pad=(",
        pad_d,
        "x",
        pad_h,
        "x",
        pad_w,
        ")/stride=(",
        stride_d,
        "x",
        stride_h,
        "x",
        stride_w,
        ")",
    )

    # Host buffers.
    var input_host = List(length=input_size, fill=Scalar[dtype](0))
    var filter_qrscf_host = List(length=filter_size, fill=Scalar[dtype](0))
    var filter_fcqrs_host = List(length=filter_size, fill=Scalar[dtype](0))
    rand[dtype](input_host)
    rand[dtype](filter_qrscf_host)

    # QRSCF [Q,R,S,C,F] -> FCQRS [F,C,Q,R,S] for cuDNN.
    for f in range(out_channels):
        for c in range(in_channels):
            for q in range(filter_q):
                for r in range(filter_r):
                    for s in range(filter_s):
                        var qrscf_idx = (
                            q * filter_r * filter_s * in_channels * out_channels
                            + r * filter_s * in_channels * out_channels
                            + s * in_channels * out_channels
                            + c * out_channels
                            + f
                        )
                        var fcqrs_idx = (
                            f * in_channels * filter_q * filter_r * filter_s
                            + c * filter_q * filter_r * filter_s
                            + q * filter_r * filter_s
                            + r * filter_s
                            + s
                        )
                        filter_fcqrs_host[fcqrs_idx] = filter_qrscf_host[
                            qrscf_idx
                        ]

    # Device buffers.
    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_qrscf_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var filter_fcqrs_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)
    var output_ref_dev = ctx.enqueue_create_buffer[dtype](
        output_size if verify else 1
    )

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_qrscf_dev, filter_qrscf_host)
    ctx.enqueue_copy(filter_fcqrs_dev, filter_fcqrs_host)
    ctx.synchronize()

    # LayoutTensor views (used by naive + cudnn 5D paths).
    var input_buf = LayoutTensor[dtype, input_layout](input_dev.unsafe_ptr())
    var filter_qrscf_buf = LayoutTensor[dtype, filter_qrscf_layout](
        filter_qrscf_dev.unsafe_ptr()
    )
    var filter_fcqrs_buf = LayoutTensor[dtype, filter_fcqrs_layout](
        filter_fcqrs_dev.unsafe_ptr()
    )
    var output_runtime_layout = RuntimeLayout[output_layout].row_major(
        IndexList[5](batch, d_out, h_out, w_out, out_channels)
    )
    var output_buf = LayoutTensor[dtype, output_layout](
        output_dev.unsafe_ptr(), output_runtime_layout
    )

    # TileTensor views (used by dispatcher-based paths).
    var input_tt = TileTensor(
        input_dev,
        row_major(
            Coord(
                batch,
                in_depth,
                in_height,
                in_width,
                Idx[in_channels],
            )
        ),
    )
    var filter_qrscf_tt = TileTensor(
        filter_qrscf_dev,
        row_major(
            Coord(
                Idx[filter_q],
                Idx[filter_r],
                Idx[filter_s],
                Idx[in_channels],
                Idx[out_channels],
            )
        ),
    )
    var output_tt = TileTensor(
        output_dev,
        row_major(
            Coord(
                batch,
                d_out,
                h_out,
                w_out,
                Idx[out_channels],
            )
        ),
    )

    var stride_idx = IndexList[3](stride_d, stride_h, stride_w)
    var dilation_idx = IndexList[3](1, 1, 1)
    var pad_idx = IndexList[3](pad_d, pad_h, pad_w)

    comptime block_size = 16

    # Probe dispatcher-based impls once. On decline, raise so kbench logs
    # this (impl, shape) as failed instead of timing a no-op (which would
    # otherwise look fastest in the CSV).
    if impl == "im2col":
        var accepted = dispatch_im2col_matmul_conv3d(
            input_tt,
            filter_qrscf_tt,
            output_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
        ctx.synchronize()
        if not accepted:
            raise Error(
                "dispatch_im2col_matmul_conv3d declined: " + bench_input_id
            )
    elif impl == "1x1x1":
        var accepted = dispatch_1x1x1_matmul_conv3d(
            input_tt,
            filter_qrscf_tt,
            output_tt,
            stride_idx,
            dilation_idx,
            pad_idx,
            1,
            ctx,
        )
        ctx.synchronize()
        if not accepted:
            raise Error(
                "dispatch_1x1x1_matmul_conv3d declined: " + bench_input_id
            )
    elif impl == "qslice":
        comptime if not has_amd_gpu_accelerator():
            var accepted = dispatch_qslice_conv3d_sm100(
                input_tt,
                filter_qrscf_tt,
                output_tt,
                stride_idx,
                dilation_idx,
                pad_idx,
                1,
                ctx,
            )
            ctx.synchronize()
            if not accepted:
                raise Error(
                    "dispatch_qslice_conv3d_sm100 declined: " + bench_input_id
                )
        else:
            raise Error(
                "impl=qslice is SM100-only; use native_3d on AMD: "
                + bench_input_id
            )
    elif impl == "native_3d":
        comptime if has_amd_gpu_accelerator():
            # Autotune-override knobs (0 = use the dispatcher's
            # built-in heuristic). Pass via `-D BM_OVERRIDE=64` etc
            # to sweep configs for the per-shape dispatch table.
            comptime _BM_OVERRIDE = get_defined_int["BM_OVERRIDE", 0]()
            comptime _BN_OVERRIDE = get_defined_int["BN_OVERRIDE", 0]()
            comptime _BK_OVERRIDE = get_defined_int["BK_OVERRIDE", 0]()
            var accepted = dispatch_amd_4wave_conv3d[
                input_type=dtype,
                filter_type=dtype,
                output_type=dtype,
                filter_is_fcqrs=False,
                block_m_override=_BM_OVERRIDE,
                block_n_override=_BN_OVERRIDE,
                block_k_override=_BK_OVERRIDE,
            ](
                input_tt,
                filter_qrscf_tt,
                output_tt,
                stride_idx,
                dilation_idx,
                pad_idx,
                1,
                ctx,
            )
            ctx.synchronize()
            if not accepted:
                raise Error(
                    "dispatch_amd_4wave_conv3d declined: " + bench_input_id
                )
        else:
            raise Error("impl=native_3d is AMD-only: " + bench_input_id)

    if impl == "im2col":

        @always_inline
        def im2col_bench(
            mut bencher: Bencher,
        ) raises {var input_tt, var filter_qrscf_tt, var output_tt, imm,}:
            @always_inline
            def kernel(ctx: DeviceContext) raises {imm}:
                _ = dispatch_im2col_matmul_conv3d(
                    input_tt,
                    filter_qrscf_tt,
                    output_tt,
                    stride_idx,
                    dilation_idx,
                    pad_idx,
                    1,
                    ctx,
                )

            bencher_iter_custom(bencher, kernel, ctx)

        b.bench_function(
            im2col_bench,
            BenchId("conv3d_im2col", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )
    elif impl == "1x1x1":

        @always_inline
        def p1x1x1_bench(
            mut bencher: Bencher,
        ) raises {var input_tt, var filter_qrscf_tt, var output_tt, imm,}:
            @always_inline
            def kernel(ctx: DeviceContext) raises {imm}:
                _ = dispatch_1x1x1_matmul_conv3d(
                    input_tt,
                    filter_qrscf_tt,
                    output_tt,
                    stride_idx,
                    dilation_idx,
                    pad_idx,
                    1,
                    ctx,
                )

            bencher_iter_custom(bencher, kernel, ctx)

        b.bench_function(
            p1x1x1_bench,
            BenchId("conv3d_1x1x1", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )
    elif impl == "qslice":
        comptime if not has_amd_gpu_accelerator():

            @always_inline
            def qslice_bench(
                mut bencher: Bencher,
            ) raises {var input_tt, var filter_qrscf_tt, var output_tt, imm,}:
                @always_inline
                def kernel(ctx: DeviceContext) raises {imm}:
                    _ = dispatch_qslice_conv3d_sm100(
                        input_tt,
                        filter_qrscf_tt,
                        output_tt,
                        stride_idx,
                        dilation_idx,
                        pad_idx,
                        1,
                        ctx,
                    )

                bencher_iter_custom(bencher, kernel, ctx)

            b.bench_function(
                qslice_bench,
                BenchId("conv3d_qslice", input_id=bench_input_id),
                [ThroughputMeasure(BenchMetric.flops, flops)],
            )
    elif impl == "native_3d":
        comptime if has_amd_gpu_accelerator():
            comptime _BM_OVERRIDE = get_defined_int["BM_OVERRIDE", 0]()
            comptime _BN_OVERRIDE = get_defined_int["BN_OVERRIDE", 0]()
            comptime _BK_OVERRIDE = get_defined_int["BK_OVERRIDE", 0]()

            @always_inline
            def native_3d_bench(
                mut bencher: Bencher,
            ) raises {var input_tt, var filter_qrscf_tt, var output_tt, imm,}:
                @always_inline
                def kernel(ctx: DeviceContext) raises {imm}:
                    _ = dispatch_amd_4wave_conv3d[
                        input_type=dtype,
                        filter_type=dtype,
                        output_type=dtype,
                        filter_is_fcqrs=False,
                        block_m_override=_BM_OVERRIDE,
                        block_n_override=_BN_OVERRIDE,
                        block_k_override=_BK_OVERRIDE,
                    ](
                        input_tt,
                        filter_qrscf_tt,
                        output_tt,
                        stride_idx,
                        dilation_idx,
                        pad_idx,
                        1,
                        ctx,
                    )

                bencher_iter_custom(bencher, kernel, ctx)

            b.bench_function(
                native_3d_bench,
                BenchId("conv3d_native_3d", input_id=bench_input_id),
                [ThroughputMeasure(BenchMetric.flops, flops)],
            )
    elif impl == "cudnn":
        comptime if has_amd_gpu_accelerator():
            # Capturing `input_buf` / `output_buf` here would alias the
            # `input_tt` / `output_tt` views this arm launches through.
            @always_inline
            def miopen_bench(
                mut bencher: Bencher,
            ) raises {var filter_qrscf_tt, imm,}:
                @always_inline
                def kernel(ctx: DeviceContext) raises {imm}:
                    conv_miopen(
                        input_tt,
                        filter_qrscf_tt,
                        output_tt,
                        stride_idx,
                        dilation_idx,
                        pad_idx,
                        1,
                        ctx,
                    )

                bencher_iter_custom(bencher, kernel, ctx)

            b.bench_function(
                miopen_bench,
                BenchId("conv3d_miopen", input_id=bench_input_id),
                [ThroughputMeasure(BenchMetric.flops, flops)],
            )

        else:

            @always_inline
            def cudnn_bench(
                mut bencher: Bencher,
            ) raises {
                var input_buf,
                var filter_fcqrs_buf,
                var output_buf,
                imm,
            }:
                @always_inline
                def kernel(ctx: DeviceContext) raises {imm}:
                    conv3d_cudnn(
                        input_buf,
                        filter_fcqrs_buf,
                        output_buf,
                        stride_idx,
                        dilation_idx,
                        pad_idx,
                        1,
                        ctx,
                    )

                bencher_iter_custom(bencher, kernel, ctx)

            b.bench_function(
                cudnn_bench,
                BenchId("conv3d_cudnn", input_id=bench_input_id),
                [ThroughputMeasure(BenchMetric.flops, flops)],
            )
    else:
        # Naive Mojo NDHWC-QRSCF kernel.
        comptime naive_kernel = conv3d_gpu_naive_ndhwc_qrscf[
            input_layout,
            filter_qrscf_layout,
            output_layout,
            dtype,
            dtype,
            dtype,
            block_size,
            None,
        ]
        var grid_dim_x = ceildiv(w_out * h_out, block_size)
        var grid_dim_y = ceildiv(d_out, block_size)
        var grid_dim_z = batch

        @always_inline
        def naive_bench(
            mut bencher: Bencher,
        ) raises {var input_buf, var filter_qrscf_buf, var output_buf, imm,}:
            @always_inline
            def kernel(ctx: DeviceContext) raises {imm}:
                ctx.enqueue_function[naive_kernel](
                    input_buf,
                    filter_qrscf_buf,
                    output_buf,
                    stride_idx,
                    dilation_idx,
                    pad_idx,
                    Int(1),
                    grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
                    block_dim=(block_size, block_size, 1),
                )

            bencher_iter_custom(bencher, kernel, ctx)

        b.bench_function(
            naive_bench,
            BenchId("conv3d_naive", input_id=bench_input_id),
            [ThroughputMeasure(BenchMetric.flops, flops)],
        )

    # Optional correctness cross-check against cuDNN.
    if verify:
        comptime if has_amd_gpu_accelerator():
            conv_miopen(
                input_tt,
                filter_qrscf_tt,
                TileTensor(output_ref_dev, output_tt.layout),
                stride_idx,
                dilation_idx,
                pad_idx,
                1,
                ctx,
            )
        else:
            var output_ref_buf = LayoutTensor[dtype, output_layout](
                output_ref_dev.unsafe_ptr(), output_runtime_layout
            )
            conv3d_cudnn(
                input_buf,
                filter_fcqrs_buf,
                output_ref_buf,
                stride_idx,
                dilation_idx,
                pad_idx,
                1,
                ctx,
            )
        var output_host = List(length=output_size, fill=Scalar[dtype](0))
        var output_ref_host = List(length=output_size, fill=Scalar[dtype](0))
        ctx.enqueue_copy(output_host, output_dev)
        ctx.enqueue_copy(output_ref_host, output_ref_dev)
        ctx.synchronize()
        var max_diff: Float32 = 0.0
        for i in range(output_size):
            var a = output_host[i].cast[.float32]()
            var c = output_ref_host[i].cast[.float32]()
            var d = abs(a - c)
            if d > max_diff:
                max_diff = d
        print("verify max |", impl, " - cuDNN| = ", max_diff, sep="")
        _ = output_host^
        _ = output_ref_host^

    _ = input_dev^
    _ = filter_qrscf_dev^
    _ = filter_fcqrs_dev^
    _ = output_dev^
    _ = output_ref_dev^
    _ = filter_fcqrs_host^
    _ = filter_qrscf_host^
    _ = input_host^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime N = get_defined_int["N", 1]()
    comptime D = get_defined_int["D", 21]()
    comptime H = get_defined_int["H", 30]()
    comptime W = get_defined_int["W", 52]()
    comptime C_in = get_defined_int["C_in", 16]()
    comptime C_out = get_defined_int["C_out", 384]()
    comptime Q = get_defined_int["Q", 3]()
    comptime R = get_defined_int["R", 3]()
    comptime S = get_defined_int["S", 3]()

    var label = arg_parse("label", String("conv3d"))
    # impl selects the kernel path at runtime so one compiled binary covers
    # all (naive, im2col, 1x1x1, qslice, native_3d, cudnn) options.
    var impl = arg_parse("impl", String("naive"))
    var verify = arg_parse("verify", False)
    # pad / stride flow into the kernels as runtime IndexLists, so reading
    # them via arg_parse instead of get_defined_int avoids spinning up a
    # fresh compiled binary per (pad, stride) combination.
    var pad_d = arg_parse("pad_d", 1)
    var pad_h = arg_parse("pad_h", 1)
    var pad_w = arg_parse("pad_w", 1)
    var stride_d = arg_parse("stride_d", 1)
    var stride_h = arg_parse("stride_h", 1)
    var stride_w = arg_parse("stride_w", 1)
    var warmup_iters = arg_parse("warmup_iters", 1)
    var max_iters = arg_parse("max_iters", 5)
    var max_runtime_secs = arg_parse("max_runtime_secs", 5.0)

    var m = Bench(
        BenchConfig(
            num_repetitions=1,
            num_warmup_iters=warmup_iters,
            max_iters=max_iters,
            max_runtime_secs=max_runtime_secs,
        )
    )
    with DeviceContext() as ctx:
        bench_conv3d[
            dtype,
            N,
            D,
            H,
            W,
            C_in,
            C_out,
            Q,
            R,
            S,
        ](
            ctx,
            m,
            label,
            impl,
            verify,
            pad_d,
            pad_h,
            pad_w,
            stride_d,
            stride_h,
            stride_w,
        )
    m.dump_report()
