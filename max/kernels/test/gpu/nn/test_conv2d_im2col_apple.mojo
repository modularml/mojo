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
"""Apple M5: NHWC conv2d via im2col + `_matmul_gpu` (-> `AppleM5MatMul`).

Discovery / correctness harness for the FLUX.2 VAE decoder unblock. Drives
`dispatch_im2col_matmul_conv2d` directly on diffusion-VAE-shaped 2-D convs
(3x3 stride-1 same-pad is the common case) and the full `conv_gpu` dispatch,
each checked against a CPU naive fp32-accumulator reference narrowed to the
kernel dtype.

Target: Apple silicon M5 (compute_capability == 5), bf16/fp16.
Run with: `mojo max/kernels/test/gpu/nn/test_conv2d_im2col_apple.mojo`.
"""

from std.math import ceildiv, round

from layout import (
    Coord,
    Idx,
    LTToTTLayout,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from max.gpu.host import DeviceContext
from nn.conv.conv import Naive2dConvolution, conv_gpu
from nn.conv.conv_utils import elementwise_simd_epilogue_type
from nn.conv.gpu.im2col_matmul_2d import (
    dispatch_fused_im2col_conv2d_apple,
    dispatch_im2col_matmul_conv2d,
)
from std.sys import align_of
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


# Opaque identity barrier: keep the conv dims runtime-valued so the fused-conv
# kernel is JIT-compiled with full dynamic index math (no static-shape constant
# folding) -- the exact instantiation FLUX.2's [dyn,dyn,dyn,*] conv hits.
@no_inline
def _dyn(x: Int) -> Int:
    return x


def test_conv2d_fused_apple_dynamic_round(
    ctx: DeviceContext, name: StaticString
) raises:
    """Dynamic-shape fused conv with a `round`-bearing fused epilogue.

    Regression for the FLUX.2 VAE decoder conv_out device-codegen crash. The
    crash was NOT the conv kernel: `pop.round` (the `mo.round` in conv_out's
    fused 9-op epilogue) lowered to `llvm.roundeven`, which Apple's Metal/AIR
    backend cannot JIT -- MTLCompilerService died with
    `XPC_ERROR_CONNECTION_INTERRUPTED` at compute-pipeline-state creation. The
    fix lowers `pop.round` to `llvm.rint` on Metal (round-to-nearest-even in the
    default FP mode -- numerically identical to `roundeven`).

    Unlike the other tests here, the operands are built with RUNTIME (`_dyn`)
    dims so M/N/K are dynamic inside the kernel (the static-shape path folds the
    index math and dodges the codegen path FLUX takes). The epilogue mirrors the
    FLUX conv_out tail: cast f32 -> mul -> add -> relu -> min(255) -> round, then
    store. Both the device JIT succeeding AND the numeric match vs a CPU fp32
    reference are checked. C_in=128 -> C_out=3 (RGB), 3x3 same-pad.
    """
    print("== FUSED DYN+round ", name, " ==")
    var batch = _dyn(1)
    var H = _dyn(48)
    var W = _dyn(48)
    var C_in = _dyn(128)
    var C_out = _dyn(3)
    var R = _dyn(3)
    var S = _dyn(3)
    var pad = 1
    var stride = 1

    var H_out = (H + 2 * pad - (R - 1) - 1) // stride + 1
    var W_out = (W + 2 * pad - (S - 1) - 1) // stride + 1

    var input_size = batch * H * W * C_in
    var filter_size = R * S * C_in * C_out
    var output_size = batch * H_out * W_out * C_out

    var input_host = ctx.enqueue_create_host_buffer[.bfloat16](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[.bfloat16](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[.bfloat16](output_size)
    ctx.synchronize()

    for i in range(input_size):
        var t = Float64(i % 17) / 17.0
        input_host[i] = BFloat16(t * 1.0 - 0.5)
    for i in range(filter_size):
        var t = Float64(i % 13) / 13.0
        filter_host[i] = BFloat16(t * 0.5 - 0.25)

    var input_dev = ctx.enqueue_create_buffer[.bfloat16](input_size)
    var filter_dev = ctx.enqueue_create_buffer[.bfloat16](filter_size)
    var output_dev = ctx.enqueue_create_buffer[.bfloat16](output_size)
    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    # Dynamic (runtime-shaped) NHWC input + RSCF filter + NHWC output.
    var input_tt = TileTensor(
        input_dev.unsafe_ptr(), row_major(batch, H, W, C_in)
    ).as_immut()
    var filter_tt = TileTensor(
        filter_dev.unsafe_ptr(), row_major(R, S, C_in, C_out)
    ).as_immut()
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(), row_major(batch, H_out, W_out, C_out)
    )

    var out_ptr = output_dev.unsafe_ptr()
    var W_out_i = W_out
    var HW_out_i = H_out * W_out
    var C_out_i = C_out

    @__parameter
    @always_inline
    @__copy_capture(out_ptr, W_out_i, HW_out_i, C_out_i)
    def round_epilogue[
        _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
        # FLUX conv_out-style tail: scale, shift, clamp to [0,255], round.
        var x = val.cast[.float32]()
        x = x * 8.0 + 4.0
        x = max(x, 0.0)
        x = min(x, 255.0)
        x = round(x)
        var y = x.cast[_dtype]()
        var b = rebind[IndexList[4]](coords)
        var flat = (b[0] * HW_out_i + b[1] * W_out_i + b[2]) * C_out_i + b[3]
        (out_ptr + flat).store(rebind[SIMD[.bfloat16, _width]](y))

    var handled = dispatch_fused_im2col_conv2d_apple[
        maybe_epilogue_func=Optional[elementwise_simd_epilogue_type](
            round_epilogue
        ),
    ](
        input_tt,
        filter_tt,
        output_tt,
        IndexList[2](stride, stride),
        IndexList[2](1, 1),
        IndexList[2](pad, pad),
        1,
        ctx,
    )
    if not handled:
        if ctx.compute_capability() < 5:
            print("  SKIP: fused conv requires M5+")
            return
        raise Error("dynamic-shape fused conv unexpectedly declined")

    ctx.synchronize()
    ctx.enqueue_copy(output_gpu_host, output_dev)
    ctx.synchronize()

    # CPU fp32 reference with the same epilogue (round-to-nearest-even).
    for b in range(batch):
        for ho in range(H_out):
            for wo in range(W_out):
                for co in range(C_out):
                    var acc = Float32(0)
                    for r in range(R):
                        var hi = ho * stride - pad + r
                        for s in range(S):
                            var wi = wo * stride - pad + s
                            if 0 <= hi < H and 0 <= wi < W:
                                for ci in range(C_in):
                                    var a = Float32(
                                        input_host[
                                            ((b * H + hi) * W + wi) * C_in + ci
                                        ]
                                    )
                                    var f = Float32(
                                        filter_host[
                                            ((r * S + s) * C_in + ci) * C_out
                                            + co
                                        ]
                                    )
                                    acc += a * f
                    var x = acc * 8.0 + 4.0
                    x = max(x, Float32(0))
                    x = min(x, Float32(255))
                    x = round(x)
                    var ref_val = Float32(x)
                    var flat = ((b * H_out + ho) * W_out + wo) * C_out + co
                    var got = Float32(output_gpu_host[flat])
                    # `round` is discontinuous, so a sub-ULP difference between
                    # the GPU's bf16 accumulation and this fp32 reference can
                    # land on opposite sides of a `.5` tie -- a legitimate +-1
                    # post-round difference. Assert within 1 (this guards the
                    # device codegen + that `round` rounds, without demanding
                    # bit-exact accumulation across two different reductions).
                    if abs(got - ref_val) > 1.0:
                        raise Error(
                            "round-epilogue conv mismatch: got=",
                            got,
                            " ref=",
                            ref_val,
                            " at flat=",
                            flat,
                        )
    print("  RESULT: PASS (device JIT + round numerics)")

    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def test_conv2d_im2col_direct[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[2],
    pad: IndexList[2],
    with_epilogue: Bool,
    rtol: Float64 = 2e-2,
    atol: Float64 = 2e-2,
](ctx: DeviceContext, name: StaticString) raises:
    """Drive `dispatch_im2col_matmul_conv2d` directly and compare to CPU naive.

    NHWC input, RSCF filter, dilation=1, num_groups=1. `with_epilogue` adds a
    scale-by-2 fused elementwise lambda (the production MOGG always supplies an
    output_fn), which also exercises the 4-D NHWC coord-unpack closure.
    """
    print("== ", name, " (with_epilogue=", with_epilogue, ") ==")
    comptime N = Int(input_layout.shape[0])
    comptime H = Int(input_layout.shape[1])
    comptime W = Int(input_layout.shape[2])
    comptime C = Int(input_layout.shape[3])

    comptime R = Int(filter_layout.shape[0])
    comptime S = Int(filter_layout.shape[1])
    comptime F = Int(filter_layout.shape[3])

    comptime pad_h = IndexList[2](pad[0], pad[0])
    comptime pad_w = IndexList[2](pad[1], pad[1])

    comptime H_out = (H + pad_h[0] + pad_h[1] - (R - 1) - 1) // stride[0] + 1
    comptime W_out = (W + pad_w[0] + pad_w[1] - (S - 1) - 1) // stride[1] + 1

    comptime output_layout = Layout.row_major(N, H_out, W_out, F)

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (output_layout.size())

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_ref_host = ctx.enqueue_create_host_buffer[dtype](output_size)

    # Deterministic small values (not RNG): keeps every element O(1) so the
    # fp32-accumulator reference and the bf16 GPU result stay inside a sane
    # relative tolerance, and keeps the test reproducible.
    for i in range(input_size):
        var t = Float64(i % 17) / 17.0
        input_host[i] = Scalar[dtype](t * 1.0 - 0.5)
    for i in range(filter_size):
        var t = Float64(i % 13) / 13.0
        filter_host[i] = Scalar[dtype](t * 0.5 - 0.25)

    # CPU reference at fp32 accumulator (matches the GPU matmul), narrowed back.
    comptime accum_dtype = DType.float32 if dtype == DType.bfloat16 else dtype
    var output_ref_accum_host = ctx.enqueue_create_host_buffer[accum_dtype](
        output_size
    )
    # Naive2dConvolution uses 5-D NDHWC with D=Q=1 for a 2-D conv.
    Naive2dConvolution[accum_dtype, dtype, dtype].run(
        output_ref_accum_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Index(N, 1, H_out, W_out, F),
        Index(N, 1, H, W, C),
        Index(1, R, S, C, F),
        IndexList[2](0, 0),
        pad_h,
        pad_w,
        IndexList[3](1, stride[0], stride[1]),
        IndexList[3](1, 1, 1),
        1,
    )
    var scale = Scalar[accum_dtype](2.0) if with_epilogue else Scalar[
        accum_dtype
    ](1.0)
    for i in range(output_size):
        output_ref_host[i] = (output_ref_accum_host[i] * scale).cast[dtype]()

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    var output_lt = LayoutTensor[dtype, output_layout](output_dev.unsafe_ptr())
    var input_tt = TileTensor(
        input_dev.unsafe_ptr(), LTToTTLayout[input_layout]()
    )
    var filter_tt = TileTensor(
        filter_dev.unsafe_ptr(), LTToTTLayout[filter_layout]()
    )
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(), LTToTTLayout[output_layout]()
    )

    var handled: Bool
    comptime if with_epilogue:

        @__parameter
        @always_inline
        @__copy_capture(output_lt)
        def scale_epilogue[
            _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            var scaled = (val.cast[.float32]() * 2.0).cast[dtype]()
            output_lt.store[
                width=_width, store_alignment=align_of[dtype]() * _alignment
            ](
                rebind[IndexList[4]](coords),
                rebind[SIMD[dtype, _width]](scaled),
            )

        handled = dispatch_im2col_matmul_conv2d[
            maybe_epilogue_func=Optional[elementwise_simd_epilogue_type](
                scale_epilogue
            ),
        ](
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )
    else:
        handled = dispatch_im2col_matmul_conv2d(
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )

    if not handled:
        print("  SKIP: dispatcher declined this shape (1x1 / K<16 / N<16)")
        _ = input_dev^
        _ = filter_dev^
        _ = output_dev^
        return

    ctx.synchronize()
    ctx.enqueue_copy(output_gpu_host, output_dev)
    ctx.synchronize()

    for i in range(output_size):
        assert_almost_equal(
            output_ref_host[i], output_gpu_host[i], rtol=rtol, atol=atol
        )
    print("  RESULT: PASS")

    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def test_conv2d_fused_apple[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[2],
    pad: IndexList[2],
    with_epilogue: Bool,
    rtol: Float64 = 2e-2,
    atol: Float64 = 2e-2,
](ctx: DeviceContext, name: StaticString) raises:
    """Drive `dispatch_fused_im2col_conv2d_apple` directly; compare to CPU naive.

    Same NHWC input / RSCF filter / fp32-accum reference as
    `test_conv2d_im2col_direct`, but exercises the FUSED online-im2col conv path
    (`AppleM5MatMul.run_conv`) -- no `[M, K]` scratch materialised. Result must
    match the same CPU reference the materialised path matches, proving the
    fused gather + OOB zero-fill reproduces the im2col matrix bit-for-bit
    (within bf16 tolerance).
    """
    print("== FUSED ", name, " (with_epilogue=", with_epilogue, ") ==")
    comptime N = Int(input_layout.shape[0])
    comptime H = Int(input_layout.shape[1])
    comptime W = Int(input_layout.shape[2])
    comptime C = Int(input_layout.shape[3])

    comptime R = Int(filter_layout.shape[0])
    comptime S = Int(filter_layout.shape[1])
    comptime F = Int(filter_layout.shape[3])

    comptime pad_h = IndexList[2](pad[0], pad[0])
    comptime pad_w = IndexList[2](pad[1], pad[1])

    comptime H_out = (H + pad_h[0] + pad_h[1] - (R - 1) - 1) // stride[0] + 1
    comptime W_out = (W + pad_w[0] + pad_w[1] - (S - 1) - 1) // stride[1] + 1

    comptime output_layout = Layout.row_major(N, H_out, W_out, F)

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (output_layout.size())

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_ref_host = ctx.enqueue_create_host_buffer[dtype](output_size)

    for i in range(input_size):
        var t = Float64(i % 17) / 17.0
        input_host[i] = Scalar[dtype](t * 1.0 - 0.5)
    for i in range(filter_size):
        var t = Float64(i % 13) / 13.0
        filter_host[i] = Scalar[dtype](t * 0.5 - 0.25)

    comptime accum_dtype = DType.float32 if dtype == DType.bfloat16 else dtype
    var output_ref_accum_host = ctx.enqueue_create_host_buffer[accum_dtype](
        output_size
    )
    Naive2dConvolution[accum_dtype, dtype, dtype].run(
        output_ref_accum_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Index(N, 1, H_out, W_out, F),
        Index(N, 1, H, W, C),
        Index(1, R, S, C, F),
        IndexList[2](0, 0),
        pad_h,
        pad_w,
        IndexList[3](1, stride[0], stride[1]),
        IndexList[3](1, 1, 1),
        1,
    )
    var scale = Scalar[accum_dtype](2.0) if with_epilogue else Scalar[
        accum_dtype
    ](1.0)
    for i in range(output_size):
        output_ref_host[i] = (output_ref_accum_host[i] * scale).cast[dtype]()

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    var output_lt = LayoutTensor[dtype, output_layout](output_dev.unsafe_ptr())
    var input_tt = TileTensor(
        input_dev.unsafe_ptr(), LTToTTLayout[input_layout]()
    )
    var filter_tt = TileTensor(
        filter_dev.unsafe_ptr(), LTToTTLayout[filter_layout]()
    )
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(), LTToTTLayout[output_layout]()
    )

    var handled: Bool
    comptime if with_epilogue:

        @__parameter
        @always_inline
        @__copy_capture(output_lt)
        def scale_epilogue[
            _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
        ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
            var scaled = (val.cast[.float32]() * 2.0).cast[dtype]()
            output_lt.store[
                width=_width, store_alignment=align_of[dtype]() * _alignment
            ](
                rebind[IndexList[4]](coords),
                rebind[SIMD[dtype, _width]](scaled),
            )

        handled = dispatch_fused_im2col_conv2d_apple[
            maybe_epilogue_func=Optional[elementwise_simd_epilogue_type](
                scale_epilogue
            ),
        ](
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )
    else:
        handled = dispatch_fused_im2col_conv2d_apple(
            input_tt,
            filter_tt,
            output_tt,
            stride,
            IndexList[2](1, 1),
            pad,
            1,
            ctx,
        )

    if not handled:
        print("  SKIP: fused dispatcher declined (1x1 / K<16, or not Apple M5)")
        _ = input_dev^
        _ = filter_dev^
        _ = output_dev^
        return

    ctx.synchronize()
    ctx.enqueue_copy(output_gpu_host, output_dev)
    ctx.synchronize()

    for i in range(output_size):
        assert_almost_equal(
            output_ref_host[i], output_gpu_host[i], rtol=rtol, atol=atol
        )
    print("  RESULT: PASS")

    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def test_conv2d_gpu_dispatch[
    input_layout: Layout,
    filter_layout: Layout,
    dtype: DType,
    stride: IndexList[2],
    pad: IndexList[2],
    rtol: Float64 = 2e-2,
    atol: Float64 = 2e-2,
](ctx: DeviceContext, name: StaticString) raises:
    """Exercise the full `conv_gpu` dispatch with a scale-by-2 epilogue.

    This is the end-to-end check that the Apple branch added to `conv_gpu`'s
    2-D `input_lt.rank == 4` arm actually routes to
    `dispatch_im2col_matmul_conv2d` (-> `AppleM5MatMul`) and produces correct
    results -- the functional FLUX-VAE unblock. Mirrors the production MOGG
    wiring where an `output_fn` is always supplied. NHWC input, RSCF filter.
    """
    print("== conv_gpu dispatch: ", name, " ==")
    comptime N = Int(input_layout.shape[0])
    comptime H = Int(input_layout.shape[1])
    comptime W = Int(input_layout.shape[2])
    comptime C = Int(input_layout.shape[3])

    comptime R = Int(filter_layout.shape[0])
    comptime S = Int(filter_layout.shape[1])
    comptime F = Int(filter_layout.shape[3])

    comptime pad_h = IndexList[2](pad[0], pad[0])
    comptime pad_w = IndexList[2](pad[1], pad[1])

    comptime H_out = (H + pad_h[0] + pad_h[1] - (R - 1) - 1) // stride[0] + 1
    comptime W_out = (W + pad_w[0] + pad_w[1] - (S - 1) - 1) // stride[1] + 1

    comptime output_layout = Layout.row_major(N, H_out, W_out, F)

    var input_size = comptime (input_layout.size())
    var filter_size = comptime (filter_layout.size())
    var output_size = comptime (output_layout.size())

    var input_host = ctx.enqueue_create_host_buffer[dtype](input_size)
    var filter_host = ctx.enqueue_create_host_buffer[dtype](filter_size)
    var output_gpu_host = ctx.enqueue_create_host_buffer[dtype](output_size)
    var output_ref_host = ctx.enqueue_create_host_buffer[dtype](output_size)

    for i in range(input_size):
        var t = Float64(i % 17) / 17.0
        input_host[i] = Scalar[dtype](t * 1.0 - 0.5)
    for i in range(filter_size):
        var t = Float64(i % 13) / 13.0
        filter_host[i] = Scalar[dtype](t * 0.5 - 0.25)

    comptime accum_dtype = DType.float32 if dtype == DType.bfloat16 else dtype
    var output_ref_accum_host = ctx.enqueue_create_host_buffer[accum_dtype](
        output_size
    )
    Naive2dConvolution[accum_dtype, dtype, dtype].run(
        output_ref_accum_host.unsafe_ptr(),
        input_host.unsafe_ptr(),
        filter_host.unsafe_ptr(),
        Index(N, 1, H_out, W_out, F),
        Index(N, 1, H, W, C),
        Index(1, R, S, C, F),
        IndexList[2](0, 0),
        pad_h,
        pad_w,
        IndexList[3](1, stride[0], stride[1]),
        IndexList[3](1, 1, 1),
        1,
    )
    for i in range(output_size):
        # Scale by 2 to match the GPU epilogue.
        output_ref_host[i] = (
            output_ref_accum_host[i] * Scalar[accum_dtype](2.0)
        ).cast[dtype]()

    var input_dev = ctx.enqueue_create_buffer[dtype](input_size)
    var filter_dev = ctx.enqueue_create_buffer[dtype](filter_size)
    var output_dev = ctx.enqueue_create_buffer[dtype](output_size)

    ctx.enqueue_copy(input_dev, input_host)
    ctx.enqueue_copy(filter_dev, filter_host)

    var output_lt = LayoutTensor[dtype, output_layout](output_dev.unsafe_ptr())
    var input_tt = TileTensor(
        input_dev.unsafe_ptr(), LTToTTLayout[input_layout]()
    )
    var filter_tt = TileTensor(
        filter_dev.unsafe_ptr(), LTToTTLayout[filter_layout]()
    )
    var output_tt = TileTensor(
        output_dev.unsafe_ptr(), LTToTTLayout[output_layout]()
    )

    @__parameter
    @always_inline
    @__copy_capture(output_lt)
    def scale_epilogue[
        _dtype: DType, _rank: Int, _width: SIMDLength, _alignment: Int = 1
    ](coords: IndexList[_rank], val: SIMD[_dtype, _width]):
        var scaled = (val.cast[.float32]() * 2.0).cast[dtype]()
        output_lt.store[
            width=_width, store_alignment=align_of[dtype]() * _alignment
        ](
            rebind[IndexList[4]](coords),
            rebind[SIMD[dtype, _width]](scaled),
        )

    conv_gpu[
        dtype,
        dtype,
        dtype,
        Optional[elementwise_simd_epilogue_type](scale_epilogue),
    ](
        input_tt,
        filter_tt,
        output_tt,
        stride,
        IndexList[2](1, 1),
        IndexList[4](pad[0], pad[0], pad[1], pad[1]),
        1,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(output_gpu_host, output_dev)
    ctx.synchronize()

    for i in range(output_size):
        assert_almost_equal(
            output_ref_host[i], output_gpu_host[i], rtol=rtol, atol=atol
        )
    print("  RESULT: PASS")

    _ = input_dev^
    _ = filter_dev^
    _ = output_dev^


def main() raises:
    with DeviceContext() as ctx:
        # ---- 3x3 stride-1 same-pad: the FLUX VAE common case ----
        # Small, no epilogue.
        test_conv2d_im2col_direct[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=False,
        ](ctx, "bf16 3x3 s1 same-pad C64->64")

        # Same, with scale-by-2 epilogue (production MOGG shape).
        test_conv2d_im2col_direct[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C64->64")

        # Larger channel count, batch=1 (FLUX VAE mid-block scale).
        test_conv2d_im2col_direct[
            Layout.row_major(1, 32, 32, 128),
            Layout.row_major(3, 3, 128, 128),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C128->128")

        # Non-128-aligned channels (the case the SM100 fast path declines).
        test_conv2d_im2col_direct[
            Layout.row_major(1, 32, 32, 96),
            Layout.row_major(3, 3, 96, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C96->96")

        # fp16 coverage.
        test_conv2d_im2col_direct[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.float16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "fp16 3x3 s1 same-pad C64->64")

        # Channel up-projection 64 -> 128 (asymmetric N).
        test_conv2d_im2col_direct[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(3, 3, 64, 128),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C64->128")

        # stride=2 downsample (VAE encoder side, exercises stride math).
        test_conv2d_im2col_direct[
            Layout.row_major(1, 32, 32, 64),
            Layout.row_major(3, 3, 64, 128),
            DType.bfloat16,
            IndexList[2](2, 2),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s2 C64->128")

        # batch=2 (coord-unpack across batch).
        test_conv2d_im2col_direct[
            Layout.row_major(2, 16, 16, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C64->64 N=2")

        # ---- End-to-end conv_gpu dispatch (exercises the Apple branch) ----
        # The FLUX VAE common case through the full dispatch path.
        test_conv2d_gpu_dispatch[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "bf16 3x3 s1 same-pad C64->64")

        test_conv2d_gpu_dispatch[
            Layout.row_major(1, 32, 32, 96),
            Layout.row_major(3, 3, 96, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "bf16 3x3 s1 same-pad C96->96")

        test_conv2d_gpu_dispatch[
            Layout.row_major(2, 16, 16, 64),
            Layout.row_major(3, 3, 64, 128),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
        ](ctx, "bf16 3x3 s1 same-pad C64->128 N=2")

        # 1x1 conv: dispatcher declines, conv_gpu falls through to the naive
        # kernel. End-to-end correctness must still hold via that fallback.
        test_conv2d_gpu_dispatch[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(1, 1, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](0, 0),
        ](ctx, "bf16 1x1 C64->64 (naive fallback)")

        # ---- Fused online-im2col path (AppleM5MatMul.run_conv) ----
        # Small case (exercises ragged-M / partial-tile bounds at tiny M).
        test_conv2d_fused_apple[
            Layout.row_major(1, 16, 16, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=False,
        ](ctx, "bf16 3x3 s1 same-pad C64->64")

        # FLUX-VAE common case with the scale-by-2 epilogue (production shape).
        test_conv2d_fused_apple[
            Layout.row_major(1, 32, 32, 128),
            Layout.row_major(3, 3, 128, 128),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C128->128")

        # Non-128-aligned channels (K=R*S*C not a multiple of 16-friendly N).
        test_conv2d_fused_apple[
            Layout.row_major(1, 32, 32, 96),
            Layout.row_major(3, 3, 96, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C96->96")

        # stride=2 downsample (stride math in the gather).
        test_conv2d_fused_apple[
            Layout.row_major(1, 32, 32, 64),
            Layout.row_major(3, 3, 64, 128),
            DType.bfloat16,
            IndexList[2](2, 2),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s2 C64->128")

        # batch=2 (M-decomposition across batch in the gather).
        test_conv2d_fused_apple[
            Layout.row_major(2, 16, 16, 64),
            Layout.row_major(3, 3, 64, 64),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C64->64 N=2")

        # Previously-LOSING memory-bound shape: low C_out at high spatial
        # resolution (C32->32 @ 256x256). The materialised path's [M,K] scratch
        # round-trip lost to naive here (~0.28x); the fused gather avoids it.
        # Correctness must hold -- this is the shape the memory-bound guard used
        # to decline.
        test_conv2d_fused_apple[
            Layout.row_major(1, 256, 256, 32),
            Layout.row_major(3, 3, 32, 32),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C32->32 @256x256 (mem-bound)")

        # ---- Small C_out (the N<16 gate the fused path drops) ----
        # FLUX.2 VAE decoder FINAL conv: C_in=128 -> C_out=3 (RGB), 3x3 same-pad
        # at high spatial. C_out=3 < SG_N=32 so this is an N-edge tile; correct
        # output proves the `b_valid_cols`/`acol < n` masking handles tiny N.
        # This is the e2e blocker -- the materialised path declined (N<16) and
        # the naive fallback is broken on Metal for this shape.
        test_conv2d_fused_apple[
            Layout.row_major(1, 256, 256, 128),
            Layout.row_major(3, 3, 128, 3),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C128->3 @256x256 (FLUX VAE->RGB)")

        # Same FLUX VAE->RGB shape without the fused epilogue (fast-path store
        # bounded write at tiny N -- exercises `_apply_epilogue[bounded]` vs the
        # cast-store path; bf16 out still routes through the epilogue closure).
        test_conv2d_fused_apple[
            Layout.row_major(1, 256, 256, 128),
            Layout.row_major(3, 3, 128, 3),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=False,
        ](ctx, "bf16 3x3 s1 same-pad C128->3 @256x256 (no epilogue)")

        # C_out=1: the single-column N edge (valid_cols=1; only `acol=0` writes).
        test_conv2d_fused_apple[
            Layout.row_major(1, 64, 64, 128),
            Layout.row_major(3, 3, 128, 1),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C128->1 @64x64")

        # C_out=8: half a 16-wide MMA_N tile (valid_cols=8; straddles the 4-wide
        # vectorized store boundary in the bounded epilogue).
        test_conv2d_fused_apple[
            Layout.row_major(1, 64, 64, 128),
            Layout.row_major(3, 3, 128, 8),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 same-pad C128->8 @64x64")

        # One shape that drives every otherwise-untested width-8 edge at once,
        # chosen so the params aren't arbitrary: C_in=20 (not a multiple of 8,
        # < BK=32) hits the channel-straddle slow path AND the multi-iteration
        # K-state carry; K=R*S*C_in=180 (not a multiple of 32) hits the partial-K
        # tail; M=17²=289 and N=96 (neither a multiple of 64) make it a ragged-M
        # and ragged-N edge tile.
        test_conv2d_fused_apple[
            Layout.row_major(1, 17, 17, 20),
            Layout.row_major(3, 3, 20, 96),
            DType.bfloat16,
            IndexList[2](1, 1),
            IndexList[2](1, 1),
            with_epilogue=True,
        ](ctx, "bf16 3x3 s1 C20->96 @17x17 (straddle+multicarry+tail+ragged)")

        # ---- Dynamic-shape conv with a `round` epilogue (FLUX VAE unblock) ----
        # The exact device-codegen crash repro: DYNAMIC M/N/K + a `round`-bearing
        # fused epilogue. `pop.round` -> `llvm.roundeven` crashed Apple's Metal
        # compiler; the fix lowers it to `llvm.rint`. Static-shape tests above
        # fold the index math and dodge the codegen path, so this dynamic case is
        # the one that guards the regression.
        test_conv2d_fused_apple_dynamic_round(
            ctx, "bf16 3x3 s1 same-pad C128->3 dyn-shape +round"
        )

        print("ALL TESTS DONE")
